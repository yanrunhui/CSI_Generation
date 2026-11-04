#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.data.delay_power_residual import DelayPowerResidualCacheDataset
from physics_csi.models.scene_delay_power_residual import SceneConditionedDelayPowerResidual
from physics_csi.models.scene_link_global_aod_residual import (
    SceneConditionedLinkGlobalAoDResidual,
)
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig
from physics_csi.training.trainer import move_to_device, resolve_device
from physics_csi.utils.io import atomic_json_dump
from scripts.diagnose_path_count_control import (
    _balanced_indices,
    _condition_control,
    _fidelity_metrics,
    _power_aod_path,
    _power_aod_link,
    _safe_summary,
    _standardized_swd,
    _tx_spatial_relative_frobenius,
)


PRIMARY_METRICS = (
    "path_power_aod_swd",
    "link_power_aod_swd",
    "tx_spatial_correlation_relative_frobenius",
)
NONREGRESSION_METRICS = (
    "path_delay_power_swd",
    "link_delay_power_swd",
    "pdp_swd",
    "phase_invariant_full_csi_projected_swd",
)


def _parse_cache(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("cache must be GENERATION_SEED=PATH")
    raw_seed, raw_path = value.split("=", 1)
    try:
        seed = int(raw_seed)
    except ValueError as error:
        raise argparse.ArgumentTypeError("cache seed must be an integer") from error
    return seed, Path(raw_path)


def _parse_alphas(value: str) -> tuple[float, ...]:
    try:
        alphas = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("alphas must be comma-separated numbers") from error
    if not alphas or any(not math.isfinite(alpha) or alpha < 0 for alpha in alphas):
        raise argparse.ArgumentTypeError("alphas must be finite and non-negative")
    if len(set(alphas)) != len(alphas):
        raise argparse.ArgumentTypeError("alphas must be unique")
    if 1.0 not in alphas:
        raise argparse.ArgumentTypeError("alphas must include the current-model baseline 1.0")
    return alphas


def _alpha_key(alpha: float) -> str:
    return f"{alpha:.8g}"


def _subset(payload: Mapping[str, Any], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    sample_count = int(
        next(value for value in payload.values() if isinstance(value, torch.Tensor)).shape[0]
    )
    return {
        key: value[indices]
        for key, value in payload.items()
        if isinstance(value, torch.Tensor) and value.shape[:1] == (sample_count,)
    }


def _append(chunks: dict[str, list[torch.Tensor]], payload: Mapping[str, torch.Tensor]) -> None:
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            chunks.setdefault(key, []).append(value.detach().cpu())


def _finish(chunks: Mapping[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat(values, dim=0) for key, values in chunks.items()}


def _synthesize(
    payload: Mapping[str, torch.Tensor], synthesizer: MIMOOFDMSynthesizer
) -> torch.Tensor:
    return synthesizer(
        payload["path_delays_ns"],
        payload["path_powers_linear"],
        payload["path_phases_rad"],
        payload["aoa_az_deg"],
        payload["aoa_el_deg"],
        payload["aod_az_deg"],
        payload["aod_el_deg"],
        payload["path_mask"],
    )


def _load_dp(
    checkpoint: Mapping[str, Any], device: torch.device
) -> SceneConditionedDelayPowerResidual:
    model = SceneConditionedDelayPowerResidual(**checkpoint["model_init"]).to(device)
    model.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))
    return model.requires_grad_(False).eval()


def _load_link(
    checkpoint: Mapping[str, Any], device: torch.device
) -> SceneConditionedLinkGlobalAoDResidual:
    if checkpoint.get("model_kind") != "scene_link_global_aod_residual":
        raise ValueError("--link-checkpoint is not a Link-global AoD residual")
    model = SceneConditionedLinkGlobalAoDResidual(**checkpoint["model_init"]).to(device)
    model.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))
    return model.requires_grad_(False).eval()


def _scaled_link_payload(
    dp_payload: Mapping[str, torch.Tensor],
    output: Any,
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Scale the final shared Link-global residual, not its learned gate logit."""
    result = {
        key: value for key, value in dp_payload.items() if isinstance(value, torch.Tensor)
    }
    mask = dp_payload["path_mask"].bool()
    azimuth = torch.deg2rad(dp_payload["aod_az_deg"].float())
    azimuth = azimuth + torch.deg2rad(output.azimuth_residual_deg * float(alpha))
    azimuth = torch.atan2(azimuth.sin(), azimuth.cos())
    elevation = (
        dp_payload["aod_el_deg"].float()
        + output.elevation_residual_deg * float(alpha)
    ).clamp(-90.0, 90.0)
    result["aod_az_deg"] = torch.rad2deg(azimuth) * mask
    result["aod_el_deg"] = elevation * mask
    return result


def _assert_cache_contract(
    cache: Mapping[str, Any],
    dp_checkpoint: Mapping[str, Any],
    link_checkpoint: Mapping[str, Any],
) -> None:
    for checkpoint in (dp_checkpoint, link_checkpoint):
        contract = checkpoint["cache_contract"]
        if contract["scene_plan_fingerprint"] != cache["scene_plan_fingerprint"]:
            raise ValueError("checkpoint and cache scene plans differ")
        if contract["b3_checkpoint_sha256"] != cache["b3_checkpoint_sha256"]:
            raise ValueError("checkpoint and cache B3 backbones differ")


def _primary_map_metrics(
    candidate: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    *,
    projections: int,
    seed: int,
) -> dict[str, float]:
    return {
        "path_power_aod_swd": _standardized_swd(
            _power_aod_path(candidate),
            _power_aod_path(reference),
            projections=projections,
            seed=seed,
        ),
        "link_power_aod_swd": _standardized_swd(
            _power_aod_link(candidate),
            _power_aod_link(reference),
            projections=projections,
            seed=seed + 1,
        ),
        "tx_spatial_correlation_relative_frobenius": _tx_spatial_relative_frobenius(
            candidate["csi"], reference["csi"]
        ),
    }


def _evaluate_seed_split(
    *,
    generation_seed: int,
    cache: Mapping[str, Any],
    split: str,
    alphas: tuple[float, ...],
    dp_model: SceneConditionedDelayPowerResidual,
    link_model: SceneConditionedLinkGlobalAoDResidual,
    synthesizer: MIMOOFDMSynthesizer,
    device: torch.device,
    selection_seed: int,
    maximum_samples: int,
    minimum_map_samples: int,
    batch_size: int,
    swd_projections: int,
    map_swd_projections: int,
    projection_dimension: int,
    projection_seeds: tuple[int, ...],
) -> dict[str, Any]:
    dataset = DelayPowerResidualCacheDataset(cache, split)
    path_count = dataset.reference["path_mask"].bool().sum(-1).long()
    all_indices = torch.nonzero(path_count == 5, as_tuple=False).flatten()
    pooled_indices = _balanced_indices(
        all_indices,
        dataset.map_ids,
        maximum_samples,
        selection_seed,
    )
    if pooled_indices.numel() < minimum_map_samples:
        raise ValueError(f"seed {generation_seed} split {split} has insufficient L=5 samples")
    pooled_lookup = torch.zeros(len(dataset), dtype=torch.bool)
    pooled_lookup[pooled_indices] = True
    pooled_reference_chunks: dict[str, list[torch.Tensor]] = {}
    pooled_candidate_chunks: dict[str, dict[str, list[torch.Tensor]]] = {
        _alpha_key(alpha): {} for alpha in alphas
    }
    per_map: dict[str, dict[str, Any]] = {_alpha_key(alpha): {} for alpha in alphas}
    map_support: dict[str, Any] = {}
    raw_azimuth_residuals: list[torch.Tensor] = []
    raw_elevation_residuals: list[torch.Tensor] = []
    with torch.inference_mode():
        for map_id in sorted(int(value) for value in torch.unique(dataset.map_ids[all_indices])):
            map_indices = all_indices[dataset.map_ids[all_indices] == map_id]
            map_support[str(map_id)] = {
                "samples": int(map_indices.numel()),
                "eligible": int(map_indices.numel()) >= minimum_map_samples,
            }
            if map_indices.numel() < minimum_map_samples:
                continue
            local_reference_chunks: dict[str, list[torch.Tensor]] = {}
            local_candidate_chunks: dict[str, dict[str, list[torch.Tensor]]] = {
                _alpha_key(alpha): {} for alpha in alphas
            }
            for start in range(0, map_indices.numel(), batch_size):
                selected = map_indices[start : start + batch_size]
                reference = move_to_device(_subset(dataset.reference, selected), device)
                base = move_to_device(_subset(dataset.base, selected), device)
                conditions = reference["conditions"]
                validity = reference["condition_validity"]
                map_index = link_model.map_indices(reference["map_id"]).to(device)
                dp_output = dp_model(base, conditions, validity, map_index=map_index)
                dp_payload = dp_model.apply_to_payload(base, dp_output)
                link_output = link_model(
                    dp_payload, conditions, validity, map_index=map_index
                )
                reference["csi"] = _synthesize(reference, synthesizer)
                _append(local_reference_chunks, reference)
                active = dp_payload["path_mask"].bool()
                raw_azimuth_residuals.append(
                    link_output.azimuth_residual_deg[active].detach().cpu()
                )
                raw_elevation_residuals.append(
                    link_output.elevation_residual_deg[active].detach().cpu()
                )
                batch_candidates: dict[str, dict[str, torch.Tensor]] = {}
                for alpha in alphas:
                    key = _alpha_key(alpha)
                    candidate = _scaled_link_payload(dp_payload, link_output, alpha)
                    candidate["csi"] = _synthesize(candidate, synthesizer)
                    candidate["map_id"] = reference["map_id"]
                    batch_candidates[key] = candidate
                    _append(local_candidate_chunks[key], candidate)

                keep_pooled = pooled_lookup[selected].to(device)
                if bool(keep_pooled.any()):
                    _append(
                        pooled_reference_chunks,
                        {key: value[keep_pooled] for key, value in reference.items()},
                    )
                    for key, candidate in batch_candidates.items():
                        _append(
                            pooled_candidate_chunks[key],
                            {name: value[keep_pooled] for name, value in candidate.items()},
                        )

            local_reference = {
                key: value.to(device)
                for key, value in _finish(local_reference_chunks).items()
            }
            for alpha in alphas:
                key = _alpha_key(alpha)
                local_candidate = {
                    name: value.to(device)
                    for name, value in _finish(local_candidate_chunks[key]).items()
                }
                per_map[key][str(map_id)] = {
                    "samples": int(map_indices.numel()),
                    **_primary_map_metrics(
                        local_candidate,
                        local_reference,
                        projections=map_swd_projections,
                        seed=selection_seed + map_id,
                    ),
                }

    reference = {
        key: value.to(device)
        for key, value in _finish(pooled_reference_chunks).items()
    }
    candidates = {
        alpha: {
            key: value.to(device) for key, value in _finish(chunks).items()
        }
        for alpha, chunks in pooled_candidate_chunks.items()
    }
    global_metrics: dict[str, Any] = {}
    control_audit: dict[str, Any] = {}
    for alpha in alphas:
        key = _alpha_key(alpha)
        control_audit[key] = _condition_control(
            candidates[key],
            reference["conditions"],
            reference["condition_validity"],
            requested_l=5,
        )
        global_metrics[key] = _fidelity_metrics(
            candidates[key],
            reference,
            swd_projections=swd_projections,
            projection_dimension=projection_dimension,
            projection_seeds=projection_seeds,
            # Identical random projections across alpha candidates make the
            # paired gate comparison insensitive to Monte Carlo SWD noise.
            seed=selection_seed,
        )
    eligible_maps = sum(item["eligible"] for item in map_support.values())
    return {
        "generation_seed": generation_seed,
        "split": split,
        "available_l5_samples": int(all_indices.numel()),
        "per_map_primary_samples": int(
            sum(item["samples"] for item in map_support.values() if item["eligible"])
        ),
        "pooled_auxiliary_samples": int(reference["map_id"].numel()),
        "maps_available": len(map_support),
        "maps_evaluated": int(eligible_maps),
        "map_support": map_support,
        "per_map_primary_uses_all_available_samples": True,
        "pooled_auxiliary_selection": "Map-balanced round-robin without replacement",
        "raw_link_global_residual_deg": {
            "azimuth": _safe_summary(torch.cat(raw_azimuth_residuals)),
            "elevation": _safe_summary(torch.cat(raw_elevation_residuals)),
        },
        "control_audit_on_pooled_subset": control_audit,
        "global_metrics": global_metrics,
        "per_map_primary_metrics": per_map,
    }


def _aggregate_runs(
    runs: list[Mapping[str, Any]], alphas: tuple[float, ...]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for alpha in alphas:
        key = _alpha_key(alpha)
        metric_names = tuple(runs[0]["global_metrics"][key])
        metrics: dict[str, Any] = {}
        for metric in metric_names:
            values = torch.tensor(
                [run["global_metrics"][key][metric] for run in runs],
                dtype=torch.float64,
            )
            metrics[metric] = {
                "mean": float(values.mean()),
                "std_across_generation_seeds": float(values.std(unbiased=False)),
                "per_generation_seed": {
                    str(run["generation_seed"]): run["global_metrics"][key][metric]
                    for run in runs
                },
            }
        result[key] = metrics
    return result


def _macro_primary_aggregate(
    runs: list[Mapping[str, Any]], alphas: tuple[float, ...]
) -> dict[str, Any]:
    """Equal-weight Maps inside each generation seed, then equal-weight seeds."""
    result: dict[str, Any] = {}
    for alpha in alphas:
        key = _alpha_key(alpha)
        result[key] = {}
        for metric in PRIMARY_METRICS:
            per_seed: dict[str, float] = {}
            map_counts: dict[str, int] = {}
            for run in runs:
                values = [
                    item[metric]
                    for item in run["per_map_primary_metrics"][key].values()
                ]
                if not values:
                    raise ValueError("no eligible Maps for macro primary aggregation")
                per_seed[str(run["generation_seed"])] = sum(values) / len(values)
                map_counts[str(run["generation_seed"])] = len(values)
            tensor = torch.tensor(list(per_seed.values()), dtype=torch.float64)
            result[key][metric] = {
                "mean": float(tensor.mean()),
                "std_across_generation_seeds": float(tensor.std(unbiased=False)),
                "per_generation_seed": per_seed,
                "maps_per_generation_seed": map_counts,
            }
    return result


def _composite_hierarchical_bootstrap(
    runs: list[Mapping[str, Any]],
    candidate_alpha: float,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    candidate_key = _alpha_key(candidate_alpha)
    baseline_key = _alpha_key(1.0)
    common_maps = sorted(
        set.intersection(
            *(
                set(run["per_map_primary_metrics"][candidate_key])
                & set(run["per_map_primary_metrics"][baseline_key])
                for run in runs
            )
        )
    )
    if not common_maps:
        return {"status": "no_common_maps"}
    baseline_scale = {
        metric: torch.tensor(
            [
                run["per_map_primary_metrics"][baseline_key][map_id][metric]
                for run in runs
                for map_id in common_maps
            ],
            dtype=torch.float64,
        ).mean().clamp_min(1e-20)
        for metric in PRIMARY_METRICS
    }
    improvement = torch.tensor(
        [
            [
                sum(
                    (
                        run["per_map_primary_metrics"][baseline_key][map_id][metric]
                        - run["per_map_primary_metrics"][candidate_key][map_id][metric]
                    )
                    / float(baseline_scale[metric])
                    for metric in PRIMARY_METRICS
                )
                / len(PRIMARY_METRICS)
                for map_id in common_maps
            ]
            for run in runs
        ],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(seed)
    seed_indices = torch.randint(
        improvement.shape[0],
        (bootstrap_samples, improvement.shape[0]),
        generator=generator,
    )
    map_indices = torch.randint(
        improvement.shape[1],
        (bootstrap_samples, improvement.shape[1]),
        generator=generator,
    )
    sampled = improvement[
        seed_indices.unsqueeze(-1), map_indices.unsqueeze(1)
    ].mean((1, 2))
    interval = torch.quantile(
        sampled, torch.tensor([0.025, 0.975], dtype=sampled.dtype)
    )
    return {
        "generation_seeds": len(runs),
        "maps": len(common_maps),
        "mean_normalized_primary_improvement": float(improvement.mean()),
        "hierarchical_bootstrap_95_percent_ci": [
            float(interval[0]),
            float(interval[1]),
        ],
        "seed_map_win_rate": float((improvement > 0).float().mean()),
        "evidence": (
            "positive"
            if float(interval[0]) > 0
            else "negative"
            if float(interval[1]) < 0
            else "inconclusive"
        ),
    }


def _selection(
    pooled_aggregate: Mapping[str, Any],
    macro_aggregate: Mapping[str, Any],
    runs: list[Mapping[str, Any]],
    alphas: tuple[float, ...],
    maximum_nonregression: float,
    maximum_primary_regression: float,
    minimum_primary_improvement: float,
    minimum_seed_wins: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    pooled_baseline = pooled_aggregate[_alpha_key(1.0)]
    macro_baseline = macro_aggregate[_alpha_key(1.0)]
    candidates: dict[str, Any] = {}
    for alpha in alphas:
        key = _alpha_key(alpha)
        primary_ratios = {
            metric: macro_aggregate[key][metric]["mean"]
            / max(macro_baseline[metric]["mean"], 1e-20)
            for metric in PRIMARY_METRICS
        }
        nonregression_ratios = {
            metric: pooled_aggregate[key][metric]["mean"]
            / max(pooled_baseline[metric]["mean"], 1e-20)
            for metric in NONREGRESSION_METRICS
        }
        diversity = pooled_aggregate[key]["csi_diversity_ratio_to_real"]["mean"]
        physical_control_ok = True
        for run in runs:
            audit = run["control_audit_on_pooled_subset"][key]
            accuracy = audit["condition_accuracy"]
            finite_values = [
                accuracy["total_power_db_mae"],
                accuracy["delay_spread_ns_mae"],
                accuracy["k_factor_db_mae"],
            ]
            physical_control_ok &= (
                audit["path_count_accuracy"] >= 1.0 - 1e-6
                and audit["finite_csi_rate"] >= 1.0 - 1e-6
                and accuracy["los_consistency"] >= 1.0 - 1e-6
                and all(
                    (not math.isfinite(value)) or value <= 1e-3
                    for value in finite_values
                )
            )
        seed_scores = {
            str(run["generation_seed"]): sum(
                (
                    sum(
                        item[metric]
                        for item in run["per_map_primary_metrics"][key].values()
                    )
                    / len(run["per_map_primary_metrics"][key])
                )
                / max(
                    sum(
                        item[metric]
                        for item in run["per_map_primary_metrics"][_alpha_key(1.0)].values()
                    )
                    / len(run["per_map_primary_metrics"][_alpha_key(1.0)]),
                    1e-20,
                )
                for metric in PRIMARY_METRICS
            )
            / len(PRIMARY_METRICS)
            for run in runs
        }
        seed_wins = sum(value < 1.0 for value in seed_scores.values())
        composite_bootstrap = _composite_hierarchical_bootstrap(
            runs,
            alpha,
            bootstrap_samples=bootstrap_samples,
            seed=seed + int(round(alpha * 1000)),
        )
        auxiliary_ok = all(
            ratio <= 1.0 + maximum_nonregression
            for ratio in nonregression_ratios.values()
        )
        primary_nonregression_ok = all(
            ratio <= 1.0 + maximum_primary_regression
            for ratio in primary_ratios.values()
        )
        score = sum(primary_ratios.values()) / len(primary_ratios)
        improvement_large_enough = score <= 1.0 - minimum_primary_improvement
        bootstrap_positive = composite_bootstrap.get("evidence") == "positive"
        eligible = (
            alpha != 1.0
            and auxiliary_ok
            and primary_nonregression_ok
            and 0.95 <= diversity <= 1.05
            and physical_control_ok
            and seed_wins >= minimum_seed_wins
            and improvement_large_enough
            and bootstrap_positive
        )
        candidates[key] = {
            "alpha": alpha,
            "macro_primary_mean_ratio_to_alpha1": score,
            "primary_ratios_to_alpha1": primary_ratios,
            "pooled_auxiliary_ratios_to_alpha1": nonregression_ratios,
            "pooled_diversity_ratio_to_real": diversity,
            "per_seed_macro_primary_score_ratio_to_alpha1": seed_scores,
            "generation_seed_wins": seed_wins,
            "composite_hierarchical_bootstrap": composite_bootstrap,
            "checks": {
                "primary_no_metric_worse_than_limit": primary_nonregression_ok,
                "auxiliary_nonregression": auxiliary_ok,
                "diversity_in_0.95_to_1.05": 0.95 <= diversity <= 1.05,
                "physical_control_exact_within_1e-3": physical_control_ok,
                "minimum_generation_seed_wins": seed_wins >= minimum_seed_wins,
                "minimum_primary_improvement": improvement_large_enough,
                "bootstrap_positive": bootstrap_positive,
            },
            "eligible_to_replace_alpha1": eligible,
        }
    eligible = [
        item for item in candidates.values() if item["eligible_to_replace_alpha1"]
    ]
    selected = (
        min(
            eligible,
            key=lambda item: (
                item["macro_primary_mean_ratio_to_alpha1"],
                abs(item["alpha"] - 1.0),
            ),
        )
        if eligible
        else candidates[_alpha_key(1.0)]
    )
    return {
        "selected_alpha5": selected["alpha"],
        "fell_back_to_alpha1": not bool(eligible),
        "rule": (
            "Primary selection uses equal-Map macro distances, equal-weighted across path "
            "Power--AoD, link Power--AoD and Tx spatial correlation after normalization to "
            "alpha5=1. A replacement must keep every primary metric within "
            f"{maximum_primary_regression:.1%}, improve the composite by at least "
            f"{minimum_primary_improvement:.1%}, win at least {minimum_seed_wins} generation "
            "seeds, have a positive hierarchical seed-by-Map bootstrap interval, keep pooled "
            f"DP/PDP/phase-invariant Full-CSI within {maximum_nonregression:.1%}, and keep "
            "diversity in [0.95, 1.05]. If none qualifies, retain alpha5=1."
        ),
        "candidates": candidates,
    }


def _hierarchical_bootstrap(
    runs: list[Mapping[str, Any]],
    candidate_alpha: float,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    candidate_key = _alpha_key(candidate_alpha)
    baseline_key = _alpha_key(1.0)
    result: dict[str, Any] = {}
    for metric_number, metric in enumerate(PRIMARY_METRICS):
        common_maps = sorted(
            set.intersection(
                *(
                    set(run["per_map_primary_metrics"][candidate_key])
                    & set(run["per_map_primary_metrics"][baseline_key])
                    for run in runs
                )
            )
        )
        if not common_maps:
            result[metric] = {"status": "no_common_maps"}
            continue
        improvement = torch.tensor(
            [
                [
                    run["per_map_primary_metrics"][baseline_key][map_id][metric]
                    - run["per_map_primary_metrics"][candidate_key][map_id][metric]
                    for map_id in common_maps
                ]
                for run in runs
            ],
            dtype=torch.float64,
        )
        generator = torch.Generator().manual_seed(seed + metric_number)
        seed_indices = torch.randint(
            improvement.shape[0],
            (bootstrap_samples, improvement.shape[0]),
            generator=generator,
        )
        map_indices = torch.randint(
            improvement.shape[1],
            (bootstrap_samples, improvement.shape[1]),
            generator=generator,
        )
        sampled = improvement[
            seed_indices.unsqueeze(-1), map_indices.unsqueeze(1)
        ].mean((1, 2))
        interval = torch.quantile(
            sampled, torch.tensor([0.025, 0.975], dtype=sampled.dtype)
        )
        baseline_mean = torch.tensor(
            [
                run["per_map_primary_metrics"][baseline_key][map_id][metric]
                for run in runs
                for map_id in common_maps
            ],
            dtype=torch.float64,
        ).mean()
        result[metric] = {
            "generation_seeds": len(runs),
            "maps": len(common_maps),
            "mean_improvement_alpha1_minus_candidate": float(improvement.mean()),
            "relative_mean_improvement": float(improvement.mean() / baseline_mean.clamp_min(1e-20)),
            "hierarchical_bootstrap_95_percent_ci": [
                float(interval[0]),
                float(interval[1]),
            ],
            "seed_map_win_rate": float((improvement > 0).float().mean()),
            "evidence": (
                "positive"
                if float(interval[0]) > 0
                else "negative"
                if float(interval[1]) < 0
                else "inconclusive"
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-generation-seed L=5 Link-global gate calibration."
    )
    parser.add_argument("--cache", action="append", type=_parse_cache, required=True)
    parser.add_argument("--dp-checkpoint", required=True)
    parser.add_argument("--link-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alphas", type=_parse_alphas, default=_parse_alphas("0,0.5,1,1.5"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument(
        "--maximum-samples-per-split",
        type=int,
        default=2048,
        help="Maximum Map-balanced pooled samples for auxiliary metrics only.",
    )
    parser.add_argument("--minimum-map-samples", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--swd-projections", type=int, default=128)
    parser.add_argument("--map-swd-projections", type=int, default=64)
    parser.add_argument("--projection-dimension", type=int, default=64)
    parser.add_argument("--projection-seeds", default="0,1,2")
    parser.add_argument("--maximum-nonregression", type=float, default=0.05)
    parser.add_argument("--maximum-primary-regression", type=float, default=0.02)
    parser.add_argument("--minimum-primary-improvement", type=float, default=0.02)
    parser.add_argument("--minimum-seed-wins", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    caches_by_seed = dict(args.cache)
    if len(caches_by_seed) != len(args.cache):
        parser.error("generation cache seeds must be unique")
    if len(caches_by_seed) < 2:
        parser.error("at least two independent B3 generation seeds are required")
    if not 0 <= args.maximum_nonregression < 1:
        parser.error("maximum non-regression must lie in [0, 1)")
    if not 0 <= args.maximum_primary_regression < 1:
        parser.error("maximum primary regression must lie in [0, 1)")
    if not 0 <= args.minimum_primary_improvement < 1:
        parser.error("minimum primary improvement must lie in [0, 1)")
    if not 1 <= args.minimum_seed_wins <= len(caches_by_seed):
        parser.error("minimum seed wins must lie between one and the number of caches")
    projection_seeds = tuple(
        int(value.strip()) for value in args.projection_seeds.split(",") if value.strip()
    )
    if not projection_seeds:
        parser.error("at least one projection seed is required")

    device = resolve_device(args.device)
    dp_checkpoint = torch.load(args.dp_checkpoint, map_location="cpu", weights_only=False)
    link_checkpoint = torch.load(args.link_checkpoint, map_location="cpu", weights_only=False)
    if dp_checkpoint.get("model_kind") != "scene_delay_power_residual":
        raise ValueError("--dp-checkpoint is not a DP residual")
    dp_model = _load_dp(dp_checkpoint, device)
    link_model = _load_link(link_checkpoint, device)
    if dp_model.map_id_to_index != link_model.map_id_to_index:
        raise ValueError("DP and Link-global checkpoints use different Map mappings")
    synthesizer = MIMOOFDMSynthesizer(
        SystemConfig.from_mapping(link_checkpoint["system"])
    ).to(device).eval()

    loaded_caches: dict[int, Mapping[str, Any]] = {}
    for generation_seed, path in sorted(caches_by_seed.items()):
        cache = torch.load(path, map_location="cpu", weights_only=False)
        _assert_cache_contract(cache, dp_checkpoint, link_checkpoint)
        for split in ("validation", "test"):
            if split not in cache["splits"]:
                raise ValueError(f"cache seed {generation_seed} is missing {split}")
        recorded_seed = int(cache["sampling"]["seed"])
        if recorded_seed != generation_seed:
            raise ValueError(
                f"cache label seed {generation_seed} disagrees with recorded seed {recorded_seed}"
            )
        loaded_caches[generation_seed] = cache

    validation_runs: list[dict[str, Any]] = []
    for generation_seed, cache in loaded_caches.items():
        validation_runs.append(
            _evaluate_seed_split(
                generation_seed=generation_seed,
                cache=cache,
                split="validation",
                alphas=args.alphas,
                dp_model=dp_model,
                link_model=link_model,
                synthesizer=synthesizer,
                device=device,
                selection_seed=args.selection_seed,
                maximum_samples=args.maximum_samples_per_split,
                minimum_map_samples=args.minimum_map_samples,
                batch_size=args.batch_size,
                swd_projections=args.swd_projections,
                map_swd_projections=args.map_swd_projections,
                projection_dimension=args.projection_dimension,
                projection_seeds=projection_seeds,
            )
        )
        print(
            json.dumps(
                {
                    "event": "path_count_gate_validation_seed_complete",
                    "generation_seed": generation_seed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    validation_aggregate = _aggregate_runs(validation_runs, args.alphas)
    validation_macro = _macro_primary_aggregate(validation_runs, args.alphas)
    selection = _selection(
        validation_aggregate,
        validation_macro,
        validation_runs,
        args.alphas,
        args.maximum_nonregression,
        args.maximum_primary_regression,
        args.minimum_primary_improvement,
        args.minimum_seed_wins,
        args.bootstrap_samples,
        args.selection_seed,
    )
    selected_alpha = float(selection["selected_alpha5"])
    test_alphas = tuple(dict.fromkeys((1.0, selected_alpha)))

    test_runs: list[dict[str, Any]] = []
    for generation_seed, cache in loaded_caches.items():
        test_runs.append(
            _evaluate_seed_split(
                generation_seed=generation_seed,
                cache=cache,
                split="test",
                alphas=test_alphas,
                dp_model=dp_model,
                link_model=link_model,
                synthesizer=synthesizer,
                device=device,
                selection_seed=args.selection_seed,
                maximum_samples=args.maximum_samples_per_split,
                minimum_map_samples=args.minimum_map_samples,
                batch_size=args.batch_size,
                swd_projections=args.swd_projections,
                map_swd_projections=args.map_swd_projections,
                projection_dimension=args.projection_dimension,
                projection_seeds=projection_seeds,
            )
        )
        print(
            json.dumps(
                {
                    "event": "path_count_gate_test_seed_complete",
                    "generation_seed": generation_seed,
                    "selected_alpha5": selected_alpha,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    test_aggregate = _aggregate_runs(test_runs, test_alphas)
    test_macro = _macro_primary_aggregate(test_runs, test_alphas)
    report = {
        "format_version": 1,
        "experiment": "path_count_aware_link_global_gate",
        "generation_seeds": sorted(loaded_caches),
        "cache_paths": {
            str(seed): str(caches_by_seed[seed].resolve()) for seed in sorted(caches_by_seed)
        },
        "dp_checkpoint": str(Path(args.dp_checkpoint).resolve()),
        "link_checkpoint": str(Path(args.link_checkpoint).resolve()),
        "gate_definition": (
            "For L=5, multiply both final Link-global azimuth and elevation residuals by alpha5. "
            "For L=1..4 alpha remains exactly 1; DP and all other path parameters are unchanged."
        ),
        "evaluation_protocol": {
            "primary": (
                "Per-Map L=5 Power--AoD and Tx spatial metrics use every available sample in "
                "each Map meeting the minimum support threshold; Maps and generation seeds "
                "receive equal macro weight."
            ),
            "auxiliary": (
                "Pooled DP/PDP/Full-CSI/diversity metrics use a Map-balanced subset capped by "
                "maximum_samples_per_split and do not drive the primary score."
            ),
            "minimum_map_samples": args.minimum_map_samples,
            "maximum_pooled_samples_per_split": args.maximum_samples_per_split,
        },
        "validation": {
            "alphas": list(args.alphas),
            "per_generation_seed": validation_runs,
            "aggregate": validation_aggregate,
            "pooled_auxiliary_aggregate": validation_aggregate,
            "macro_primary_aggregate": validation_macro,
            "selection": selection,
            "all_candidates_vs_alpha1_hierarchical_bootstrap": {
                _alpha_key(alpha): _hierarchical_bootstrap(
                    validation_runs,
                    alpha,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.selection_seed + index * 10000,
                )
                for index, alpha in enumerate(args.alphas)
            },
            "selected_vs_alpha1_hierarchical_bootstrap": _hierarchical_bootstrap(
                validation_runs,
                selected_alpha,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.selection_seed,
            ),
        },
        "test": {
            "evaluated_alphas": list(test_alphas),
            "selection_was_frozen_before_test": True,
            "per_generation_seed": test_runs,
            "aggregate": test_aggregate,
            "pooled_auxiliary_aggregate": test_aggregate,
            "macro_primary_aggregate": test_macro,
            "selected_vs_alpha1_hierarchical_bootstrap": _hierarchical_bootstrap(
                test_runs,
                selected_alpha,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.selection_seed + 100000,
            ),
        },
        "non_target_l_audit": {
            "l_values": [1, 2, 3, 4],
            "alpha": 1.0,
            "status": "unchanged_by_construction",
            "reason": "The conditional scaler is only applied when path_count equals five.",
        },
        "cautions": [
            "These are independent B3 generation seeds, not independent training seeds.",
            "The same frozen DP and Link-global checkpoints are used for every seed and alpha.",
            "Only the validation-selected alpha and alpha=1 are evaluated on test.",
            "Fine Map-by-physical-condition neighborhoods are intentionally not primary because their local support can collapse even when a Map has many L=5 samples.",
            "A stable inference calibration does not prove that Link-global learned the correct L-conditioned mechanism.",
        ],
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "path_count_gate_evaluation_complete",
                "output": str(Path(args.output).resolve()),
                "selected_alpha5": selected_alpha,
                "generation_seeds": sorted(loaded_caches),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
