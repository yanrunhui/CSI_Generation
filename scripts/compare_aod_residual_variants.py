#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.evaluation.scene_diagnostics import paired_map_summary
from physics_csi.training.aod_residual import AoDDistributionLoss, AoDLossWeights
from physics_csi.training.trainer import move_to_device, resolve_device
from physics_csi.utils.io import atomic_json_dump


def _subset(payload: Mapping[str, Any], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    sample_count = int(
        next(value for value in payload.values() if isinstance(value, torch.Tensor)).shape[0]
    )
    return {
        key: value[indices]
        for key, value in payload.items()
        if key != "csi"
        and isinstance(value, torch.Tensor)
        and value.shape[:1] == (sample_count,)
    }


def _wrap_degrees(value: torch.Tensor) -> torch.Tensor:
    radians = torch.deg2rad(value.float())
    return torch.rad2deg(torch.atan2(radians.sin(), radians.cos()))


def _quantiles(value: torch.Tensor, *, absolute: bool = False) -> dict[str, float]:
    value = value.detach().float().flatten()
    if absolute:
        value = value.abs()
    return {
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "p01": float(torch.quantile(value, 0.01)),
        "p05": float(torch.quantile(value, 0.05)),
        "p50": float(torch.quantile(value, 0.50)),
        "p95": float(torch.quantile(value, 0.95)),
        "p99": float(torch.quantile(value, 0.99)),
        "minimum": float(value.min()),
        "maximum": float(value.max()),
    }


def _assert_same_samples(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> None:
    if candidate.get("scene_plan_fingerprint") != baseline.get("scene_plan_fingerprint"):
        raise ValueError("candidate and baseline use different scene plans")
    for section, keys in {
        "reference": (
            "map_id",
            "conditions",
            "condition_validity",
            "path_mask",
            "path_delays_ns",
            "path_powers_linear",
            "aod_az_deg",
            "aod_el_deg",
        ),
        "dp_only": (
            "map_id",
            "path_mask",
            "path_delays_ns",
            "path_powers_linear",
            "aod_az_deg",
            "aod_el_deg",
        ),
    }.items():
        for key in keys:
            first = candidate[section][key]
            second = baseline[section][key]
            if first.shape != second.shape:
                raise ValueError(f"{section}.{key} shapes differ")
            if first.is_floating_point():
                same = torch.allclose(
                    first, second, atol=1e-7, rtol=1e-6, equal_nan=True
                )
            else:
                same = torch.equal(first, second)
            if not same:
                raise ValueError(
                    f"candidate and baseline do not contain identical {section}.{key} samples"
                )


def _make_loss(checkpoint: Mapping[str, Any], seed: int) -> AoDDistributionLoss:
    values = copy.deepcopy(checkpoint["loss_init"])
    values["seed"] = seed
    values["projections"] = max(128, int(values.get("projections", 64)))
    values["weighted_quantiles"] = max(
        512, int(values.get("weighted_quantiles", 128))
    )
    weights = AoDLossWeights(**values.pop("weights"))
    return AoDDistributionLoss(**values, weights=weights)


def _residual_audit(
    artifact: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    dp = artifact["dp_only"]
    corrected = artifact["correct_map"]
    mask = dp["path_mask"].bool()
    azimuth = _wrap_degrees(corrected["aod_az_deg"] - dp["aod_az_deg"])
    elevation = corrected["aod_el_deg"].float() - dp["aod_el_deg"].float()
    state = checkpoint.get("ema_state_dict", checkpoint["model_state_dict"])
    azimuth_gate = float(torch.sigmoid(state["azimuth_gate_logit"]))
    elevation_gate = float(torch.sigmoid(state["elevation_gate_logit"]))
    model_init = checkpoint["model_init"]
    limits = {
        "azimuth": azimuth_gate
        * float(model_init["maximum_azimuth_residual_deg"]),
        "elevation": elevation_gate
        * float(model_init["maximum_elevation_residual_deg"]),
    }

    def summarize(values: torch.Tensor, limit: float) -> dict[str, Any]:
        active = values[mask].float()
        within_link = torch.stack(
            [values[index][mask[index]].float().std(unbiased=False) for index in range(mask.shape[0])]
        )
        ratio = active.abs() / max(limit, 1e-12)
        return {
            "effective_limit_deg": limit,
            "signed": _quantiles(active),
            "absolute": _quantiles(active, absolute=True),
            "mean_absolute_fraction_of_limit": float(ratio.mean()),
            "fraction_at_least_90_percent_of_limit": float((ratio >= 0.90).float().mean()),
            "fraction_at_least_95_percent_of_limit": float((ratio >= 0.95).float().mean()),
            "fraction_at_least_99_percent_of_limit": float((ratio >= 0.99).float().mean()),
            "within_link_path_std_deg": _quantiles(within_link, absolute=True),
        }

    return {
        "model_kind": checkpoint["model_kind"],
        "aod_architecture": checkpoint.get("aod_architecture", "per_path"),
        "architectural_maximum_deg": {
            "azimuth": float(model_init["maximum_azimuth_residual_deg"]),
            "elevation": float(model_init["maximum_elevation_residual_deg"]),
        },
        "learned_gate": {
            "azimuth": azimuth_gate,
            "elevation": elevation_gate,
        },
        "loss_residual_normalization_deg": {
            "azimuth": float(checkpoint["loss_init"]["maximum_azimuth_residual_deg"]),
            "elevation": float(checkpoint["loss_init"]["maximum_elevation_residual_deg"]),
        },
        "azimuth": summarize(azimuth, limits["azimuth"]),
        "elevation": summarize(elevation, limits["elevation"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two AoD residual variants on exactly matched test samples."
    )
    parser.add_argument("--candidate-artifact", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--baseline-artifact", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument(
        "--experiment", choices=["range_limit", "link_global"], required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=12026)
    parser.add_argument("--minimum-samples-per-map", type=int, default=12)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    device = resolve_device(args.device)
    candidate = torch.load(
        args.candidate_artifact, map_location="cpu", weights_only=False
    )
    baseline = torch.load(args.baseline_artifact, map_location="cpu", weights_only=False)
    candidate_checkpoint = torch.load(
        args.candidate_checkpoint, map_location="cpu", weights_only=False
    )
    baseline_checkpoint = torch.load(
        args.baseline_checkpoint, map_location="cpu", weights_only=False
    )
    for artifact in (candidate, baseline):
        if artifact.get("model_kind") != "b3_plus_dp_plus_aod_residual":
            raise ValueError("both artifacts must be B3+DP+AoD evaluations")
    _assert_same_samples(candidate, baseline)
    for artifact, checkpoint, supplied_path in (
        (candidate, candidate_checkpoint, args.candidate_checkpoint),
        (baseline, baseline_checkpoint, args.baseline_checkpoint),
    ):
        recorded = artifact.get("checkpoint")
        if recorded and Path(recorded).resolve() != Path(supplied_path).resolve():
            raise ValueError("artifact and supplied checkpoint do not match")
        if checkpoint.get("model_kind") not in {
            "scene_aod_residual",
            "scene_link_global_aod_residual",
        }:
            raise ValueError("unsupported AoD checkpoint kind")

    loss_function = _make_loss(baseline_checkpoint, args.seed).to(device).eval()
    reference = candidate["reference"]
    map_ids = reference["map_id"].long()
    per_map: dict[str, Any] = {}
    metric_names = (
        "path_power_aod_swd",
        "delay_power_aod_swd",
        "link_power_aod_swd",
        "power_weighted_aod_swd",
        "distribution_score",
    )
    with torch.inference_mode():
        for map_id in sorted(int(item) for item in torch.unique(map_ids)):
            indices = torch.nonzero(map_ids == map_id, as_tuple=False).flatten()
            if indices.numel() < args.minimum_samples_per_map:
                per_map[f"map_{map_id}"] = {
                    "map_id": map_id,
                    "samples": int(indices.numel()),
                    "status": "insufficient_samples",
                }
                continue
            current_reference = move_to_device(_subset(reference, indices), device)
            payloads = {
                args.candidate_label: candidate["correct_map"],
                f"{args.candidate_label}_shuffled": candidate["shuffled_map"],
                args.baseline_label: baseline["correct_map"],
                f"{args.baseline_label}_shuffled": baseline["shuffled_map"],
                "dp_only": candidate["dp_only"],
            }
            models: dict[str, Any] = {}
            for label, payload in payloads.items():
                terms = loss_function.distribution_terms(
                    move_to_device(_subset(payload, indices), device), current_reference
                )
                models[label] = {key: float(value) for key, value in terms.items()}
            per_map[f"map_{map_id}"] = {
                "map_id": map_id,
                "samples": int(indices.numel()),
                "status": "ok",
                "models": models,
            }

    maps = [item for item in per_map.values() if item.get("status") == "ok"]
    if not maps:
        raise RuntimeError("no Map has enough matched samples")
    comparisons = {
        "candidate_vs_baseline": (args.candidate_label, args.baseline_label),
        "candidate_vs_dp_only": (args.candidate_label, "dp_only"),
        "baseline_vs_dp_only": (args.baseline_label, "dp_only"),
        "candidate_correct_vs_shuffled": (
            args.candidate_label,
            f"{args.candidate_label}_shuffled",
        ),
        "baseline_correct_vs_shuffled": (
            args.baseline_label,
            f"{args.baseline_label}_shuffled",
        ),
    }
    summaries: dict[str, Any] = {}
    for comparison_number, (name, (first, second)) in enumerate(comparisons.items()):
        summaries[name] = {}
        for metric_number, metric in enumerate(metric_names):
            summaries[name][metric] = paired_map_summary(
                [item["models"][first][metric] for item in maps],
                [item["models"][second][metric] for item in maps],
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + comparison_number * 1000 + metric_number,
            )

    if args.experiment == "range_limit":
        decision_rule = (
            "A wider model supports a binding-range explanation only if its learned effective "
            "limit and residual magnitude increase and it beats the original model on matched "
            "per-Map AoD metrics. Continued saturation without improvement supports an objective-"
            "driven boundary strategy instead. CSI/PDP remain external non-regression gates."
        )
    else:
        decision_rule = (
            "If Link-global matches or beats Per-path, a shared link rotation is sufficient. "
            "A stable Per-path advantage, especially in Delay--Power--AoD and link-level "
            "Power--AoD, is required to justify separate path corrections."
        )
    report = {
        "format_version": 1,
        "experiment": args.experiment,
        "metric_direction": "lower_is_better",
        "candidate_label": args.candidate_label,
        "baseline_label": args.baseline_label,
        "samples": int(map_ids.numel()),
        "maps_evaluated": len(maps),
        "minimum_samples_per_map": args.minimum_samples_per_map,
        "independent_metric_configuration": {
            "seed": args.seed,
            "projections": int(loss_function.path_directions.shape[0]),
            "weighted_quantiles": int(loss_function.quantiles.numel()),
        },
        "candidate_residual_audit": _residual_audit(
            candidate, candidate_checkpoint
        ),
        "baseline_residual_audit": _residual_audit(
            baseline, baseline_checkpoint
        ),
        "summaries": summaries,
        "per_map": per_map,
        "decision_rule": decision_rule,
        "cautions": [
            "Positive paired improvement favors the candidate over the named comparison.",
            "One training seed and one B3 generation cache do not establish training stability.",
            "AoD changes alter Tx array responses even when path phases are frozen.",
        ],
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "aod_residual_variant_comparison_complete",
                "experiment": args.experiment,
                "output": str(Path(args.output).resolve()),
                "maps": len(maps),
                "samples": int(map_ids.numel()),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
