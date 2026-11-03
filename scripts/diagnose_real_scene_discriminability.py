#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.data.scene_id import load_seen_map_spatial_plan, make_scene_id_datasets
from physics_csi.evaluation.distribution import rbf_mmd, sliced_wasserstein
from physics_csi.evaluation.research_diagnostics import (
    direction_vectors,
    normalized_path_power,
    power_aod_joint_features,
)
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


PAYLOAD_KEYS = (
    "csi",
    "path_mask",
    "path_powers_linear",
    "aod_az_deg",
    "aod_el_deg",
    "conditions",
    "condition_validity",
)
PHASE_PROBE_COUNT = 16
PHASE_PROBE_SEED = 314159
CORE_METRICS = (
    "raw_full_csi_mmd_rbf",
    "raw_full_csi_sliced_wasserstein",
    "phase_invariant_full_csi_mmd_rbf",
    "phase_invariant_full_csi_sliced_wasserstein",
    "power_aod_path_mmd_rbf",
    "power_aod_path_sliced_wasserstein",
    "power_aod_link_mmd_rbf",
    "power_aod_link_sliced_wasserstein",
    "tx_spatial_correlation_relative_frobenius",
    "tx_spatial_correlation_magnitude_mae",
    "tx_spatial_correlation_phase_mae_rad",
)


def _stack(records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([record[key] for record in records]) for key in PAYLOAD_KEYS}


def _select(
    payload: Mapping[str, torch.Tensor], indices: torch.Tensor, device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value[indices].to(device) for key, value in payload.items()}


def _condition_features(
    conditions: torch.Tensor,
    validity: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    values = conditions[:, :3].float() / std[:3].clamp_min(1e-6)
    values[:, 2] = torch.where(
        validity[:, 2].bool(), values[:, 2], torch.zeros_like(values[:, 2])
    )
    return values


def _one_to_one_match(
    query_conditions: torch.Tensor,
    query_validity: torch.Tensor,
    candidate_conditions: torch.Tensor,
    candidate_validity: torch.Tensor,
    condition_std: torch.Tensor,
    maximum_distance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = _condition_features(query_conditions, query_validity, condition_std)
    candidate = _condition_features(
        candidate_conditions, candidate_validity, condition_std
    )
    distances = torch.cdist(query, candidate)
    query_los = query_conditions[:, 3] >= 0.5
    candidate_los = candidate_conditions[:, 3] >= 0.5
    query_k_valid = query_validity[:, 2].bool()
    candidate_k_valid = candidate_validity[:, 2].bool()
    compatible = (
        (query_los[:, None] == candidate_los[None, :])
        & (query_k_valid[:, None] == candidate_k_valid[None, :])
        & torch.isfinite(distances)
        & (distances <= maximum_distance)
    )
    distances = distances.masked_fill(~compatible, torch.inf).cpu()
    row_order = torch.argsort(distances.amin(1))
    used = torch.zeros(candidate_conditions.shape[0], dtype=torch.bool)
    query_indices: list[int] = []
    candidate_indices: list[int] = []
    matched_distances: list[float] = []
    for query_index in row_order.tolist():
        for candidate_index in torch.argsort(distances[query_index]).tolist():
            distance = float(distances[query_index, candidate_index])
            if not math.isfinite(distance) or distance > maximum_distance:
                break
            if not bool(used[candidate_index]):
                used[candidate_index] = True
                query_indices.append(query_index)
                candidate_indices.append(candidate_index)
                matched_distances.append(distance)
                break
    return (
        torch.tensor(query_indices, dtype=torch.long),
        torch.tensor(candidate_indices, dtype=torch.long),
        torch.tensor(matched_distances, dtype=torch.float32),
    )


def _matched_triplet(
    anchor: Mapping[str, torch.Tensor],
    other: Mapping[str, torch.Tensor],
    *,
    condition_std: torch.Tensor,
    maximum_distance: float,
    samples: int,
    generator: torch.Generator,
    maximum_attempts: int,
) -> dict[str, Any] | None:
    anchor_count = anchor["conditions"].shape[0]
    if anchor_count < samples * 2 or other["conditions"].shape[0] < samples:
        return None
    for attempt in range(1, maximum_attempts + 1):
        permutation = torch.randperm(anchor_count, generator=generator)
        midpoint = anchor_count // 2
        first_pool = permutation[:midpoint]
        second_pool = permutation[midpoint:]
        first_ab, second_local, distances_ab = _one_to_one_match(
            anchor["conditions"][first_pool],
            anchor["condition_validity"][first_pool],
            anchor["conditions"][second_pool],
            anchor["condition_validity"][second_pool],
            condition_std,
            maximum_distance,
        )
        first_ac, other_local, distances_ac = _one_to_one_match(
            anchor["conditions"][first_pool],
            anchor["condition_validity"][first_pool],
            other["conditions"],
            other["condition_validity"],
            condition_std,
            maximum_distance,
        )
        ab = {
            int(first): (int(second), float(distance))
            for first, second, distance in zip(
                first_ab.tolist(), second_local.tolist(), distances_ab.tolist()
            )
        }
        ac = {
            int(first): (int(second), float(distance))
            for first, second, distance in zip(
                first_ac.tolist(), other_local.tolist(), distances_ac.tolist()
            )
        }
        common = sorted(set(ab) & set(ac))
        if len(common) < samples:
            continue
        order = torch.randperm(len(common), generator=generator)[:samples]
        selected_common = [common[int(index)] for index in order]
        return {
            "anchor_indices": torch.tensor(
                [int(first_pool[index]) for index in selected_common], dtype=torch.long
            ),
            "within_indices": torch.tensor(
                [int(second_pool[ab[index][0]]) for index in selected_common],
                dtype=torch.long,
            ),
            "cross_indices": torch.tensor(
                [ac[index][0] for index in selected_common], dtype=torch.long
            ),
            "within_distances": torch.tensor(
                [ab[index][1] for index in selected_common], dtype=torch.float32
            ),
            "cross_distances": torch.tensor(
                [ac[index][1] for index in selected_common], dtype=torch.float32
            ),
            "attempt": attempt,
            "common_matches_before_subsampling": len(common),
        }
    return None


_PHASE_PROBE_CACHE: dict[tuple[str, int, torch.dtype], torch.Tensor] = {}


def _phase_normalize(csi: torch.Tensor) -> torch.Tensor:
    flat = csi.flatten(1)
    cache_key = (str(csi.device), flat.shape[1], csi.dtype)
    probes = _PHASE_PROBE_CACHE.get(cache_key)
    if probes is None:
        generator = torch.Generator(device=csi.device).manual_seed(PHASE_PROBE_SEED)
        real = torch.randn(
            PHASE_PROBE_COUNT,
            flat.shape[1],
            generator=generator,
            device=csi.device,
            dtype=csi.real.dtype,
        )
        imaginary = torch.randn(
            PHASE_PROBE_COUNT,
            flat.shape[1],
            generator=generator,
            device=csi.device,
            dtype=csi.real.dtype,
        )
        probes = torch.complex(real, imaginary)
        probes = probes / probes.abs().square().sum(-1, keepdim=True).sqrt().clamp_min(
            1e-20
        )
        _PHASE_PROBE_CACHE[cache_key] = probes
    anchors = flat @ probes.conj().transpose(0, 1)
    selected = anchors.gather(1, anchors.abs().argmax(-1, keepdim=True))
    unit = selected / selected.abs().clamp_min(1e-20)
    shape = (csi.shape[0],) + (1,) * (csi.ndim - 1)
    return csi * unit.conj().reshape(shape)


def _normalized_csi_features(csi: torch.Tensor) -> torch.Tensor:
    flat = csi.flatten(1)
    flat = flat / flat.abs().square().sum(-1, keepdim=True).sqrt().clamp_min(1e-20)
    return torch.cat([flat.real, flat.imag], dim=-1).float()


def _phase_invariance_validation(csi: torch.Tensor, *, seed: int) -> dict[str, Any]:
    generator = torch.Generator(device=csi.device).manual_seed(seed)
    phase = (
        torch.rand(
            csi.shape[0],
            generator=generator,
            device=csi.device,
            dtype=csi.real.dtype,
        )
        * (2.0 * math.pi)
        - math.pi
    )
    shape = (csi.shape[0],) + (1,) * (csi.ndim - 1)
    rotated = csi * torch.exp(1j * phase).reshape(shape)
    expected = _phase_normalize(csi)
    actual = _phase_normalize(rotated)
    numerator = (actual - expected).abs().square().flatten(1).sum(-1)
    denominator = expected.abs().square().flatten(1).sum(-1).clamp_min(1e-20)
    relative_error = float((numerator / denominator).mean())
    return {
        "relative_error_after_random_rotation_and_canonicalization": relative_error,
        "status": "ok" if relative_error < 1e-10 else "investigate",
    }


def _power_aod_path_features(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask = payload["path_mask"].bool()
    power_fraction = normalized_path_power(payload)
    log_power_scaled = 10.0 * torch.log10(power_fraction.clamp_min(1e-20)) / 20.0
    direction = direction_vectors(payload["aod_az_deg"], payload["aod_el_deg"])
    return torch.cat([log_power_scaled.unsqueeze(-1), direction], dim=-1)[mask]


def _tx_spatial_correlation(csi: torch.Tensor) -> torch.Tensor:
    samples = csi.permute(0, 1, 3, 2).reshape(-1, csi.shape[2])
    correlation = samples.conj().T @ samples / max(samples.shape[0], 1)
    diagonal = correlation.diagonal().real.mean().clamp_min(1e-20)
    return correlation / diagonal


def _feature_distance(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    bandwidths: list[float],
    projections: int,
    seed: int,
) -> tuple[float, float]:
    return (
        float(rbf_mmd(first, second, bandwidths)),
        float(
            sliced_wasserstein(
                first, second, projections=projections, seed=seed
            )
        ),
    )


def _feature_bundle(payload: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "raw_full_csi": _normalized_csi_features(payload["csi"]),
        "phase_invariant_full_csi": _normalized_csi_features(
            _phase_normalize(payload["csi"])
        ),
        "power_aod_path": _power_aod_path_features(payload),
        "power_aod_link": power_aod_joint_features(payload),
        "tx_spatial_correlation": _tx_spatial_correlation(payload["csi"]),
    }


def _distances(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
    *,
    projections: int,
    seed: int,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, bandwidths in (
        ("raw_full_csi", [0.25, 0.5, 1.0, 2.0]),
        ("phase_invariant_full_csi", [0.25, 0.5, 1.0, 2.0]),
        ("power_aod_path", [0.25, 0.5, 1.0, 2.0]),
        ("power_aod_link", [0.25, 0.5, 1.0, 2.0]),
    ):
        mmd, swd = _feature_distance(
            first[name],
            second[name],
            bandwidths=bandwidths,
            projections=projections,
            seed=seed,
        )
        result[f"{name}_mmd_rbf"] = mmd
        result[f"{name}_sliced_wasserstein"] = swd
    first_correlation = first["tx_spatial_correlation"]
    second_correlation = second["tx_spatial_correlation"]
    difference = first_correlation - second_correlation
    denominator = (
        torch.linalg.matrix_norm(first_correlation)
        + torch.linalg.matrix_norm(second_correlation)
    ).clamp_min(1e-20)
    valid_phase = (
        ~torch.eye(
            first_correlation.shape[0],
            dtype=torch.bool,
            device=first_correlation.device,
        )
        & (first_correlation.abs() > 1e-3)
        & (second_correlation.abs() > 1e-3)
    )
    phase_error = (
        torch.angle(
            first_correlation[valid_phase]
            * second_correlation[valid_phase].conj()
        )
        .abs()
        .mean()
        if bool(valid_phase.any())
        else torch.zeros((), device=first_correlation.device)
    )
    result.update(
        {
            "tx_spatial_correlation_relative_frobenius": float(
                2.0 * torch.linalg.matrix_norm(difference) / denominator
            ),
            "tx_spatial_correlation_magnitude_mae": float(
                (first_correlation.abs() - second_correlation.abs()).abs().mean()
            ),
            "tx_spatial_correlation_phase_mae_rad": float(phase_error),
        }
    )
    return result


def _condition_audit(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
    distances: torch.Tensor,
    condition_std: torch.Tensor,
) -> dict[str, Any]:
    first_conditions = first["conditions"].cpu().float()
    second_conditions = second["conditions"].cpu().float()
    first_validity = first["condition_validity"].cpu().bool()
    second_validity = second["condition_validity"].cpu().bool()
    valid_k = first_validity[:, 2] & second_validity[:, 2]
    gap = (first_conditions[:, :2] - second_conditions[:, :2]).abs()
    result: dict[str, Any] = {
        "samples": int(first_conditions.shape[0]),
        "standardized_distance_mean": float(distances.mean()),
        "standardized_distance_maximum": float(distances.max()),
        "power_absolute_gap_mean_in_std": float(
            (gap[:, 0] / condition_std[0].clamp_min(1e-6)).mean()
        ),
        "delay_absolute_gap_mean_in_std": float(
            (gap[:, 1] / condition_std[1].clamp_min(1e-6)).mean()
        ),
        "los_exact_for_every_pair": bool(
            torch.equal(
                first_conditions[:, 3] >= 0.5, second_conditions[:, 3] >= 0.5
            )
        ),
        "k_validity_exact_for_every_pair": bool(
            torch.equal(first_validity[:, 2], second_validity[:, 2])
        ),
    }
    result["k_absolute_gap_mean_in_std_valid_only"] = (
        float(
            (
                (first_conditions[valid_k, 2] - second_conditions[valid_k, 2]).abs()
                / condition_std[2].clamp_min(1e-6)
            ).mean()
        )
        if bool(valid_k.any())
        else None
    )
    return result


def _mean_metric(repeats: list[dict[str, Any]], section: str) -> dict[str, float]:
    return {
        metric: float(
            torch.tensor(
                [repeat[section][metric] for repeat in repeats], dtype=torch.float64
            ).mean()
        )
        for metric in CORE_METRICS
    }


def _repeat_bootstrap_summary(
    report: Mapping[str, Any], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    repeats = report["repeats"]
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        len(repeats), (bootstrap_samples, len(repeats)), generator=generator
    )
    result: dict[str, Any] = {}
    quantiles = torch.tensor([0.025, 0.975], dtype=torch.float64)
    for metric in CORE_METRICS:
        same = torch.tensor(
            [repeat["same_map"][metric] for repeat in repeats], dtype=torch.float64
        )
        cross = torch.tensor(
            [repeat["cross_map"][metric] for repeat in repeats], dtype=torch.float64
        )
        delta = cross - same
        bootstrap_delta = delta[indices].mean(-1)
        interval = torch.quantile(bootstrap_delta, quantiles)
        result[metric] = {
            "same_map_mean": float(same.mean()),
            "cross_map_mean": float(cross.mean()),
            "delta_scene_cross_minus_same": float(delta.mean()),
            "delta_scene_bootstrap_95_percent_ci": interval.tolist(),
            "cross_to_same_ratio": float(cross.mean() / same.mean().clamp_min(1e-20)),
            "repeat_win_rate": float((delta > 0).double().mean()),
            "evidence": (
                "positive_scene_discriminability"
                if float(interval[0]) > 0
                else "negative"
                if float(interval[1]) < 0
                else "inconclusive"
            ),
            "inference_unit": "repeated matched split of this single Map pair",
        }
    return result


def _map_cluster_summary(
    reports: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    if not reports:
        return {"status": "no_successful_map_pairs"}
    map_ids = sorted(
        {report["anchor_map"] for report in reports}
        | {report["other_map"] for report in reports}
    )
    map_index = {map_id: index for index, map_id in enumerate(map_ids)}
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(
        len(map_ids), (bootstrap_samples, len(map_ids)), generator=generator
    )
    multiplicity = torch.zeros(
        bootstrap_samples, len(map_ids), dtype=torch.float64
    )
    multiplicity.scatter_add_(
        1, draws, torch.ones_like(draws, dtype=torch.float64)
    )
    first = torch.tensor([map_index[r["anchor_map"]] for r in reports])
    second = torch.tensor([map_index[r["other_map"]] for r in reports])
    weights = multiplicity[:, first] * multiplicity[:, second]
    valid = weights.sum(-1) > 0
    quantiles = torch.tensor([0.025, 0.975], dtype=torch.float64)
    result: dict[str, Any] = {}
    for metric_index, metric in enumerate(CORE_METRICS):
        same = torch.tensor(
            [report["same_map_mean"][metric] for report in reports],
            dtype=torch.float64,
        )
        cross = torch.tensor(
            [report["cross_map_mean"][metric] for report in reports],
            dtype=torch.float64,
        )
        delta = cross - same
        bootstrap_delta = (
            (weights[valid] * delta).sum(-1) / weights[valid].sum(-1)
        )
        interval = torch.quantile(bootstrap_delta, quantiles)
        p_value = min(
            1.0,
            2.0
            * min(
                float((bootstrap_delta <= 0).double().mean()),
                float((bootstrap_delta >= 0).double().mean()),
            ),
        )
        result[metric] = {
            "map_pairs": len(reports),
            "same_map_mean": float(same.mean()),
            "cross_map_mean": float(cross.mean()),
            "delta_scene_cross_minus_same": float(delta.mean()),
            "delta_scene_cluster_bootstrap_95_percent_ci": interval.tolist(),
            "cross_to_same_ratio": float(cross.mean() / same.mean().clamp_min(1e-20)),
            "map_pair_win_rate": float((delta > 0).double().mean()),
            "cluster_bootstrap_two_sided_p": p_value,
            "valid_bootstrap_replicates": int(valid.sum()),
            "evidence": (
                "positive_scene_discriminability"
                if float(interval[0]) > 0
                else "negative"
                if float(interval[1]) < 0
                else "inconclusive"
            ),
            "inference_unit": "Map ID pigeonhole cluster bootstrap",
            "metric_order": metric_index,
        }
    return result


def _scope_summary(
    reports: list[dict[str, Any]],
    *,
    anchor_map: int,
    specific_map: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    specific = [
        report
        for report in reports
        if report["anchor_map"] == anchor_map
        and report["other_map"] == specific_map
    ]
    anchor = [report for report in reports if report["anchor_map"] == anchor_map]
    return {
        "specific_pair": (
            _repeat_bootstrap_summary(
                specific[0], bootstrap_samples=bootstrap_samples, seed=seed
            )
            if specific
            else {"status": "specific_pair_not_available"}
        ),
        "anchor_map_vs_others": _map_cluster_summary(
            anchor, bootstrap_samples=bootstrap_samples, seed=seed + 1000
        ),
        "all_reported_map_pairs": _map_cluster_summary(
            reports, bootstrap_samples=bootstrap_samples, seed=seed + 2000
        ),
    }


def _matching_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(section: str) -> dict[str, Any]:
        audits = [
            repeat[section]
            for report in reports
            for repeat in report["repeats"]
        ]
        return {
            "audits": len(audits),
            "all_los_pairs_exact": all(
                audit["los_exact_for_every_pair"] for audit in audits
            ),
            "all_k_validity_pairs_exact": all(
                audit["k_validity_exact_for_every_pair"] for audit in audits
            ),
            "maximum_standardized_distance_observed": max(
                audit["standardized_distance_maximum"] for audit in audits
            ),
            "mean_standardized_distance": float(
                torch.tensor(
                    [audit["standardized_distance_mean"] for audit in audits],
                    dtype=torch.float64,
                ).mean()
            ),
            "mean_power_absolute_gap_in_std": float(
                torch.tensor(
                    [audit["power_absolute_gap_mean_in_std"] for audit in audits],
                    dtype=torch.float64,
                ).mean()
            ),
            "mean_delay_absolute_gap_in_std": float(
                torch.tensor(
                    [audit["delay_absolute_gap_mean_in_std"] for audit in audits],
                    dtype=torch.float64,
                ).mean()
            ),
        }

    within = summarize("within_condition_matching")
    cross = summarize("cross_condition_matching")
    return {
        "within_map_A_to_B": within,
        "cross_map_A_to_C": cross,
        "cross_minus_within_mean_standardized_distance": (
            cross["mean_standardized_distance"]
            - within["mean_standardized_distance"]
        ),
        "warning": (
            "A large positive cross-minus-within matching distance can inflate "
            "Delta_scene and must be reported with the metric results."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Real-to-real scene discriminability using matched A/B/C RayVerse triplets."
        )
    )
    parser.add_argument("--scene-plan", required=True)
    parser.add_argument("--prepared-root", default="artifacts/prepared_d2los")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--anchor-map", type=int, default=10)
    parser.add_argument("--specific-map", type=int, default=25)
    parser.add_argument("--maximum-anchor-pairs", type=int, default=0)
    parser.add_argument("--maximum-general-pairs", type=int, default=50)
    parser.add_argument("--maximum-samples-per-map", type=int, default=256)
    parser.add_argument("--samples-per-comparison", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--minimum-successful-repeats", type=int, default=10)
    parser.add_argument("--maximum-attempts-per-repeat", type=int, default=200)
    parser.add_argument("--maximum-condition-distance", type=float, default=1.0)
    parser.add_argument("--diagnostic-projections", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    positive = (
        args.maximum_samples_per_map,
        args.samples_per_comparison,
        args.repeats,
        args.minimum_successful_repeats,
        args.maximum_attempts_per_repeat,
        args.diagnostic_projections,
        args.bootstrap_samples,
    )
    if min(positive) < 1 or args.maximum_condition_distance <= 0:
        parser.error(
            "sample, repeat, projection, bootstrap and distance values must be positive"
        )
    if args.maximum_anchor_pairs < 0 or args.maximum_general_pairs < 0:
        parser.error("maximum Map-pair counts cannot be negative")
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    if args.minimum_successful_repeats > args.repeats:
        parser.error("--minimum-successful-repeats cannot exceed --repeats")

    torch.set_grad_enabled(False)
    device = resolve_device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    plan = load_seen_map_spatial_plan(args.scene_plan)
    dataset = make_scene_id_datasets(args.prepared_root, plan)[args.split]
    records_by_map: dict[int, list[dict[str, Any]]] = {}
    for index in range(len(dataset)):
        record = dataset[index]
        records_by_map.setdefault(int(record["map_id"]), []).append(record)
        if (index + 1) % 1000 == 0 or index + 1 == len(dataset):
            print(
                json.dumps(
                    {
                        "event": "dataset_loading",
                        "samples_complete": index + 1,
                        "samples_total": len(dataset),
                        "maps_seen": len(records_by_map),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    for map_id, records in records_by_map.items():
        if len(records) > args.maximum_samples_per_map:
            selection = torch.randperm(len(records), generator=generator)[
                : args.maximum_samples_per_map
            ]
            records_by_map[map_id] = [records[int(index)] for index in selection]
    payloads = {
        map_id: _stack(records) for map_id, records in records_by_map.items()
    }
    del records_by_map
    maps = sorted(payloads)
    if args.anchor_map not in payloads or args.specific_map not in payloads:
        raise ValueError("anchor and specific comparison Maps must exist in the selected split")
    all_conditions = torch.cat(
        [payload["conditions"].float() for payload in payloads.values()]
    )
    condition_std = all_conditions.std(0, unbiased=False).clamp_min(1e-6)
    validation_csi = torch.cat(
        [payloads[map_id]["csi"][:2] for map_id in maps]
    )[:128].to(device)
    phase_validation = _phase_invariance_validation(
        validation_csi, seed=args.seed + 900000
    )
    if phase_validation["status"] != "ok":
        raise RuntimeError(
            "phase-invariant canonicalization failed its numerical invariance check"
        )
    del validation_csi

    anchor_targets = [map_id for map_id in maps if map_id != args.anchor_map]
    anchor_order = torch.randperm(len(anchor_targets), generator=generator).tolist()
    anchor_targets = [anchor_targets[index] for index in anchor_order]
    if args.specific_map in anchor_targets:
        anchor_targets.remove(args.specific_map)
    if args.maximum_anchor_pairs > 0:
        anchor_targets = anchor_targets[: max(args.maximum_anchor_pairs - 1, 0)]
    anchor_targets = [args.specific_map, *anchor_targets]
    general_pairs = [
        pair
        for pair in itertools.combinations(maps, 2)
        if args.anchor_map not in pair
    ]
    general_order = torch.randperm(len(general_pairs), generator=generator).tolist()
    general_pairs = [general_pairs[index] for index in general_order]
    general_pairs = general_pairs[: args.maximum_general_pairs]
    pair_specs = [
        {
            "anchor_map": args.anchor_map,
            "other_map": target,
            "scope": "specific" if target == args.specific_map else "anchor",
        }
        for target in anchor_targets
    ] + [
        {"anchor_map": first, "other_map": second, "scope": "general"}
        for first, second in general_pairs
    ]

    reports: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for pair_number, spec in enumerate(pair_specs, start=1):
        anchor_map = spec["anchor_map"]
        other_map = spec["other_map"]
        repeats: list[dict[str, Any]] = []
        failed_repeats = 0
        for repeat in range(args.repeats):
            match = _matched_triplet(
                payloads[anchor_map],
                payloads[other_map],
                condition_std=condition_std,
                maximum_distance=args.maximum_condition_distance,
                samples=args.samples_per_comparison,
                generator=generator,
                maximum_attempts=args.maximum_attempts_per_repeat,
            )
            if match is None:
                failed_repeats += 1
                continue
            anchor = _select(payloads[anchor_map], match["anchor_indices"], device)
            within = _select(payloads[anchor_map], match["within_indices"], device)
            cross = _select(payloads[other_map], match["cross_indices"], device)
            anchor_features = _feature_bundle(anchor)
            within_features = _feature_bundle(within)
            cross_features = _feature_bundle(cross)
            metric_seed = args.seed + pair_number * 1000 + repeat
            same_metrics = _distances(
                anchor_features,
                within_features,
                projections=args.diagnostic_projections,
                seed=metric_seed,
            )
            cross_metrics = _distances(
                anchor_features,
                cross_features,
                projections=args.diagnostic_projections,
                seed=metric_seed,
            )
            repeats.append(
                {
                    "repeat": repeat,
                    "matching_attempt": match["attempt"],
                    "common_matches_before_subsampling": match[
                        "common_matches_before_subsampling"
                    ],
                    "same_map": same_metrics,
                    "cross_map": cross_metrics,
                    "delta_scene": {
                        metric: cross_metrics[metric] - same_metrics[metric]
                        for metric in CORE_METRICS
                    },
                    "within_condition_matching": _condition_audit(
                        anchor, within, match["within_distances"], condition_std
                    ),
                    "cross_condition_matching": _condition_audit(
                        anchor, cross, match["cross_distances"], condition_std
                    ),
                }
            )
            del anchor, within, cross
        if len(repeats) < args.minimum_successful_repeats:
            exclusions.append(
                {
                    **spec,
                    "reason": "insufficient complete A/B/C matched repeats",
                    "successful_repeats": len(repeats),
                    "failed_repeats": failed_repeats,
                }
            )
        else:
            reports.append(
                {
                    **spec,
                    "successful_repeats": len(repeats),
                    "failed_repeats": failed_repeats,
                    "samples_per_comparison": args.samples_per_comparison,
                    "same_map_mean": _mean_metric(repeats, "same_map"),
                    "cross_map_mean": _mean_metric(repeats, "cross_map"),
                    "delta_scene_mean": _mean_metric(repeats, "delta_scene"),
                    "repeats": repeats,
                }
            )
        print(
            json.dumps(
                {
                    "event": "map_pair_complete",
                    "pair": pair_number,
                    "pairs_total": len(pair_specs),
                    "anchor_map": anchor_map,
                    "other_map": other_map,
                    "successful_repeats": len(repeats),
                    "failed_repeats": failed_repeats,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if not reports:
        raise RuntimeError("no Map pair produced a valid matched A/B/C comparison")
    summary = _scope_summary(
        reports,
        anchor_map=args.anchor_map,
        specific_map=args.specific_map,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed + 100000,
    )
    report = {
        "format_version": 1,
        "diagnostic": "real_to_real_scene_discriminability",
        "purpose": (
            "Test whether real CSI from different Maps is more distant than independent "
            "same-Map CSI after matching Power, Delay, K-factor and LoS."
        ),
        "scene_plan_fingerprint": plan["fingerprint"],
        "split": args.split,
        "device": str(device),
        "maps": len(payloads),
        "samples_in_split": len(dataset),
        "anchor_map": args.anchor_map,
        "specific_map": args.specific_map,
        "requested_map_pairs": len(pair_specs),
        "reported_map_pairs": len(reports),
        "excluded_map_pairs": exclusions,
        "configuration": {
            "samples_per_comparison": args.samples_per_comparison,
            "repeats": args.repeats,
            "minimum_successful_repeats": args.minimum_successful_repeats,
            "maximum_attempts_per_repeat": args.maximum_attempts_per_repeat,
            "maximum_condition_distance": args.maximum_condition_distance,
            "maximum_samples_per_map": args.maximum_samples_per_map,
            "diagnostic_projections": args.diagnostic_projections,
            "bootstrap_samples": args.bootstrap_samples,
            "condition_std": condition_std.tolist(),
            "phase_probe_count": PHASE_PROBE_COUNT,
            "phase_probe_seed": PHASE_PROBE_SEED,
        },
        "matching": {
            "method": (
                "Disjoint same-Map A/B split; one-to-one matching without replacement "
                "from A to B and from the same A pool to cross-Map C."
            ),
            "equal_sample_count_for_same_and_cross": True,
            "same_anchor_samples_for_same_and_cross": True,
            "los_exact": True,
            "k_validity_exact": True,
            "observed": _matching_summary(reports),
        },
        "phase_invariance_validation": phase_validation,
        "metric_definitions": {
            "raw_full_csi": "Per-link energy-normalized complex CSI.",
            "phase_invariant_full_csi": (
                "Global phase removed with the strongest of 16 fixed complex probes."
            ),
            "power_aod_path": (
                "Active-path log power fraction jointly with circularity-safe AoD unit vector."
            ),
            "power_aod_link": (
                "Per-link power-weighted AoD second moments, strongest direction, maximum "
                "power fraction and normalized power entropy."
            ),
            "tx_spatial_correlation": (
                "Complete normalized complex Tx correlation matrix; symmetric relative "
                "Frobenius, magnitude and valid off-diagonal phase distances."
            ),
        },
        "inference": {
            "delta_scene_definition": "D_cross - D_same; positive favors detectable scene differences.",
            "general_method": "Map-ID pigeonhole cluster bootstrap over paired Map contrasts.",
            "specific_pair_method": "Bootstrap over repeated matched A/B/C splits; not population-level Map inference.",
        },
        "summary": summary,
        "map_pair_reports": reports,
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "real_scene_discriminability_complete",
                "output": str(Path(args.output).resolve()),
                "reported_map_pairs": len(reports),
                "excluded_map_pairs": len(exclusions),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
