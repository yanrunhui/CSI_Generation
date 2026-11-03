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

from physics_csi.evaluation.distribution import sliced_wasserstein
from physics_csi.evaluation.research_diagnostics import (
    direction_vectors,
    normalized_path_power,
    power_aod_joint_features,
)
from physics_csi.evaluation.scene_diagnostics import paired_map_summary
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


MODEL_LABELS = ("b3_scene_id", "b3", "shuffled_map")
ARTIFACT_KEYS = {
    "b3_scene_id": "correct_map",
    "b3": "physics_only_baseline",
    "shuffled_map": "shuffled_map",
}
METRIC_NAMES = (
    "path_level_power_aod_sliced_wasserstein",
    "link_level_power_aod_sliced_wasserstein",
    "power_weighted_aod_sliced_wasserstein",
)
EVALUATION_SECTIONS = (
    "unconditional",
    "local_condition_weighted",
    "common_path_count_controlled",
)


def _subset(
    payload: Mapping[str, Any], indices: torch.Tensor, device: torch.device
) -> dict[str, torch.Tensor]:
    sample_count = int(
        next(value for value in payload.values() if isinstance(value, torch.Tensor)).shape[0]
    )
    return {
        key: value[indices.to(value.device)].to(device)
        for key, value in payload.items()
        if isinstance(value, torch.Tensor) and value.shape[:1] == (sample_count,)
    }


def _path_power_aod_features(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask = payload["path_mask"].bool()
    power_db = 10.0 * torch.log10(normalized_path_power(payload).clamp_min(1e-20))
    direction = direction_vectors(payload["aod_az_deg"], payload["aod_el_deg"])
    return torch.cat([direction, power_db.unsqueeze(-1)], dim=-1)[mask]


def _standardized_swd(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    projections: int,
    seeds: tuple[int, ...],
) -> tuple[float, dict[str, Any]]:
    mean = reference.mean(0, keepdim=True)
    raw_scale = reference.std(0, unbiased=False, keepdim=True)
    positive = raw_scale[torch.isfinite(raw_scale) & (raw_scale > 0)]
    typical = (
        positive.median()
        if positive.numel()
        else torch.ones((), dtype=reference.dtype, device=reference.device)
    )
    floor = torch.maximum(
        typical * 0.05,
        torch.tensor(1e-6, dtype=reference.dtype, device=reference.device),
    )
    scale = torch.where(
        torch.isfinite(raw_scale), raw_scale, torch.zeros_like(raw_scale)
    ).clamp_min(floor)
    candidate_standardized = (candidate - mean) / scale
    reference_standardized = (reference - mean) / scale
    runs = [
        float(
            sliced_wasserstein(
                candidate_standardized,
                reference_standardized,
                projections=projections,
                seed=seed,
            )
        )
        for seed in seeds
    ]
    return sum(runs) / len(runs), {
        "runs": [{"seed": seed, "distance": value} for seed, value in zip(seeds, runs)],
        "reference_dimensions": int(reference.shape[-1]),
        "reference_samples": int(reference.shape[0]),
        "candidate_samples": int(candidate.shape[0]),
        "scale_floor": float(floor),
        "dimensions_using_scale_floor": int((raw_scale < floor).sum()),
    }


def _weighted_aod_points(
    payload: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = payload["path_mask"].bool()
    direction = direction_vectors(payload["aod_az_deg"], payload["aod_el_deg"])
    per_link_weight = normalized_path_power(payload)
    weights = per_link_weight / max(mask.shape[0], 1)
    return direction[mask], weights[mask]


def _weighted_quantiles(
    projected: torch.Tensor, weights: torch.Tensor, quantiles: torch.Tensor
) -> torch.Tensor:
    values, order = projected.transpose(0, 1).sort(dim=-1)
    sorted_weights = weights[order]
    cumulative = sorted_weights.cumsum(-1)
    cumulative = cumulative / cumulative[:, -1:].clamp_min(1e-20)
    queries = quantiles.unsqueeze(0).expand(values.shape[0], -1).contiguous()
    locations = torch.searchsorted(cumulative.contiguous(), queries, right=False)
    locations = locations.clamp_max(values.shape[1] - 1)
    return values.gather(1, locations)


def _power_weighted_aod_swd(
    candidate: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    *,
    projections: int,
    quantile_count: int,
    seeds: tuple[int, ...],
) -> tuple[float, dict[str, Any]]:
    candidate_points, candidate_weights = _weighted_aod_points(candidate)
    reference_points, reference_weights = _weighted_aod_points(reference)
    quantiles = (
        torch.arange(
            quantile_count,
            device=reference_points.device,
            dtype=reference_points.dtype,
        )
        + 0.5
    ) / quantile_count
    runs: list[float] = []
    for seed in seeds:
        generator = torch.Generator(device=reference_points.device).manual_seed(seed)
        directions = torch.randn(
            projections,
            3,
            generator=generator,
            device=reference_points.device,
            dtype=reference_points.dtype,
        )
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-20)
        candidate_quantiles = _weighted_quantiles(
            candidate_points @ directions.T, candidate_weights, quantiles
        )
        reference_quantiles = _weighted_quantiles(
            reference_points @ directions.T, reference_weights, quantiles
        )
        runs.append(float((candidate_quantiles - reference_quantiles).abs().mean()))
    return sum(runs) / len(runs), {
        "runs": [{"seed": seed, "distance": value} for seed, value in zip(seeds, runs)],
        "projections": projections,
        "weighted_quantiles": quantile_count,
        "candidate_active_paths": int(candidate_points.shape[0]),
        "reference_active_paths": int(reference_points.shape[0]),
        "each_link_has_equal_total_weight": True,
    }


def _power_aod_metrics(
    candidate: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    *,
    projections: int,
    quantile_count: int,
    seeds: tuple[int, ...],
) -> tuple[dict[str, float], dict[str, Any]]:
    path_distance, path_audit = _standardized_swd(
        _path_power_aod_features(candidate),
        _path_power_aod_features(reference),
        projections=projections,
        seeds=seeds,
    )
    link_distance, link_audit = _standardized_swd(
        power_aod_joint_features(candidate),
        power_aod_joint_features(reference),
        projections=projections,
        seeds=seeds,
    )
    weighted_distance, weighted_audit = _power_weighted_aod_swd(
        candidate,
        reference,
        projections=projections,
        quantile_count=quantile_count,
        seeds=seeds,
    )
    return (
        {
            "path_level_power_aod_sliced_wasserstein": path_distance,
            "link_level_power_aod_sliced_wasserstein": link_distance,
            "power_weighted_aod_sliced_wasserstein": weighted_distance,
        },
        {
            "path_level": path_audit,
            "link_level": link_audit,
            "power_weighted_aod": weighted_audit,
        },
    )


def _path_count_audit(
    candidate: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    candidate_count = candidate["path_mask"].sum(-1).long()
    reference_count = reference["path_mask"].sum(-1).long()
    bins = max(candidate["path_mask"].shape[1], reference["path_mask"].shape[1]) + 1
    candidate_histogram = torch.bincount(candidate_count, minlength=bins).float()
    reference_histogram = torch.bincount(reference_count, minlength=bins).float()
    candidate_histogram /= candidate_histogram.sum().clamp_min(1.0)
    reference_histogram /= reference_histogram.sum().clamp_min(1.0)
    return {
        "exact_match_rate": float((candidate_count == reference_count).float().mean()),
        "mean_absolute_count_error": float(
            (candidate_count - reference_count).abs().float().mean()
        ),
        "distribution_total_variation": float(
            0.5 * (candidate_histogram - reference_histogram).abs().sum()
        ),
        "candidate_histogram": candidate_histogram.tolist(),
        "reference_histogram": reference_histogram.tolist(),
    }


def _condition_features(
    conditions: torch.Tensor, validity: torch.Tensor
) -> torch.Tensor:
    continuous = conditions[:, :3].float().clone()
    mean = continuous.mean(0)
    scale = continuous.std(0, unbiased=False).clamp_min(1e-6)
    if bool(validity[:, 2].any()):
        valid_k = validity[:, 2].bool()
        mean[2] = continuous[valid_k, 2].mean()
        scale[2] = continuous[valid_k, 2].std(unbiased=False).clamp_min(1e-6)
    values = (continuous - mean) / scale
    values[:, 2] = torch.where(
        validity[:, 2].bool(), values[:, 2], torch.zeros_like(values[:, 2])
    )
    return values


def _local_condition_groups(
    map_indices: torch.Tensor,
    conditions: torch.Tensor,
    validity: torch.Tensor,
    *,
    minimum_samples: int,
    maximum_groups_per_stratum: int,
) -> list[dict[str, Any]]:
    local_conditions = conditions[map_indices]
    local_validity = validity[map_indices]
    features = _condition_features(local_conditions, local_validity)
    los = local_conditions[:, 3] >= 0.5
    k_valid = local_validity[:, 2].bool()
    groups: list[dict[str, Any]] = []
    group_number = 0
    for current_los, current_k_valid in sorted(
        {(bool(los[index]), bool(k_valid[index])) for index in range(los.numel())}
    ):
        local_indices = torch.nonzero(
            (los == current_los) & (k_valid == current_k_valid), as_tuple=False
        ).flatten()
        if local_indices.numel() < minimum_samples:
            continue
        group_count = min(
            maximum_groups_per_stratum,
            max(1, int(local_indices.numel()) // minimum_samples),
        )
        centered = features[local_indices] - features[local_indices].mean(0, keepdim=True)
        if group_count == 1:
            ordered = local_indices
        else:
            _, _, right = torch.linalg.svd(centered, full_matrices=False)
            ordered = local_indices[torch.argsort(centered @ right[0])]
        for chunk in torch.tensor_split(ordered, group_count):
            if chunk.numel() < minimum_samples:
                continue
            local = features[chunk]
            radius = (local - local.mean(0, keepdim=True)).norm(dim=-1)
            groups.append(
                {
                    "name": f"group_{group_number}",
                    "indices": map_indices[chunk],
                    "samples": int(chunk.numel()),
                    "los": current_los,
                    "k_valid": current_k_valid,
                    "condition_radius_mean": float(radius.mean()),
                    "condition_radius_maximum": float(radius.max()),
                }
            )
            group_number += 1
    return groups


def _weighted_group_average(
    values: list[tuple[int, dict[str, float]]]
) -> dict[str, float] | None:
    if not values:
        return None
    total = sum(samples for samples, _ in values)
    return {
        metric: sum(samples * metrics[metric] for samples, metrics in values) / total
        for metric in METRIC_NAMES
    }


def _condition_input_audit(
    reference: Mapping[str, torch.Tensor],
    candidates: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    reference_conditions = reference["conditions"].float()
    reference_validity = reference["condition_validity"].bool()
    result: dict[str, Any] = {}
    for label, payload in candidates.items():
        conditions = payload.get("target_conditions", payload.get("conditions"))
        validity = payload.get(
            "target_condition_validity", payload.get("condition_validity")
        )
        if conditions is None or validity is None:
            result[label] = {"status": "target_condition_fields_missing"}
            continue
        comparable = reference_validity & validity.bool()
        difference = (conditions.float() - reference_conditions).abs()
        result[label] = {
            "maximum_absolute_difference_on_comparable_fields": float(
                difference[comparable].max() if bool(comparable.any()) else 0.0
            ),
            "los_exact": bool(
                torch.equal(
                    conditions[:, 3] >= 0.5, reference_conditions[:, 3] >= 0.5
                )
            ),
            "k_validity_exact": bool(
                torch.equal(validity[:, 2].bool(), reference_validity[:, 2])
            ),
        }
    return result


def _paired_summaries(
    per_map: Mapping[str, Any],
    *,
    comparison: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for section_index, section in enumerate(EVALUATION_SECTIONS):
        result[section] = {}
        for metric_index, metric in enumerate(METRIC_NAMES):
            scene: list[float] = []
            baseline: list[float] = []
            maps: list[int] = []
            for values in per_map.values():
                if values.get("status") != "ok":
                    continue
                scene_section = values["models"]["b3_scene_id"].get(section)
                comparison_section = values["models"][comparison].get(section)
                if scene_section is None or comparison_section is None:
                    continue
                scene.append(float(scene_section[metric]))
                baseline.append(float(comparison_section[metric]))
                maps.append(int(values["map_id"]))
            if not maps:
                result[section][metric] = {"status": "no_common_map_support"}
                continue
            summary = paired_map_summary(
                scene,
                baseline,
                bootstrap_samples=bootstrap_samples,
                seed=seed + section_index * 100 + metric_index,
            )
            summary["map_ids"] = maps
            summary["comparison"] = comparison
            summary["sign"] = (
                f"positive {comparison} minus B3-S-ID means B3-S-ID is closer to real data"
            )
            result[section][metric] = summary
    return result


def _validate(source: Mapping[str, Any]) -> None:
    for key in ("reference", "correct_map", "shuffled_map", "physics_only_baseline"):
        if not isinstance(source.get(key), Mapping):
            raise ValueError(f"generated artifact must contain mapping {key}")
    required = (
        "path_mask",
        "path_powers_linear",
        "aod_az_deg",
        "aod_el_deg",
    )
    for artifact_key in ("reference", "correct_map", "shuffled_map", "physics_only_baseline"):
        missing = [key for key in required if key not in source[artifact_key]]
        if missing:
            raise ValueError(f"{artifact_key} is missing {missing}")
    for key in ("map_id", "conditions", "condition_validity"):
        if key not in source["reference"]:
            raise ValueError(f"reference is missing {key}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-Map conditional Power-AoD fidelity for B3, B3-S-ID and shuffled Map IDs."
    )
    parser.add_argument("--generated", required=True)
    parser.add_argument("--minimum-per-map", type=int, default=12)
    parser.add_argument("--minimum-local-group", type=int, default=4)
    parser.add_argument("--maximum-local-groups-per-stratum", type=int, default=4)
    parser.add_argument("--minimum-common-path-count", type=int, default=4)
    parser.add_argument("--projections", type=int, default=128)
    parser.add_argument("--projection-seeds", default="0,1,2")
    parser.add_argument("--weighted-quantiles", type=int, default=512)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    projection_seeds = tuple(
        int(value) for value in args.projection_seeds.split(",") if value.strip()
    )
    positive = (
        args.minimum_per_map,
        args.minimum_local_group,
        args.maximum_local_groups_per_stratum,
        args.minimum_common_path_count,
        args.projections,
        args.weighted_quantiles,
        args.bootstrap_samples,
    )
    if min(positive) < 1 or not projection_seeds:
        parser.error("sample, projection, quantile and bootstrap values must be positive")

    torch.set_grad_enabled(False)
    device = resolve_device(args.device)
    source = torch.load(args.generated, map_location="cpu", weights_only=False)
    _validate(source)
    reference = source["reference"]
    candidates = {
        label: source[artifact_key] for label, artifact_key in ARTIFACT_KEYS.items()
    }
    sample_count = int(reference["map_id"].numel())
    for label, payload in candidates.items():
        if payload["path_mask"].shape[0] != sample_count:
            raise ValueError(f"{label} sample count does not match reference")
    map_ids = reference["map_id"].long()
    conditions = reference["conditions"].float()
    validity = reference["condition_validity"].bool()
    maps = sorted(int(value) for value in torch.unique(map_ids))
    condition_audit = _condition_input_audit(reference, candidates)

    per_map: dict[str, Any] = {}
    for map_number, map_id in enumerate(maps, start=1):
        indices = torch.nonzero(map_ids == map_id, as_tuple=False).flatten()
        if indices.numel() < args.minimum_per_map:
            per_map[f"map_{map_id}"] = {
                "map_id": map_id,
                "samples": int(indices.numel()),
                "status": "insufficient_map_samples",
            }
            continue
        groups = _local_condition_groups(
            indices,
            conditions,
            validity,
            minimum_samples=args.minimum_local_group,
            maximum_groups_per_stratum=args.maximum_local_groups_per_stratum,
        )
        reference_map = _subset(reference, indices, device)
        candidate_maps = {
            label: _subset(payload, indices, device)
            for label, payload in candidates.items()
        }
        reference_count = reference_map["path_mask"].sum(-1)
        common_count = torch.ones_like(reference_count, dtype=torch.bool)
        for payload in candidate_maps.values():
            common_count &= payload["path_mask"].sum(-1) == reference_count
        common_indices = torch.nonzero(common_count, as_tuple=False).flatten()
        model_reports: dict[str, Any] = {}
        for model_number, (label, candidate_map) in enumerate(candidate_maps.items()):
            unconditional, unconditional_audit = _power_aod_metrics(
                candidate_map,
                reference_map,
                projections=args.projections,
                quantile_count=args.weighted_quantiles,
                seeds=projection_seeds,
            )
            local_values: list[tuple[int, dict[str, float]]] = []
            local_reports: list[dict[str, Any]] = []
            for group_number, group in enumerate(groups):
                local_positions = torch.searchsorted(indices, group["indices"])
                reference_group = _subset(reference_map, local_positions, device)
                candidate_group = _subset(candidate_map, local_positions, device)
                metrics, audit = _power_aod_metrics(
                    candidate_group,
                    reference_group,
                    projections=args.projections,
                    quantile_count=args.weighted_quantiles,
                    seeds=tuple(
                        seed
                        + args.seed
                        + map_number * 10000
                        + group_number * 100
                        for seed in projection_seeds
                    ),
                )
                local_values.append((group["samples"], metrics))
                local_reports.append(
                    {
                        **{key: value for key, value in group.items() if key != "indices"},
                        "metrics": metrics,
                        "metric_audit": audit,
                    }
                )
            controlled: dict[str, float] | None = None
            controlled_audit: dict[str, Any] | None = None
            if common_indices.numel() >= args.minimum_common_path_count:
                candidate_controlled = _subset(candidate_map, common_indices, device)
                reference_controlled = _subset(reference_map, common_indices, device)
                controlled, controlled_audit = _power_aod_metrics(
                    candidate_controlled,
                    reference_controlled,
                    projections=args.projections,
                    quantile_count=args.weighted_quantiles,
                    seeds=tuple(seed + args.seed + map_number * 10000 for seed in projection_seeds),
                )
            model_reports[label] = {
                "unconditional": unconditional,
                "unconditional_metric_audit": unconditional_audit,
                "local_condition_weighted": _weighted_group_average(local_values),
                "local_condition_groups": local_reports,
                "common_path_count_controlled": controlled,
                "common_path_count_controlled_metric_audit": controlled_audit,
                "path_count": _path_count_audit(candidate_map, reference_map),
            }
        eligible_local_samples = sum(group["samples"] for group in groups)
        per_map[f"map_{map_id}"] = {
            "map_id": map_id,
            "samples": int(indices.numel()),
            "status": "ok",
            "support": {
                "local_condition_groups": len(groups),
                "eligible_local_samples": eligible_local_samples,
                "local_condition_coverage": eligible_local_samples / int(indices.numel()),
                "common_path_count_samples": int(common_indices.numel()),
                "common_path_count_rate": float(common_count.float().mean()),
                "los_samples": int((reference_map["conditions"][:, 3] >= 0.5).sum()),
                "nlos_samples": int((reference_map["conditions"][:, 3] < 0.5).sum()),
                "k_valid_samples": int(reference_map["condition_validity"][:, 2].sum()),
                "shuffled_target_map_ids": (
                    torch.unique(source["wrong_map_id"][indices]).tolist()
                    if isinstance(source.get("wrong_map_id"), torch.Tensor)
                    else None
                ),
            },
            "models": model_reports,
        }
        print(
            json.dumps(
                {
                    "event": "map_complete",
                    "map": map_id,
                    "map_index": map_number,
                    "maps_total": len(maps),
                    "samples": int(indices.numel()),
                    "local_groups": len(groups),
                    "common_path_count_samples": int(common_indices.numel()),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summaries = {
        "b3_scene_id_vs_b3": _paired_summaries(
            per_map,
            comparison="b3",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 100000,
        ),
        "b3_scene_id_vs_shuffled_map": _paired_summaries(
            per_map,
            comparison="shuffled_map",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 200000,
        ),
    }
    evaluated = [values for values in per_map.values() if values.get("status") == "ok"]
    path_count_summary = {
        label: {
            "mean_exact_match_rate": (
                sum(values["models"][label]["path_count"]["exact_match_rate"] for values in evaluated)
                / max(len(evaluated), 1)
            ),
            "mean_absolute_count_error": (
                sum(
                    values["models"][label]["path_count"]["mean_absolute_count_error"]
                    for values in evaluated
                )
                / max(len(evaluated), 1)
            ),
            "mean_distribution_total_variation": (
                sum(
                    values["models"][label]["path_count"]["distribution_total_variation"]
                    for values in evaluated
                )
                / max(len(evaluated), 1)
            ),
        }
        for label in MODEL_LABELS
    }
    report = {
        "format_version": 1,
        "diagnostic": "per_map_conditional_power_aod_fidelity",
        "source": str(Path(args.generated).resolve()),
        "scene_plan_fingerprint": source.get("scene_plan_fingerprint"),
        "device": str(device),
        "samples": sample_count,
        "maps_total": len(maps),
        "maps_evaluated": len(evaluated),
        "metric_direction": "lower_is_better",
        "configuration": {
            "minimum_per_map": args.minimum_per_map,
            "minimum_local_group": args.minimum_local_group,
            "maximum_local_groups_per_stratum": args.maximum_local_groups_per_stratum,
            "minimum_common_path_count": args.minimum_common_path_count,
            "projections": args.projections,
            "projection_seeds": list(projection_seeds),
            "weighted_quantiles": args.weighted_quantiles,
            "bootstrap_samples": args.bootstrap_samples,
        },
        "condition_input_audit": condition_audit,
        "path_count_control": {
            "unconditional_metrics": "retain path-count errors as part of total fidelity",
            "common_subset": (
                "Same links where RayVerse, B3, B3-S-ID and shuffled-Map generations all "
                "have exactly the RayVerse path count; every model uses identical samples."
            ),
            "warning": (
                "Controlled metrics must be interpreted with common_path_count_rate; low "
                "support is reported rather than silently treated as reliable evidence."
            ),
        },
        "support_summary": {
            "maps_without_local_groups": sum(
                values["support"]["local_condition_groups"] == 0 for values in evaluated
            ),
            "maps_without_controlled_metrics": sum(
                values["models"]["b3_scene_id"]["common_path_count_controlled"] is None
                for values in evaluated
            ),
            "mean_local_condition_coverage": (
                sum(values["support"]["local_condition_coverage"] for values in evaluated)
                / max(len(evaluated), 1)
            ),
            "mean_common_path_count_rate": (
                sum(values["support"]["common_path_count_rate"] for values in evaluated)
                / max(len(evaluated), 1)
            ),
        },
        "path_count_summary": path_count_summary,
        "summaries": summaries,
        "per_map": per_map,
        "cautions": [
            "Map is the paired bootstrap unit; each Map contributes once.",
            "This artifact represents one generation draw/seed; repeat generation seeds before a final paper claim.",
            "Path-level observations are not treated as independent inference units; inference is across Maps.",
            "Local groups with insufficient support are excluded and their coverage is reported.",
            "The Map-independent Delay-Power approximation is not tested by this Power-AoD-only diagnostic.",
        ],
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "per_map_power_aod_complete",
                "output": str(Path(args.output).resolve()),
                "samples": sample_count,
                "maps_evaluated": len(evaluated),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
