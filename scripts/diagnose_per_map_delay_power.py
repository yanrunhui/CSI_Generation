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
from physics_csi.evaluation.distribution import sliced_wasserstein
from physics_csi.evaluation.scene_diagnostics import paired_map_summary
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


MODEL_KEYS = {
    "b3_scene_id": "correct_map",
    "b3": "physics_only_baseline",
    "shuffled_map": "shuffled_map",
}
MODEL_LABELS = tuple(MODEL_KEYS)
PAYLOAD_KEYS = (
    "path_mask",
    "path_delays_ns",
    "path_powers_linear",
    "conditions",
    "condition_validity",
)
METRICS = (
    "path_level_delay_power_sliced_wasserstein",
    "link_level_delay_power_sliced_wasserstein",
    "power_weighted_delay_wasserstein",
)
FIDELITY_SECTIONS = (
    "unconditional",
    "local_condition_weighted",
    "common_path_count_controlled",
)


def _stack(records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([record[key] for record in records]) for key in PAYLOAD_KEYS}


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


def _relative_delay_and_fraction(
    payload: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = payload["path_mask"].bool()
    delays = payload["path_delays_ns"].float()
    first = torch.where(mask, delays, torch.full_like(delays, torch.inf)).amin(-1)
    relative = (delays - first.unsqueeze(-1)).clamp_min(0.0)
    power = payload["path_powers_linear"].float() * mask
    fraction = power / power.sum(-1, keepdim=True).clamp_min(1e-20)
    return mask, relative, fraction


def _canonical_order(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask = payload["path_mask"].bool()
    delay = payload["path_delays_ns"].float()
    return torch.argsort(
        torch.where(mask, delay, torch.full_like(delay, torch.inf)),
        dim=-1,
        stable=True,
    )


def _path_features(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask, relative, fraction = _relative_delay_and_fraction(payload)
    power_db = 10.0 * torch.log10(fraction.clamp_min(1e-20))
    return torch.stack([relative, power_db], dim=-1)[mask]


def _link_features(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask, relative, fraction = _relative_delay_and_fraction(payload)
    order = _canonical_order(payload)
    mask = mask.gather(1, order)
    relative = relative.gather(1, order) * mask
    fraction = fraction.gather(1, order) * mask
    return torch.cat([mask.float(), relative, fraction], dim=-1)


class FixedNormalizer:
    def __init__(
        self,
        path_mean: torch.Tensor,
        path_scale: torch.Tensor,
        link_mean: torch.Tensor,
        link_scale: torch.Tensor,
        delay_scale: torch.Tensor,
        *,
        source_split: str,
        samples: int,
    ) -> None:
        self.path_mean = path_mean
        self.path_scale = path_scale
        self.link_mean = link_mean
        self.link_scale = link_scale
        self.delay_scale = delay_scale
        self.source_split = source_split
        self.samples = samples

    @staticmethod
    def _moments(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        mean = values.mean(0, keepdim=True)
        raw_scale = values.std(0, unbiased=False, keepdim=True)
        positive = raw_scale[torch.isfinite(raw_scale) & (raw_scale > 0)]
        typical = positive.median() if positive.numel() else torch.ones(())
        floor = torch.maximum(typical * 0.05, torch.tensor(1e-6))
        scale = torch.where(
            torch.isfinite(raw_scale), raw_scale, torch.zeros_like(raw_scale)
        ).clamp_min(floor)
        return mean, scale, {
            "dimensions": int(values.shape[-1]),
            "raw_scale_minimum": float(raw_scale.min()),
            "raw_scale_median": float(raw_scale.median()),
            "raw_scale_maximum": float(raw_scale.max()),
            "scale_floor": float(floor),
            "dimensions_using_floor": int((raw_scale < floor).sum()),
        }

    @classmethod
    def from_dataset(cls, dataset: Any, *, split: str) -> "FixedNormalizer":
        path_chunks: list[torch.Tensor] = []
        link_chunks: list[torch.Tensor] = []
        delay_chunks: list[torch.Tensor] = []
        for index in range(len(dataset)):
            record = dataset[index]
            payload = {
                key: record[key].unsqueeze(0)
                for key in ("path_mask", "path_delays_ns", "path_powers_linear")
            }
            path_chunks.append(_path_features(payload).cpu())
            link_chunks.append(_link_features(payload).cpu())
            mask, relative, _ = _relative_delay_and_fraction(payload)
            delay_chunks.append(relative[mask].cpu())
            if (index + 1) % 5000 == 0 or index + 1 == len(dataset):
                print(
                    json.dumps(
                        {
                            "event": "normalization_scan",
                            "samples_complete": index + 1,
                            "samples_total": len(dataset),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        path = torch.cat(path_chunks)
        link = torch.cat(link_chunks)
        delays = torch.cat(delay_chunks)
        path_mean, path_scale, _ = cls._moments(path)
        link_mean, link_scale, _ = cls._moments(link)
        positive_delay_scale = delays.std(unbiased=False)
        delay_scale = positive_delay_scale.clamp_min(1e-6)
        return cls(
            path_mean,
            path_scale,
            link_mean,
            link_scale,
            delay_scale,
            source_split=split,
            samples=len(dataset),
        )

    def to(self, device: torch.device) -> "FixedNormalizer":
        return FixedNormalizer(
            self.path_mean.to(device),
            self.path_scale.to(device),
            self.link_mean.to(device),
            self.link_scale.to(device),
            self.delay_scale.to(device),
            source_split=self.source_split,
            samples=self.samples,
        )

    def path(self, payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return (_path_features(payload) - self.path_mean) / self.path_scale

    def link(self, payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
        values = _link_features(payload)
        if values.shape[-1] != self.link_mean.shape[-1]:
            raise ValueError("generated and normalization data have different maximum path counts")
        return (values - self.link_mean) / self.link_scale

    def audit(self) -> dict[str, Any]:
        return {
            "source_split": self.source_split,
            "source_samples": self.samples,
            "path_mean": self.path_mean.cpu().flatten().tolist(),
            "path_scale": self.path_scale.cpu().flatten().tolist(),
            "link_dimensions": int(self.link_mean.shape[-1]),
            "link_scale_minimum": float(self.link_scale.min()),
            "link_scale_median": float(self.link_scale.median()),
            "link_scale_maximum": float(self.link_scale.max()),
            "relative_delay_scale_ns": float(self.delay_scale),
            "fit_once_and_reused_for_all_maps_groups_and_models": True,
        }


def _weighted_quantiles(
    values: torch.Tensor, weights: torch.Tensor, quantiles: torch.Tensor
) -> torch.Tensor:
    order = torch.argsort(values)
    sorted_values = values[order]
    cumulative = weights[order].cumsum(0)
    cumulative = cumulative / cumulative[-1].clamp_min(1e-20)
    locations = torch.searchsorted(cumulative.contiguous(), quantiles, right=False)
    return sorted_values[locations.clamp_max(sorted_values.numel() - 1)]


def _power_weighted_delay_distance(
    candidate: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    normalizer: FixedNormalizer,
    *,
    quantile_count: int,
) -> float:
    def values(payload: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        mask, relative, fraction = _relative_delay_and_fraction(payload)
        weights = fraction / max(mask.shape[0], 1)
        return relative[mask] / normalizer.delay_scale, weights[mask]

    candidate_values, candidate_weights = values(candidate)
    reference_values, reference_weights = values(reference)
    quantiles = (
        torch.arange(
            quantile_count,
            device=reference_values.device,
            dtype=reference_values.dtype,
        )
        + 0.5
    ) / quantile_count
    candidate_quantiles = _weighted_quantiles(
        candidate_values, candidate_weights, quantiles
    )
    reference_quantiles = _weighted_quantiles(
        reference_values, reference_weights, quantiles
    )
    return float((candidate_quantiles - reference_quantiles).abs().mean())


def _delay_power_metrics(
    candidate: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    normalizer: FixedNormalizer,
    *,
    projections: int,
    projection_seeds: tuple[int, ...],
    quantile_count: int,
) -> dict[str, float]:
    candidate_path = normalizer.path(candidate)
    reference_path = normalizer.path(reference)
    candidate_link = normalizer.link(candidate)
    reference_link = normalizer.link(reference)

    def swd(first: torch.Tensor, second: torch.Tensor) -> float:
        runs = [
            float(
                sliced_wasserstein(
                    first,
                    second,
                    projections=projections,
                    seed=seed,
                )
            )
            for seed in projection_seeds
        ]
        return sum(runs) / len(runs)

    return {
        "path_level_delay_power_sliced_wasserstein": swd(
            candidate_path, reference_path
        ),
        "link_level_delay_power_sliced_wasserstein": swd(
            candidate_link, reference_link
        ),
        "power_weighted_delay_wasserstein": _power_weighted_delay_distance(
            candidate,
            reference,
            normalizer,
            quantile_count=quantile_count,
        ),
    }


def _condition_features(
    conditions: torch.Tensor, validity: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    values = conditions[:, :3].float() / std[:3].clamp_min(1e-6)
    values[:, 2] = torch.where(
        validity[:, 2].bool(), values[:, 2], torch.zeros_like(values[:, 2])
    )
    return values


def _one_to_one_match(
    query: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    *,
    condition_std: torch.Tensor,
    maximum_distance: float,
    require_equal_path_count: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_features = _condition_features(
        query["conditions"], query["condition_validity"], condition_std
    )
    candidate_features = _condition_features(
        candidate["conditions"], candidate["condition_validity"], condition_std
    )
    distances = torch.cdist(query_features, candidate_features)
    query_los = query["conditions"][:, 3] >= 0.5
    candidate_los = candidate["conditions"][:, 3] >= 0.5
    query_k_valid = query["condition_validity"][:, 2].bool()
    candidate_k_valid = candidate["condition_validity"][:, 2].bool()
    compatible = (
        (query_los[:, None] == candidate_los[None, :])
        & (query_k_valid[:, None] == candidate_k_valid[None, :])
        & torch.isfinite(distances)
        & (distances <= maximum_distance)
    )
    if require_equal_path_count:
        query_count = query["path_mask"].sum(-1)
        candidate_count = candidate["path_mask"].sum(-1)
        compatible &= query_count[:, None] == candidate_count[None, :]
    distances = distances.masked_fill(~compatible, torch.inf).cpu()
    row_order = torch.argsort(distances.amin(1))
    used = torch.zeros(candidate["conditions"].shape[0], dtype=torch.bool)
    first: list[int] = []
    second: list[int] = []
    matched_distance: list[float] = []
    for first_index in row_order.tolist():
        for second_index in torch.argsort(distances[first_index]).tolist():
            distance = float(distances[first_index, second_index])
            if not math.isfinite(distance) or distance > maximum_distance:
                break
            if not bool(used[second_index]):
                used[second_index] = True
                first.append(first_index)
                second.append(second_index)
                matched_distance.append(distance)
                break
    return (
        torch.tensor(first, dtype=torch.long),
        torch.tensor(second, dtype=torch.long),
        torch.tensor(matched_distance, dtype=torch.float32),
    )


def _triplet_match(
    anchor: Mapping[str, torch.Tensor],
    other: Mapping[str, torch.Tensor],
    *,
    condition_std: torch.Tensor,
    maximum_distance: float,
    samples: int,
    maximum_attempts: int,
    generator: torch.Generator,
) -> dict[str, Any] | None:
    anchor_count = anchor["conditions"].shape[0]
    if anchor_count < samples * 2 or other["conditions"].shape[0] < samples:
        return None
    for attempt in range(1, maximum_attempts + 1):
        permutation = torch.randperm(anchor_count, generator=generator)
        midpoint = anchor_count // 2
        first_pool = permutation[:midpoint]
        second_pool = permutation[midpoint:]
        first_payload = {key: value[first_pool] for key, value in anchor.items()}
        second_payload = {key: value[second_pool] for key, value in anchor.items()}
        first_ab, second_local, distances_ab = _one_to_one_match(
            first_payload,
            second_payload,
            condition_std=condition_std,
            maximum_distance=maximum_distance,
            require_equal_path_count=True,
        )
        first_ac, other_local, distances_ac = _one_to_one_match(
            first_payload,
            other,
            condition_std=condition_std,
            maximum_distance=maximum_distance,
            require_equal_path_count=True,
        )
        ab = {
            int(a): (int(b), float(distance))
            for a, b, distance in zip(
                first_ab.tolist(), second_local.tolist(), distances_ab.tolist()
            )
        }
        ac = {
            int(a): (int(c), float(distance))
            for a, c, distance in zip(
                first_ac.tolist(), other_local.tolist(), distances_ac.tolist()
            )
        }
        common = sorted(set(ab) & set(ac))
        if len(common) < samples:
            continue
        best = sorted(common, key=lambda index: max(ab[index][1], ac[index][1]))
        candidate_window = best[: min(len(best), samples * 2)]
        order = torch.randperm(len(candidate_window), generator=generator)[:samples]
        selected = [candidate_window[int(index)] for index in order]
        return {
            "anchor_indices": torch.tensor(
                [int(first_pool[index]) for index in selected], dtype=torch.long
            ),
            "within_indices": torch.tensor(
                [int(second_pool[ab[index][0]]) for index in selected], dtype=torch.long
            ),
            "cross_indices": torch.tensor(
                [ac[index][0] for index in selected], dtype=torch.long
            ),
            "within_distances": torch.tensor(
                [ab[index][1] for index in selected], dtype=torch.float32
            ),
            "cross_distances": torch.tensor(
                [ac[index][1] for index in selected], dtype=torch.float32
            ),
            "attempt": attempt,
            "available_common_matches": len(common),
        }
    return None


def _balanced_map_pairs(
    maps: list[int], maximum_pairs: int, generator: torch.Generator
) -> list[tuple[int, int]]:
    candidates = list(itertools.combinations(maps, 2))
    random_order = torch.randperm(len(candidates), generator=generator).tolist()
    candidates = [candidates[index] for index in random_order]
    if maximum_pairs <= 0 or maximum_pairs >= len(candidates):
        return candidates
    degree = {map_id: 0 for map_id in maps}
    selected: list[tuple[int, int]] = []
    remaining = candidates
    while remaining and len(selected) < maximum_pairs:
        best_index = min(
            range(len(remaining)),
            key=lambda index: (
                degree[remaining[index][0]] + degree[remaining[index][1]],
                max(degree[remaining[index][0]], degree[remaining[index][1]]),
                index,
            ),
        )
        pair = remaining.pop(best_index)
        selected.append(pair)
        degree[pair[0]] += 1
        degree[pair[1]] += 1
    return selected


def _map_cluster_real_summary(
    pair_reports: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
    equivalence_margin: float,
) -> dict[str, Any]:
    maps = sorted(
        {report["first_map"] for report in pair_reports}
        | {report["second_map"] for report in pair_reports}
    )
    map_index = {map_id: index for index, map_id in enumerate(maps)}
    first = torch.tensor([map_index[report["first_map"]] for report in pair_reports])
    second = torch.tensor([map_index[report["second_map"]] for report in pair_reports])
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(
        len(maps), (bootstrap_samples, len(maps)), generator=generator
    )
    multiplicity = torch.zeros(bootstrap_samples, len(maps), dtype=torch.float64)
    multiplicity.scatter_add_(1, draws, torch.ones_like(draws, dtype=torch.float64))
    weights = multiplicity[:, first] * multiplicity[:, second]
    valid = weights.sum(-1) > 0
    quantiles = torch.tensor([0.025, 0.975], dtype=torch.float64)
    result: dict[str, Any] = {}
    for metric in METRICS:
        same = torch.tensor(
            [report["same_map_mean"][metric] for report in pair_reports],
            dtype=torch.float64,
        )
        cross = torch.tensor(
            [report["cross_map_mean"][metric] for report in pair_reports],
            dtype=torch.float64,
        )
        same_boot = (weights[valid] * same).sum(-1) / weights[valid].sum(-1)
        cross_boot = (weights[valid] * cross).sum(-1) / weights[valid].sum(-1)
        delta_boot = cross_boot - same_boot
        relative_boot = delta_boot / same_boot.clamp_min(1e-20)
        delta_interval = torch.quantile(delta_boot, quantiles)
        relative_interval = torch.quantile(relative_boot, quantiles)
        observed_same = float(same.mean())
        observed_cross = float(cross.mean())
        relative_observed = (observed_cross - observed_same) / max(observed_same, 1e-20)
        statistical = (
            "positive_cross_map_difference"
            if float(delta_interval[0]) > 0
            else "negative"
            if float(delta_interval[1]) < 0
            else "inconclusive"
        )
        equivalence = (
            "practically_equivalent"
            if float(relative_interval[0]) >= -equivalence_margin
            and float(relative_interval[1]) <= equivalence_margin
            else "meaningful_map_dependence"
            if float(relative_interval[0]) > equivalence_margin
            else "inconclusive"
        )
        result[metric] = {
            "map_pairs": len(pair_reports),
            "maps": len(maps),
            "same_map_mean": observed_same,
            "cross_map_mean": observed_cross,
            "delta_scene_cross_minus_same": observed_cross - observed_same,
            "relative_scene_effect": relative_observed,
            "delta_cluster_bootstrap_95_percent_ci": delta_interval.tolist(),
            "relative_effect_cluster_bootstrap_95_percent_ci": relative_interval.tolist(),
            "map_pair_win_rate": float(((cross - same) > 0).double().mean()),
            "statistical_evidence": statistical,
            "equivalence_margin": equivalence_margin,
            "equivalence_evidence": equivalence,
            "valid_bootstrap_replicates": int(valid.sum()),
        }
    return result


def _path_count_audit(
    candidate: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]
) -> dict[str, float]:
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
    }


def _local_groups(
    map_indices: torch.Tensor,
    conditions: torch.Tensor,
    validity: torch.Tensor,
    *,
    minimum_samples: int,
    maximum_groups_per_stratum: int,
) -> list[dict[str, Any]]:
    local_conditions = conditions[map_indices]
    local_validity = validity[map_indices]
    continuous = local_conditions[:, :3].float()
    mean = continuous.mean(0)
    scale = continuous.std(0, unbiased=False).clamp_min(1e-6)
    valid_k = local_validity[:, 2].bool()
    if bool(valid_k.any()):
        mean[2] = continuous[valid_k, 2].mean()
        scale[2] = continuous[valid_k, 2].std(unbiased=False).clamp_min(1e-6)
    features = (continuous - mean) / scale
    features[:, 2] = torch.where(valid_k, features[:, 2], torch.zeros_like(features[:, 2]))
    los = local_conditions[:, 3] >= 0.5
    groups: list[dict[str, Any]] = []
    group_number = 0
    for current_los, current_k_valid in sorted(
        {(bool(los[index]), bool(valid_k[index])) for index in range(los.numel())}
    ):
        local = torch.nonzero(
            (los == current_los) & (valid_k == current_k_valid), as_tuple=False
        ).flatten()
        if local.numel() < minimum_samples:
            continue
        group_count = min(
            maximum_groups_per_stratum,
            max(1, int(local.numel()) // minimum_samples),
        )
        centered = features[local] - features[local].mean(0, keepdim=True)
        if group_count == 1:
            ordered = local
        else:
            _, _, right = torch.linalg.svd(centered, full_matrices=False)
            ordered = local[torch.argsort(centered @ right[0])]
        for chunk in torch.tensor_split(ordered, group_count):
            if chunk.numel() < minimum_samples:
                continue
            radius = (
                features[chunk] - features[chunk].mean(0, keepdim=True)
            ).norm(dim=-1)
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


def _weighted_metric_average(
    values: list[tuple[int, dict[str, float]]]
) -> dict[str, float] | None:
    if not values:
        return None
    total = sum(samples for samples, _ in values)
    return {
        metric: sum(samples * metrics[metric] for samples, metrics in values) / total
        for metric in METRICS
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
            result[label] = {"status": "missing_target_conditions"}
            continue
        comparable = reference_validity & validity.bool()
        difference = (conditions.float() - reference_conditions).abs()
        result[label] = {
            "maximum_absolute_difference_on_comparable_fields": float(
                difference[comparable].max() if bool(comparable.any()) else 0.0
            ),
            "los_exact": bool(
                torch.equal(conditions[:, 3] >= 0.5, reference_conditions[:, 3] >= 0.5)
            ),
            "k_validity_exact": bool(
                torch.equal(validity[:, 2].bool(), reference_validity[:, 2])
            ),
        }
    return result


def _paired_fidelity_summary(
    per_map: Mapping[str, Any],
    *,
    comparison: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for section_index, section in enumerate(FIDELITY_SECTIONS):
        result[section] = {}
        for metric_index, metric in enumerate(METRICS):
            scene: list[float] = []
            other: list[float] = []
            map_ids: list[int] = []
            for values in per_map.values():
                if values.get("status") != "ok":
                    continue
                scene_section = values["models"]["b3_scene_id"].get(section)
                other_section = values["models"][comparison].get(section)
                if scene_section is None or other_section is None:
                    continue
                scene.append(float(scene_section[metric]))
                other.append(float(other_section[metric]))
                map_ids.append(int(values["map_id"]))
            if not map_ids:
                result[section][metric] = {"status": "no_common_map_support"}
                continue
            summary = paired_map_summary(
                scene,
                other,
                bootstrap_samples=bootstrap_samples,
                seed=seed + section_index * 100 + metric_index,
            )
            summary["map_ids"] = map_ids
            summary["comparison"] = comparison
            result[section][metric] = summary
    return result


def _validate_generated(source: Mapping[str, Any]) -> None:
    for key in ("reference", *MODEL_KEYS.values()):
        if not isinstance(source.get(key), Mapping):
            raise ValueError(f"generated artifact must contain mapping {key}")
    for key in ("map_id", "conditions", "condition_validity", *PAYLOAD_KEYS[:3]):
        if key not in source["reference"]:
            raise ValueError(f"reference is missing {key}")
    for artifact_key in MODEL_KEYS.values():
        missing = [key for key in PAYLOAD_KEYS[:3] if key not in source[artifact_key]]
        if missing:
            raise ValueError(f"{artifact_key} is missing {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Real Delay-Power scene discriminability plus per-Map B3/B3-S-ID fidelity."
        )
    )
    parser.add_argument("--generated", required=True)
    parser.add_argument("--scene-plan", required=True)
    parser.add_argument("--prepared-root", default="artifacts/prepared_d2los")
    parser.add_argument("--normalization-split", choices=["train", "validation"], default="train")
    parser.add_argument("--real-split", choices=["validation", "test"], default="test")
    parser.add_argument("--maximum-real-samples-per-map", type=int, default=256)
    parser.add_argument("--maximum-real-map-pairs", type=int, default=100)
    parser.add_argument("--real-samples-per-comparison", type=int, default=32)
    parser.add_argument("--real-repeats", type=int, default=20)
    parser.add_argument("--real-minimum-successful-repeats", type=int, default=10)
    parser.add_argument("--maximum-match-attempts", type=int, default=200)
    parser.add_argument("--maximum-condition-distance", type=float, default=1.0)
    parser.add_argument("--equivalence-margin", type=float, default=0.05)
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
        args.maximum_real_samples_per_map,
        args.real_samples_per_comparison,
        args.real_repeats,
        args.real_minimum_successful_repeats,
        args.maximum_match_attempts,
        args.minimum_per_map,
        args.minimum_local_group,
        args.maximum_local_groups_per_stratum,
        args.minimum_common_path_count,
        args.projections,
        args.weighted_quantiles,
        args.bootstrap_samples,
    )
    if min(positive) < 1 or not projection_seeds:
        parser.error("sample, repeat, projection, quantile and bootstrap values must be positive")
    if not 0 < args.equivalence_margin < 1:
        parser.error("--equivalence-margin must be between zero and one")
    if args.maximum_condition_distance <= 0:
        parser.error("--maximum-condition-distance must be positive")
    if args.real_minimum_successful_repeats > args.real_repeats:
        parser.error("minimum successful repeats cannot exceed requested real repeats")

    torch.set_grad_enabled(False)
    device = resolve_device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    plan = load_seen_map_spatial_plan(args.scene_plan)
    datasets = make_scene_id_datasets(args.prepared_root, plan)
    normalizer_cpu = FixedNormalizer.from_dataset(
        datasets[args.normalization_split], split=args.normalization_split
    )
    normalizer = normalizer_cpu.to(device)

    real_dataset = datasets[args.real_split]
    records_by_map: dict[int, list[dict[str, Any]]] = {}
    for index in range(len(real_dataset)):
        record = real_dataset[index]
        records_by_map.setdefault(int(record["map_id"]), []).append(record)
        if (index + 1) % 1000 == 0 or index + 1 == len(real_dataset):
            print(
                json.dumps(
                    {
                        "event": "real_dataset_loading",
                        "samples_complete": index + 1,
                        "samples_total": len(real_dataset),
                        "maps_seen": len(records_by_map),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    for map_id, records in records_by_map.items():
        if len(records) > args.maximum_real_samples_per_map:
            selection = torch.randperm(len(records), generator=generator)[
                : args.maximum_real_samples_per_map
            ]
            records_by_map[map_id] = [records[int(index)] for index in selection]
    real_payloads = {
        map_id: _stack(records) for map_id, records in records_by_map.items()
    }
    del records_by_map
    real_maps = sorted(real_payloads)
    all_real_conditions = torch.cat(
        [payload["conditions"].float() for payload in real_payloads.values()]
    )
    condition_std = all_real_conditions.std(0, unbiased=False).clamp_min(1e-6)
    selected_pairs = _balanced_map_pairs(
        real_maps, args.maximum_real_map_pairs, generator
    )
    real_pair_reports: list[dict[str, Any]] = []
    real_exclusions: list[dict[str, Any]] = []
    for pair_number, (first_map, second_map) in enumerate(selected_pairs, start=1):
        repeats: list[dict[str, Any]] = []
        for repeat in range(args.real_repeats):
            match = _triplet_match(
                real_payloads[first_map],
                real_payloads[second_map],
                condition_std=condition_std,
                maximum_distance=args.maximum_condition_distance,
                samples=args.real_samples_per_comparison,
                maximum_attempts=args.maximum_match_attempts,
                generator=generator,
            )
            if match is None:
                continue
            first = _subset(real_payloads[first_map], match["anchor_indices"], device)
            within = _subset(real_payloads[first_map], match["within_indices"], device)
            cross = _subset(real_payloads[second_map], match["cross_indices"], device)
            repeat_seeds = tuple(
                seed + args.seed + pair_number * 10000 + repeat * 100
                for seed in projection_seeds
            )
            same_metrics = _delay_power_metrics(
                first,
                within,
                normalizer,
                projections=args.projections,
                projection_seeds=repeat_seeds,
                quantile_count=args.weighted_quantiles,
            )
            cross_metrics = _delay_power_metrics(
                first,
                cross,
                normalizer,
                projections=args.projections,
                projection_seeds=repeat_seeds,
                quantile_count=args.weighted_quantiles,
            )
            repeats.append(
                {
                    "repeat": repeat,
                    "matching_attempt": match["attempt"],
                    "available_common_matches": match["available_common_matches"],
                    "same_map": same_metrics,
                    "cross_map": cross_metrics,
                    "within_condition_distance_mean": float(
                        match["within_distances"].mean()
                    ),
                    "cross_condition_distance_mean": float(
                        match["cross_distances"].mean()
                    ),
                }
            )
        if len(repeats) < args.real_minimum_successful_repeats:
            real_exclusions.append(
                {
                    "first_map": first_map,
                    "second_map": second_map,
                    "successful_repeats": len(repeats),
                    "reason": "insufficient path-count-and-condition matched triplets",
                }
            )
        else:
            real_pair_reports.append(
                {
                    "first_map": first_map,
                    "second_map": second_map,
                    "successful_repeats": len(repeats),
                    "same_map_mean": {
                        metric: float(
                            torch.tensor(
                                [repeat["same_map"][metric] for repeat in repeats],
                                dtype=torch.float64,
                            ).mean()
                        )
                        for metric in METRICS
                    },
                    "cross_map_mean": {
                        metric: float(
                            torch.tensor(
                                [repeat["cross_map"][metric] for repeat in repeats],
                                dtype=torch.float64,
                            ).mean()
                        )
                        for metric in METRICS
                    },
                    "repeats": repeats,
                }
            )
        print(
            json.dumps(
                {
                    "event": "real_map_pair_complete",
                    "pair": pair_number,
                    "pairs_total": len(selected_pairs),
                    "first_map": first_map,
                    "second_map": second_map,
                    "successful_repeats": len(repeats),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not real_pair_reports:
        raise RuntimeError("no real Map pair met the matching and repeat requirements")
    real_summary = _map_cluster_real_summary(
        real_pair_reports,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed + 300000,
        equivalence_margin=args.equivalence_margin,
    )

    generated = torch.load(args.generated, map_location="cpu", weights_only=False)
    _validate_generated(generated)
    reference = generated["reference"]
    candidates = {
        label: generated[artifact_key] for label, artifact_key in MODEL_KEYS.items()
    }
    generated_sample_count = int(reference["map_id"].numel())
    for label, payload in candidates.items():
        if payload["path_mask"].shape[0] != generated_sample_count:
            raise ValueError(f"{label} sample count does not match the reference")
    map_ids = reference["map_id"].long()
    conditions = reference["conditions"].float()
    validity = reference["condition_validity"].bool()
    generated_maps = sorted(int(value) for value in torch.unique(map_ids))
    condition_input_audit = _condition_input_audit(reference, candidates)
    per_map: dict[str, Any] = {}
    for map_number, map_id in enumerate(generated_maps, start=1):
        indices = torch.nonzero(map_ids == map_id, as_tuple=False).flatten()
        if indices.numel() < args.minimum_per_map:
            per_map[f"map_{map_id}"] = {
                "map_id": map_id,
                "samples": int(indices.numel()),
                "status": "insufficient_map_samples",
            }
            continue
        groups = _local_groups(
            indices,
            conditions,
            validity,
            minimum_samples=args.minimum_local_group,
            maximum_groups_per_stratum=args.maximum_local_groups_per_stratum,
        )
        reference_map = _subset(reference, indices, device)
        model_payloads = {
            label: _subset(payload, indices, device)
            for label, payload in candidates.items()
        }
        reference_count = reference_map["path_mask"].sum(-1)
        common_count = torch.ones_like(reference_count, dtype=torch.bool)
        for payload in model_payloads.values():
            common_count &= payload["path_mask"].sum(-1) == reference_count
        common_indices = torch.nonzero(common_count, as_tuple=False).flatten()
        model_reports: dict[str, Any] = {}
        for label, payload in model_payloads.items():
            unconditional = _delay_power_metrics(
                payload,
                reference_map,
                normalizer,
                projections=args.projections,
                projection_seeds=projection_seeds,
                quantile_count=args.weighted_quantiles,
            )
            local_values: list[tuple[int, dict[str, float]]] = []
            local_reports: list[dict[str, Any]] = []
            for group_number, group in enumerate(groups):
                local_positions = torch.searchsorted(indices, group["indices"])
                reference_group = _subset(reference_map, local_positions, device)
                candidate_group = _subset(payload, local_positions, device)
                group_metrics = _delay_power_metrics(
                    candidate_group,
                    reference_group,
                    normalizer,
                    projections=args.projections,
                    projection_seeds=tuple(
                        seed
                        + args.seed
                        + map_number * 10000
                        + group_number * 100
                        for seed in projection_seeds
                    ),
                    quantile_count=args.weighted_quantiles,
                )
                local_values.append((group["samples"], group_metrics))
                local_reports.append(
                    {
                        **{key: value for key, value in group.items() if key != "indices"},
                        "metrics": group_metrics,
                    }
                )
            controlled: dict[str, float] | None = None
            if common_indices.numel() >= args.minimum_common_path_count:
                controlled = _delay_power_metrics(
                    _subset(payload, common_indices, device),
                    _subset(reference_map, common_indices, device),
                    normalizer,
                    projections=args.projections,
                    projection_seeds=tuple(
                        seed + args.seed + map_number * 10000
                        for seed in projection_seeds
                    ),
                    quantile_count=args.weighted_quantiles,
                )
            model_reports[label] = {
                "unconditional": unconditional,
                "local_condition_weighted": _weighted_metric_average(local_values),
                "local_condition_groups": local_reports,
                "common_path_count_controlled": controlled,
                "path_count": _path_count_audit(payload, reference_map),
            }
        eligible_local = sum(group["samples"] for group in groups)
        per_map[f"map_{map_id}"] = {
            "map_id": map_id,
            "samples": int(indices.numel()),
            "status": "ok",
            "support": {
                "local_condition_groups": len(groups),
                "eligible_local_samples": eligible_local,
                "local_condition_coverage": eligible_local / int(indices.numel()),
                "common_path_count_samples": int(common_indices.numel()),
                "common_path_count_rate": float(common_count.float().mean()),
                "los_samples": int((reference_map["conditions"][:, 3] >= 0.5).sum()),
                "k_valid_samples": int(reference_map["condition_validity"][:, 2].sum()),
            },
            "models": model_reports,
        }
        print(
            json.dumps(
                {
                    "event": "fidelity_map_complete",
                    "map": map_id,
                    "map_index": map_number,
                    "maps_total": len(generated_maps),
                    "samples": int(indices.numel()),
                    "local_groups": len(groups),
                    "common_path_count_samples": int(common_indices.numel()),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    fidelity_summaries = {
        "b3_scene_id_vs_b3": _paired_fidelity_summary(
            per_map,
            comparison="b3",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 400000,
        ),
        "b3_scene_id_vs_shuffled_map": _paired_fidelity_summary(
            per_map,
            comparison="shuffled_map",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 500000,
        ),
    }
    evaluated_maps = [
        values for values in per_map.values() if values.get("status") == "ok"
    ]
    path_count_summary = {
        label: {
            "mean_exact_match_rate": (
                sum(
                    values["models"][label]["path_count"]["exact_match_rate"]
                    for values in evaluated_maps
                )
                / max(len(evaluated_maps), 1)
            ),
            "mean_absolute_count_error": (
                sum(
                    values["models"][label]["path_count"]["mean_absolute_count_error"]
                    for values in evaluated_maps
                )
                / max(len(evaluated_maps), 1)
            ),
            "mean_distribution_total_variation": (
                sum(
                    values["models"][label]["path_count"]["distribution_total_variation"]
                    for values in evaluated_maps
                )
                / max(len(evaluated_maps), 1)
            ),
        }
        for label in MODEL_LABELS
    }
    requested_pair_degree = {map_id: 0 for map_id in real_maps}
    for first_map, second_map in selected_pairs:
        requested_pair_degree[first_map] += 1
        requested_pair_degree[second_map] += 1
    reported_pair_degree = {map_id: 0 for map_id in real_maps}
    for pair_report in real_pair_reports:
        reported_pair_degree[pair_report["first_map"]] += 1
        reported_pair_degree[pair_report["second_map"]] += 1
    matching_repeats = [
        repeat
        for pair_report in real_pair_reports
        for repeat in pair_report["repeats"]
    ]
    within_condition_mean = float(
        torch.tensor(
            [repeat["within_condition_distance_mean"] for repeat in matching_repeats],
            dtype=torch.float64,
        ).mean()
    )
    cross_condition_mean = float(
        torch.tensor(
            [repeat["cross_condition_distance_mean"] for repeat in matching_repeats],
            dtype=torch.float64,
        ).mean()
    )
    report = {
        "format_version": 1,
        "diagnostic": "per_map_conditional_delay_power",
        "generated_source": str(Path(args.generated).resolve()),
        "scene_plan_fingerprint": plan["fingerprint"],
        "device": str(device),
        "normalization": normalizer_cpu.audit(),
        "real_to_real": {
            "split": args.real_split,
            "samples": len(real_dataset),
            "maps": len(real_maps),
            "requested_balanced_map_pairs": len(selected_pairs),
            "reported_map_pairs": len(real_pair_reports),
            "excluded_map_pairs": real_exclusions,
            "requested_pair_degree_minimum": min(requested_pair_degree.values()),
            "requested_pair_degree_maximum": max(requested_pair_degree.values()),
            "reported_pair_degree_minimum": min(reported_pair_degree.values()),
            "reported_pair_degree_maximum": max(reported_pair_degree.values()),
            "samples_per_comparison": args.real_samples_per_comparison,
            "repeats": args.real_repeats,
            "matching": {
                "los_exact": True,
                "k_validity_exact": True,
                "path_count_exact": True,
                "maximum_standardized_condition_distance": args.maximum_condition_distance,
                "same_anchor_samples_for_same_and_cross": True,
                "one_to_one_without_replacement": True,
                "within_map_condition_distance_mean": within_condition_mean,
                "cross_map_condition_distance_mean": cross_condition_mean,
                "cross_minus_within_condition_distance": (
                    cross_condition_mean - within_condition_mean
                ),
                "warning": (
                    "A positive cross-minus-within condition distance can inflate the "
                    "observed scene contrast and must be considered before interpretation."
                ),
            },
            "equivalence_margin": args.equivalence_margin,
            "summary": real_summary,
            "map_pair_reports": real_pair_reports,
        },
        "model_fidelity": {
            "samples": int(map_ids.numel()),
            "maps_total": len(generated_maps),
            "maps_evaluated": len(evaluated_maps),
            "condition_input_audit": condition_input_audit,
            "support_summary": {
                "maps_without_local_groups": sum(
                    values["support"]["local_condition_groups"] == 0
                    for values in evaluated_maps
                ),
                "maps_without_controlled_metrics": sum(
                    values["models"]["b3_scene_id"]["common_path_count_controlled"]
                    is None
                    for values in evaluated_maps
                ),
                "mean_local_condition_coverage": (
                    sum(
                        values["support"]["local_condition_coverage"]
                        for values in evaluated_maps
                    )
                    / max(len(evaluated_maps), 1)
                ),
                "mean_common_path_count_rate": (
                    sum(
                        values["support"]["common_path_count_rate"]
                        for values in evaluated_maps
                    )
                    / max(len(evaluated_maps), 1)
                ),
            },
            "path_count_summary": path_count_summary,
            "summaries": fidelity_summaries,
            "per_map": per_map,
        },
        "interpretation": {
            "real_equivalence": (
                "Do not infer weak Map dependence from a nonsignificant difference alone; "
                "the relative-effect CI must lie inside the equivalence margin."
            ),
            "fidelity_sign": (
                "Positive comparison-minus-B3-S-ID improvement means B3-S-ID is closer "
                "to held-out real Delay-Power data."
            ),
            "decision": (
                "A generic B3 Delay-Power factor is supported only if real Maps are "
                "practically equivalent and B3 is not worse than B3-S-ID."
            ),
        },
        "cautions": [
            "Map ID is the cluster/bootstrap unit; path observations are not independent inference units.",
            "The fixed normalizer is fitted once on the training split and reused everywhere.",
            "This generated artifact represents one generation seed; multi-seed stability remains a separate experiment.",
        ],
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "delay_power_diagnostic_complete",
                "output": str(Path(args.output).resolve()),
                "real_map_pairs": len(real_pair_reports),
                "fidelity_maps": len(evaluated_maps),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
