#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.evaluation.distribution import (
    full_multipath_features,
    rbf_mmd,
    sliced_wasserstein,
)
from physics_csi.evaluation.research_diagnostics import (
    direction_vectors,
    normalized_path_power,
    power_aod_joint_features,
)
from physics_csi.evaluation.scene_diagnostics import (
    paired_map_summary,
    spatial_correlation_matrix,
)
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


MODEL_KEYS = {
    "real_vs_real": "reference",
    "b3_vs_real": "physics_only_baseline",
    "current_vs_real": "correct_map",
}
LOWER_IS_BETTER = {
    "path_count_total_variation",
    "pdp_mmd_rbf",
    "pdp_sliced_wasserstein",
    "raw_full_csi_projected_mmd_rbf",
    "raw_full_csi_projected_sliced_wasserstein",
    "phase_invariant_full_csi_projected_mmd_rbf",
    "phase_invariant_full_csi_projected_sliced_wasserstein",
    "path_level_delay_power_sliced_wasserstein",
    "link_level_delay_power_sliced_wasserstein",
    "path_level_power_aod_sliced_wasserstein",
    "link_level_power_aod_sliced_wasserstein",
    "power_weighted_aod_sliced_wasserstein",
    "tx_spatial_correlation_relative_frobenius",
    "tx_spatial_correlation_magnitude_mae",
    "tx_spatial_correlation_phase_mae_rad",
    "frequency_correlation_relative_l2",
    "frequency_correlation_magnitude_mae",
    "frequency_correlation_phase_mae_rad",
    "phase_invariant_csi_pairwise_log_ratio_abs",
    "multipath_pairwise_log_ratio_abs",
}
HIGHER_IS_BETTER = {
    "phase_invariant_csi_precision",
    "phase_invariant_csi_recall",
    "phase_invariant_csi_density",
    "phase_invariant_csi_coverage",
    "phase_invariant_csi_rare_reference_coverage",
    "multipath_precision",
    "multipath_recall",
    "multipath_density",
    "multipath_coverage",
    "multipath_rare_reference_coverage",
}
METRIC_NAMES = tuple(sorted(LOWER_IS_BETTER | HIGHER_IS_BETTER))
PHASE_PROBE_COUNT = 16
PHASE_PROBE_SEED = 314159
FREQUENCY_LAGS = (1, 2, 4, 8, 16, 32)


def _validate_artifact(source: Mapping[str, Any]) -> None:
    if source.get("model_kind") != "b3_plus_dp_plus_aod_residual":
        raise ValueError("artifact is not a B3+DP+AoD residual evaluation")
    if source.get("aod_architecture") != "link_global":
        raise ValueError("artifact is not the frozen Link-global AoD model evaluation")
    for key in ("reference", "physics_only_baseline", "correct_map"):
        if key not in source:
            raise ValueError(f"artifact is missing {key}")
        payload = source[key]
        for required in (
            "csi",
            "path_mask",
            "path_delays_ns",
            "path_powers_linear",
            "aod_az_deg",
            "aod_el_deg",
        ):
            if required not in payload:
                raise ValueError(f"artifact is missing {key}.{required}")
    reference = source["reference"]
    for required in ("map_id", "conditions", "condition_validity"):
        if required not in reference:
            raise ValueError(f"artifact is missing reference.{required}")


def _sample_tensors(payload: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    sample_count = int(payload["csi"].shape[0])
    return {
        key: value.to(device)
        for key, value in payload.items()
        if isinstance(value, torch.Tensor) and value.shape[:1] == (sample_count,)
    }


def _select(payload: Mapping[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value[indices] for key, value in payload.items()}


def _condition_features(
    conditions: torch.Tensor,
    validity: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    features = (conditions[:, :3].float() - center) / scale
    features[:, 2] = torch.where(
        validity[:, 2].bool(), features[:, 2], torch.zeros_like(features[:, 2])
    )
    return features


def _greedy_match(
    first: torch.Tensor,
    second: torch.Tensor,
    features: torch.Tensor,
    *,
    maximum_distance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not first.numel() or not second.numel():
        empty = torch.empty(0, dtype=torch.long, device=features.device)
        return empty, empty, torch.empty(0, device=features.device)
    distances = torch.cdist(features[first], features[second]).cpu()
    order = torch.argsort(distances.amin(1))
    used = torch.zeros(second.numel(), dtype=torch.bool)
    matched_first: list[int] = []
    matched_second: list[int] = []
    matched_distance: list[float] = []
    for row in order.tolist():
        for column in torch.argsort(distances[row]).tolist():
            distance = float(distances[row, column])
            if distance > maximum_distance:
                break
            if not bool(used[column]):
                used[column] = True
                matched_first.append(int(first[row]))
                matched_second.append(int(second[column]))
                matched_distance.append(distance)
                break
    return (
        torch.tensor(matched_first, dtype=torch.long, device=features.device),
        torch.tensor(matched_second, dtype=torch.long, device=features.device),
        torch.tensor(matched_distance, dtype=torch.float32, device=features.device),
    )


def _make_map_pairs(
    reference: Mapping[str, torch.Tensor],
    *,
    seed: int,
    maximum_distance: float,
    minimum_pairs: int,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, Any]]:
    conditions = reference["conditions"].float()
    validity = reference["condition_validity"].bool()
    center = conditions[:, :3].mean(0)
    scale = conditions[:, :3].std(0, unbiased=False).clamp_min(1e-6)
    valid_k = validity[:, 2]
    if bool(valid_k.any()):
        center[2] = conditions[valid_k, 2].mean()
        scale[2] = conditions[valid_k, 2].std(unbiased=False).clamp_min(1e-6)
    features = _condition_features(conditions, validity, center, scale)
    map_ids = reference["map_id"].long()
    los = conditions[:, 3] >= 0.5
    generator = torch.Generator().manual_seed(seed)
    pairs: dict[int, dict[str, torch.Tensor]] = {}
    excluded: dict[int, int] = {}
    all_distances: list[torch.Tensor] = []
    for map_id_tensor in torch.unique(map_ids, sorted=True):
        map_id = int(map_id_tensor)
        map_indices = torch.nonzero(map_ids == map_id, as_tuple=False).flatten()
        first_chunks: list[torch.Tensor] = []
        second_chunks: list[torch.Tensor] = []
        distance_chunks: list[torch.Tensor] = []
        strata = sorted(
            {
                (bool(los[index]), bool(validity[index, 2]))
                for index in map_indices.tolist()
            }
        )
        for current_los, current_k_valid in strata:
            stratum = map_indices[
                (los[map_indices] == current_los)
                & (validity[map_indices, 2] == current_k_valid)
            ]
            if stratum.numel() < 2:
                continue
            permutation = stratum[
                torch.randperm(stratum.numel(), generator=generator).to(stratum.device)
            ]
            midpoint = permutation.numel() // 2
            first, second, distances = _greedy_match(
                permutation[:midpoint],
                permutation[midpoint : midpoint * 2],
                features,
                maximum_distance=maximum_distance,
            )
            first_chunks.append(first)
            second_chunks.append(second)
            distance_chunks.append(distances)
        if not first_chunks:
            excluded[map_id] = 0
            continue
        first = torch.cat(first_chunks)
        second = torch.cat(second_chunks)
        distances = torch.cat(distance_chunks)
        if first.numel() < minimum_pairs:
            excluded[map_id] = int(first.numel())
            continue
        order = torch.argsort(first)
        pairs[map_id] = {
            "candidate_indices": first[order],
            "reference_indices": second[order],
            "condition_distance": distances[order],
        }
        all_distances.append(distances)
    if not pairs:
        raise RuntimeError("no Map has enough independently matched real samples")
    distances = torch.cat(all_distances)
    return pairs, {
        "seed": seed,
        "maps_retained": len(pairs),
        "maps_excluded_with_available_pair_count": excluded,
        "pairs": int(sum(item["candidate_indices"].numel() for item in pairs.values())),
        "condition_distance": {
            "mean": float(distances.mean()),
            "p50": float(torch.quantile(distances, 0.50)),
            "p90": float(torch.quantile(distances, 0.90)),
            "p95": float(torch.quantile(distances, 0.95)),
            "maximum": float(distances.max()),
        },
        "matching_constraints": "same Map, exact LoS, exact K-validity, nearest continuous physical conditions",
    }


def _local_groups(
    pairs: Mapping[int, Mapping[str, torch.Tensor]],
    reference: Mapping[str, torch.Tensor],
    *,
    minimum_samples: int,
    maximum_groups_per_stratum: int,
) -> list[dict[str, Any]]:
    conditions = reference["conditions"].float()
    validity = reference["condition_validity"].bool()
    center = conditions[:, :3].mean(0)
    scale = conditions[:, :3].std(0, unbiased=False).clamp_min(1e-6)
    features = _condition_features(conditions, validity, center, scale)
    los = conditions[:, 3] >= 0.5
    groups: list[dict[str, Any]] = []
    for map_id, pair in pairs.items():
        candidate = pair["candidate_indices"]
        target = pair["reference_indices"]
        pair_features = 0.5 * (features[candidate] + features[target])
        strata = sorted(
            {
                (bool(los[index]), bool(validity[index, 2]))
                for index in candidate.tolist()
            }
        )
        for current_los, current_k_valid in strata:
            local = torch.nonzero(
                (los[candidate] == current_los)
                & (validity[candidate, 2] == current_k_valid),
                as_tuple=False,
            ).flatten()
            if local.numel() < minimum_samples:
                continue
            count = min(
                maximum_groups_per_stratum,
                max(1, int(local.numel()) // minimum_samples),
            )
            centered = pair_features[local] - pair_features[local].mean(0, keepdim=True)
            if count > 1 and centered.shape[0] > 1:
                _, _, right = torch.linalg.svd(centered, full_matrices=False)
                order = local[torch.argsort(centered @ right[0])]
            else:
                order = local
            for group_number, chunk in enumerate(torch.tensor_split(order, count)):
                if chunk.numel() < minimum_samples:
                    continue
                groups.append(
                    {
                        "map_id": map_id,
                        "group": f"map_{map_id}_los_{int(current_los)}_kvalid_{int(current_k_valid)}_{group_number}",
                        "candidate_indices": candidate[chunk],
                        "reference_indices": target[chunk],
                        "samples": int(chunk.numel()),
                        "los": current_los,
                        "k_valid": current_k_valid,
                    }
                )
    return groups


def _phase_normalize(csi: torch.Tensor) -> torch.Tensor:
    flat = csi.flatten(1)
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
    probes = probes / probes.abs().square().sum(-1, keepdim=True).sqrt().clamp_min(1e-20)
    anchors = flat @ probes.conj().transpose(0, 1)
    selected = anchors.gather(1, anchors.abs().argmax(-1, keepdim=True))
    unit = selected / selected.abs().clamp_min(1e-20)
    return csi * unit.conj().reshape((csi.shape[0],) + (1,) * (csi.ndim - 1))


def _energy_normalized_features(csi: torch.Tensor) -> torch.Tensor:
    flat = csi.flatten(1)
    flat = flat / flat.abs().square().sum(-1, keepdim=True).sqrt().clamp_min(1e-20)
    return torch.cat([flat.real, flat.imag], dim=-1).float()


def _normalized_pdp(csi: torch.Tensor) -> torch.Tensor:
    values = torch.fft.ifft(csi, dim=-1, norm="ortho").abs().square().mean((1, 2))
    return values / values.sum(-1, keepdim=True).clamp_min(1e-20)


def _moments(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    mean = values.mean(0, keepdim=True)
    raw_scale = values.std(0, unbiased=False, keepdim=True)
    positive = raw_scale[torch.isfinite(raw_scale) & (raw_scale > 0)]
    typical = positive.median() if positive.numel() else torch.ones((), device=values.device)
    floor = torch.maximum(typical * 0.05, torch.tensor(1e-6, device=values.device))
    scale = torch.where(torch.isfinite(raw_scale), raw_scale, torch.zeros_like(raw_scale)).clamp_min(floor)
    return mean, scale, {
        "dimensions": int(raw_scale.numel()),
        "scale_floor": float(floor),
        "dimensions_using_floor": int((raw_scale <= floor).sum()),
    }


class FixedProjection:
    def __init__(self, reference: torch.Tensor, dimension: int, seeds: tuple[int, ...]) -> None:
        self.mean, self.scale, self.moment_audit = _moments(reference)
        self.dimension = dimension
        self.seeds = seeds
        self.directions: dict[int, torch.Tensor] = {}
        for seed in seeds:
            generator = torch.Generator(device=reference.device).manual_seed(seed)
            signs = torch.randint(
                0,
                2,
                (reference.shape[1], dimension),
                generator=generator,
                device=reference.device,
                dtype=torch.int8,
            ).float()
            self.directions[seed] = (2.0 * signs - 1.0) / math.sqrt(reference.shape[1])

    def transform(self, values: torch.Tensor) -> dict[int, torch.Tensor]:
        standardized = (values - self.mean) / self.scale
        return {seed: standardized @ direction for seed, direction in self.directions.items()}

    def audit(self) -> dict[str, Any]:
        return self.moment_audit | {
            "input_dimensions": int(self.mean.shape[1]),
            "projection_dimensions": self.dimension,
            "seeds": list(self.seeds),
        }


def _relative_delay_power(payload: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = payload["path_mask"].bool()
    delay = payload["path_delays_ns"].float()
    first = torch.where(mask, delay, torch.full_like(delay, torch.inf)).amin(-1)
    relative = (delay - first.unsqueeze(-1)).clamp_min(0.0)
    power = payload["path_powers_linear"].float() * mask
    fraction = power / power.sum(-1, keepdim=True).clamp_min(1e-20)
    return mask, relative, fraction


def _delay_power_path(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask, delay, fraction = _relative_delay_power(payload)
    return torch.stack([delay, 10.0 * torch.log10(fraction.clamp_min(1e-20))], dim=-1)[mask]


def _delay_power_link(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask, delay, fraction = _relative_delay_power(payload)
    order = torch.argsort(
        torch.where(mask, payload["path_delays_ns"].float(), torch.full_like(delay, torch.inf)),
        dim=-1,
        stable=True,
    )
    mask = mask.gather(1, order)
    return torch.cat(
        [mask.float(), delay.gather(1, order) * mask, fraction.gather(1, order) * mask],
        dim=-1,
    )


def _power_aod_path(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask = payload["path_mask"].bool()
    power_db = 10.0 * torch.log10(normalized_path_power(payload).clamp_min(1e-20))
    direction = direction_vectors(payload["aod_az_deg"], payload["aod_el_deg"])
    return torch.cat([direction, power_db.unsqueeze(-1)], dim=-1)[mask]


def _weighted_quantiles(
    projected: torch.Tensor,
    weights: torch.Tensor,
    quantiles: torch.Tensor,
) -> torch.Tensor:
    values, order = projected.transpose(0, 1).sort(dim=-1)
    sorted_weights = weights[order]
    cumulative = sorted_weights.cumsum(-1)
    cumulative = cumulative / cumulative[:, -1:].clamp_min(1e-20)
    query = quantiles.unsqueeze(0).expand(values.shape[0], -1).contiguous()
    locations = torch.searchsorted(cumulative.contiguous(), query, right=False)
    return values.gather(1, locations.clamp_max(values.shape[1] - 1))


def _power_weighted_aod_swd(
    candidate: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    *,
    projections: int,
    quantiles: int,
    seeds: tuple[int, ...],
) -> float:
    def points(payload: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        mask = payload["path_mask"].bool()
        direction = direction_vectors(payload["aod_az_deg"], payload["aod_el_deg"])
        weights = normalized_path_power(payload) / max(mask.shape[0], 1)
        return direction[mask], weights[mask]

    candidate_points, candidate_weights = points(candidate)
    reference_points, reference_weights = points(reference)
    query = (torch.arange(quantiles, device=reference_points.device).float() + 0.5) / quantiles
    runs: list[float] = []
    for seed in seeds:
        generator = torch.Generator(device=reference_points.device).manual_seed(seed)
        directions = torch.randn(
            projections,
            3,
            generator=generator,
            device=reference_points.device,
        )
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-20)
        first = _weighted_quantiles(candidate_points @ directions.T, candidate_weights, query)
        second = _weighted_quantiles(reference_points @ directions.T, reference_weights, query)
        runs.append(float((first - second).abs().mean()))
    return sum(runs) / len(runs)


def _multi_seed_swd(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    projections: int,
    seeds: tuple[int, ...],
) -> float:
    return sum(
        float(sliced_wasserstein(candidate, reference, projections=projections, seed=seed))
        for seed in seeds
    ) / len(seeds)


def _projection_metrics(
    candidate: Mapping[int, torch.Tensor],
    reference: Mapping[int, torch.Tensor],
    candidate_indices: torch.Tensor,
    reference_indices: torch.Tensor,
    *,
    projections: int,
) -> tuple[float, float]:
    mmd: list[float] = []
    swd: list[float] = []
    for seed in sorted(candidate):
        first = candidate[seed][candidate_indices]
        second = reference[seed][reference_indices]
        mmd.append(float(rbf_mmd(first, second, [2.0, 4.0, 8.0, 16.0])))
        swd.append(float(sliced_wasserstein(first, second, projections=projections, seed=seed + 1000)))
    return sum(mmd) / len(mmd), sum(swd) / len(swd)


def _path_count_tv(candidate: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]) -> float:
    first = candidate["path_mask"].sum(-1).long()
    second = reference["path_mask"].sum(-1).long()
    bins = max(candidate["path_mask"].shape[1], reference["path_mask"].shape[1]) + 1
    first_hist = torch.bincount(first, minlength=bins).float() / max(first.numel(), 1)
    second_hist = torch.bincount(second, minlength=bins).float() / max(second.numel(), 1)
    return float(0.5 * (first_hist - second_hist).abs().sum())


def _spatial_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    first = spatial_correlation_matrix(candidate)
    second = spatial_correlation_matrix(reference)
    difference = first - second
    off_diagonal = ~torch.eye(second.shape[0], dtype=torch.bool, device=second.device)
    valid = off_diagonal & (second.abs() > 1e-3)
    phase = (
        torch.angle(first[valid] * second[valid].conj()).abs().mean()
        if bool(valid.any())
        else torch.zeros((), device=second.device)
    )
    return {
        "tx_spatial_correlation_relative_frobenius": float(
            torch.linalg.matrix_norm(difference) / torch.linalg.matrix_norm(second).clamp_min(1e-20)
        ),
        "tx_spatial_correlation_magnitude_mae": float((first.abs() - second.abs()).abs().mean()),
        "tx_spatial_correlation_phase_mae_rad": float(phase),
    }


def _frequency_correlation(csi: torch.Tensor) -> torch.Tensor:
    samples = csi.flatten(0, 2)
    energy = samples.abs().square().mean().clamp_min(1e-20)
    return torch.stack(
        [
            (samples[:, :-lag].conj() * samples[:, lag:]).mean() / energy
            for lag in FREQUENCY_LAGS
            if lag < csi.shape[-1]
        ]
    )


def _frequency_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    first = _frequency_correlation(candidate)
    second = _frequency_correlation(reference)
    valid = second.abs() > 1e-3
    phase = (
        torch.angle(first[valid] * second[valid].conj()).abs().mean()
        if bool(valid.any())
        else torch.zeros((), device=second.device)
    )
    return {
        "frequency_correlation_relative_l2": float(
            torch.linalg.vector_norm(first - second) / torch.linalg.vector_norm(second).clamp_min(1e-20)
        ),
        "frequency_correlation_magnitude_mae": float((first.abs() - second.abs()).abs().mean()),
        "frequency_correlation_phase_mae_rad": float(phase),
    }


def _manifold_metrics(candidate: torch.Tensor, reference: torch.Tensor, requested_k: int) -> dict[str, float]:
    k = min(requested_k, candidate.shape[0] - 1, reference.shape[0] - 1)
    if k < 1:
        raise ValueError("coverage metrics require at least two samples per set")
    reference_self = torch.cdist(reference, reference)
    candidate_self = torch.cdist(candidate, candidate)
    reference_self.fill_diagonal_(torch.inf)
    candidate_self.fill_diagonal_(torch.inf)
    reference_radius = reference_self.kthvalue(k, dim=1).values
    candidate_radius = candidate_self.kthvalue(k, dim=1).values
    cross = torch.cdist(candidate, reference)
    precision = (cross <= reference_radius.unsqueeze(0)).any(1).float().mean()
    recall = (cross <= candidate_radius.unsqueeze(1)).any(0).float().mean()
    density = (cross <= reference_radius.unsqueeze(0)).float().sum(1).mean() / k
    covered = cross.amin(0) <= reference_radius
    rare = reference_radius >= torch.quantile(reference_radius, 0.90)
    rare_coverage = covered[rare].float().mean() if bool(rare.any()) else covered.float().mean()
    candidate_pairwise = torch.pdist(candidate).mean().clamp_min(1e-20)
    reference_pairwise = torch.pdist(reference).mean().clamp_min(1e-20)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "density": float(density),
        "coverage": float(covered.float().mean()),
        "rare_reference_coverage": float(rare_coverage),
        "pairwise_log_ratio_abs": float(torch.log(candidate_pairwise / reference_pairwise).abs()),
    }


class Evaluator:
    def __init__(
        self,
        payloads: Mapping[str, Mapping[str, torch.Tensor]],
        *,
        projection_dimension: int,
        projection_seeds: tuple[int, ...],
        projections: int,
        quantiles: int,
        coverage_k: int,
    ) -> None:
        self.payloads = payloads
        self.seeds = projection_seeds
        self.projections = projections
        self.quantiles = quantiles
        self.coverage_k = coverage_k
        reference = payloads["reference"]

        reference_raw = _energy_normalized_features(reference["csi"])
        raw_basis = FixedProjection(reference_raw, projection_dimension, projection_seeds)
        self.raw = {
            label: raw_basis.transform(_energy_normalized_features(payload["csi"]))
            for label, payload in payloads.items()
        }
        del reference_raw
        reference_invariant = _energy_normalized_features(_phase_normalize(reference["csi"]))
        invariant_basis = FixedProjection(reference_invariant, projection_dimension, projection_seeds)
        self.invariant = {
            label: invariant_basis.transform(_energy_normalized_features(_phase_normalize(payload["csi"])))
            for label, payload in payloads.items()
        }
        del reference_invariant

        reference_pdp = _normalized_pdp(reference["csi"])
        pdp_mean, pdp_scale, pdp_audit = _moments(reference_pdp)
        self.pdp = {
            label: (_normalized_pdp(payload["csi"]) - pdp_mean) / pdp_scale
            for label, payload in payloads.items()
        }

        self.dp_path_mean, self.dp_path_scale, dp_path_audit = _moments(_delay_power_path(reference))
        self.dp_link_mean, self.dp_link_scale, dp_link_audit = _moments(_delay_power_link(reference))
        self.pa_path_mean, self.pa_path_scale, pa_path_audit = _moments(_power_aod_path(reference))
        self.pa_link_mean, self.pa_link_scale, pa_link_audit = _moments(power_aod_joint_features(reference))
        multipath_reference = full_multipath_features(reference)
        multipath_mean, multipath_scale, multipath_audit = _moments(multipath_reference)
        self.multipath = {
            label: (full_multipath_features(payload) - multipath_mean) / multipath_scale
            for label, payload in payloads.items()
        }
        self.audit = {
            "raw_full_csi": raw_basis.audit(),
            "phase_invariant_full_csi": invariant_basis.audit(),
            "pdp": pdp_audit,
            "delay_power_path": dp_path_audit,
            "delay_power_link": dp_link_audit,
            "power_aod_path": pa_path_audit,
            "power_aod_link": pa_link_audit,
            "full_multipath": multipath_audit,
            "normalization_source": "the complete held-out real reference payload; fitted once and reused",
        }

    def evaluate(
        self,
        candidate_label: str,
        candidate_indices: torch.Tensor,
        reference_indices: torch.Tensor,
        *,
        local: bool = False,
    ) -> dict[str, float]:
        reference_label = "reference"
        candidate = _select(self.payloads[candidate_label], candidate_indices)
        reference = _select(self.payloads[reference_label], reference_indices)
        projections = max(32, self.projections // 2) if local else self.projections
        seeds = self.seeds[:1] if local else self.seeds

        raw_mmd, raw_swd = _projection_metrics(
            self.raw[candidate_label], self.raw[reference_label], candidate_indices, reference_indices, projections=projections
        )
        invariant_mmd, invariant_swd = _projection_metrics(
            self.invariant[candidate_label], self.invariant[reference_label], candidate_indices, reference_indices, projections=projections
        )
        candidate_pdp = self.pdp[candidate_label][candidate_indices]
        reference_pdp = self.pdp[reference_label][reference_indices]
        candidate_dp_path = (_delay_power_path(candidate) - self.dp_path_mean) / self.dp_path_scale
        reference_dp_path = (_delay_power_path(reference) - self.dp_path_mean) / self.dp_path_scale
        candidate_dp_link = (_delay_power_link(candidate) - self.dp_link_mean) / self.dp_link_scale
        reference_dp_link = (_delay_power_link(reference) - self.dp_link_mean) / self.dp_link_scale
        candidate_pa_path = (_power_aod_path(candidate) - self.pa_path_mean) / self.pa_path_scale
        reference_pa_path = (_power_aod_path(reference) - self.pa_path_mean) / self.pa_path_scale
        candidate_pa_link = (power_aod_joint_features(candidate) - self.pa_link_mean) / self.pa_link_scale
        reference_pa_link = (power_aod_joint_features(reference) - self.pa_link_mean) / self.pa_link_scale

        invariant_seed = self.seeds[0]
        csi_coverage = _manifold_metrics(
            self.invariant[candidate_label][invariant_seed][candidate_indices],
            self.invariant[reference_label][invariant_seed][reference_indices],
            self.coverage_k,
        )
        multipath_coverage = _manifold_metrics(
            self.multipath[candidate_label][candidate_indices],
            self.multipath[reference_label][reference_indices],
            self.coverage_k,
        )
        result = {
            "path_count_total_variation": _path_count_tv(candidate, reference),
            "pdp_mmd_rbf": float(rbf_mmd(candidate_pdp, reference_pdp, [0.5, 1.0, 2.0, 4.0])),
            "pdp_sliced_wasserstein": _multi_seed_swd(candidate_pdp, reference_pdp, projections=projections, seeds=seeds),
            "raw_full_csi_projected_mmd_rbf": raw_mmd,
            "raw_full_csi_projected_sliced_wasserstein": raw_swd,
            "phase_invariant_full_csi_projected_mmd_rbf": invariant_mmd,
            "phase_invariant_full_csi_projected_sliced_wasserstein": invariant_swd,
            "path_level_delay_power_sliced_wasserstein": _multi_seed_swd(candidate_dp_path, reference_dp_path, projections=projections, seeds=seeds),
            "link_level_delay_power_sliced_wasserstein": _multi_seed_swd(candidate_dp_link, reference_dp_link, projections=projections, seeds=seeds),
            "path_level_power_aod_sliced_wasserstein": _multi_seed_swd(candidate_pa_path, reference_pa_path, projections=projections, seeds=seeds),
            "link_level_power_aod_sliced_wasserstein": _multi_seed_swd(candidate_pa_link, reference_pa_link, projections=projections, seeds=seeds),
            "power_weighted_aod_sliced_wasserstein": _power_weighted_aod_swd(candidate, reference, projections=projections, quantiles=self.quantiles, seeds=seeds),
            **_spatial_metrics(candidate["csi"], reference["csi"]),
            **_frequency_metrics(candidate["csi"], reference["csi"]),
            **{f"phase_invariant_csi_{key}": value for key, value in csi_coverage.items()},
            **{f"multipath_{key}": value for key, value in multipath_coverage.items()},
        }
        missing = set(METRIC_NAMES) - set(result)
        if missing:
            raise RuntimeError(f"metric implementation is incomplete: {sorted(missing)}")
        return result


def _floor_summary(
    model: list[float],
    floor: list[float],
    *,
    lower_is_better: bool,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    model_tensor = torch.tensor(model, dtype=torch.float64)
    floor_tensor = torch.tensor(floor, dtype=torch.float64)
    oriented_gap = model_tensor - floor_tensor if lower_is_better else floor_tensor - model_tensor
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        oriented_gap.numel(),
        (bootstrap_samples, oriented_gap.numel()),
        generator=generator,
    )
    bootstrap = oriented_gap[indices].mean(-1)
    interval = torch.quantile(bootstrap, torch.tensor([0.025, 0.975], dtype=torch.float64))
    lower, upper = float(interval[0]), float(interval[1])
    status = "detectably_above_real_real_floor" if lower > 0 else (
        "at_or_better_than_real_real_floor" if upper < 0 else "inconclusive_vs_real_real_floor"
    )
    floor_mean = float(floor_tensor.mean())
    return {
        "maps": int(model_tensor.numel()),
        "model_mean": float(model_tensor.mean()),
        "real_real_mean": floor_mean,
        "oriented_gap_model_worse_if_positive": float(oriented_gap.mean()),
        "bootstrap_95_percent_ci": [lower, upper],
        "model_to_real_real_ratio": (
            float(model_tensor.mean()) / floor_mean
            if lower_is_better and abs(floor_mean) > 1e-20
            else None
        ),
        "status": status,
    }


def _summary_sections(
    records: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if not records:
        return {}
    output: dict[str, Any] = {"b3_vs_real_real_floor": {}, "current_vs_real_real_floor": {}, "current_vs_b3": {}}
    for metric_number, metric in enumerate(METRIC_NAMES):
        lower = metric in LOWER_IS_BETTER
        real = [item["models"]["real_vs_real"][metric] for item in records]
        b3 = [item["models"]["b3_vs_real"][metric] for item in records]
        current = [item["models"]["current_vs_real"][metric] for item in records]
        output["b3_vs_real_real_floor"][metric] = _floor_summary(
            b3, real, lower_is_better=lower, bootstrap_samples=bootstrap_samples, seed=seed + metric_number
        )
        output["current_vs_real_real_floor"][metric] = _floor_summary(
            current, real, lower_is_better=lower, bootstrap_samples=bootstrap_samples, seed=seed + 1000 + metric_number
        )
        if lower:
            output["current_vs_b3"][metric] = paired_map_summary(
                current, b3, bootstrap_samples=bootstrap_samples, seed=seed + 2000 + metric_number
            )
        else:
            summary = paired_map_summary(
                [-value for value in current],
                [-value for value in b3],
                bootstrap_samples=bootstrap_samples,
                seed=seed + 2000 + metric_number,
            )
            b3_mean = sum(b3) / len(b3)
            current_mean = sum(current) / len(current)
            summary["correct_map_mean_distance"] = current_mean
            summary["comparison_mean_distance"] = b3_mean
            summary["mean_improvement_comparison_minus_correct"] = current_mean - b3_mean
            summary["relative_mean_improvement"] = (
                (current_mean - b3_mean) / abs(b3_mean)
                if abs(b3_mean) > 1e-20
                else None
            )
            summary["note"] = "Positive improvement means current has higher coverage/precision than B3."
            output["current_vs_b3"][metric] = summary
    return output


def _macro_local_records(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_map[int(group["map_id"])].append(group)
    records: list[dict[str, Any]] = []
    for map_id, values in sorted(by_map.items()):
        models: dict[str, dict[str, float]] = {}
        for label in MODEL_KEYS:
            models[label] = {
                metric: sum(item["models"][label][metric] for item in values) / len(values)
                for metric in METRIC_NAMES
            }
        records.append({"map_id": map_id, "groups": len(values), "models": models})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Absolute CSI quality evaluation against an independently matched Real-vs-Real floor."
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--minimum-pairs-per-map", type=int, default=12)
    parser.add_argument("--maximum-condition-distance", type=float, default=2.5)
    parser.add_argument("--local-minimum-samples", type=int, default=8)
    parser.add_argument("--local-maximum-groups-per-stratum", type=int, default=3)
    parser.add_argument("--projection-dimension", type=int, default=64)
    parser.add_argument("--projection-seeds", default="0,1,2")
    parser.add_argument("--swd-projections", type=int, default=128)
    parser.add_argument("--weighted-quantiles", type=int, default=256)
    parser.add_argument("--coverage-k", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.projection_seeds.split(",") if value)
    if not seeds:
        parser.error("--projection-seeds requires at least one integer")
    if args.minimum_pairs_per_map < 2 or args.local_minimum_samples < 2:
        parser.error("minimum sample counts must be at least two")

    device = resolve_device(args.device)
    source = torch.load(args.artifact, map_location="cpu", weights_only=False)
    _validate_artifact(source)
    payloads = {
        "reference": _sample_tensors(source["reference"], device),
        "b3": _sample_tensors(source["physics_only_baseline"], device),
        "current": _sample_tensors(source["correct_map"], device),
    }
    pairs, pairing_audit = _make_map_pairs(
        payloads["reference"],
        seed=args.seed,
        maximum_distance=args.maximum_condition_distance,
        minimum_pairs=args.minimum_pairs_per_map,
    )
    local_groups = _local_groups(
        pairs,
        payloads["reference"],
        minimum_samples=args.local_minimum_samples,
        maximum_groups_per_stratum=args.local_maximum_groups_per_stratum,
    )
    evaluator = Evaluator(
        payloads,
        projection_dimension=args.projection_dimension,
        projection_seeds=seeds,
        projections=args.swd_projections,
        quantiles=args.weighted_quantiles,
        coverage_k=args.coverage_k,
    )

    per_map: list[dict[str, Any]] = []
    for map_number, (map_id, pair) in enumerate(sorted(pairs.items()), start=1):
        models: dict[str, dict[str, float]] = {}
        for output_label, payload_label in MODEL_KEYS.items():
            candidate_indices = pair["candidate_indices"]
            models[output_label] = evaluator.evaluate(
                "reference" if payload_label == "reference" else ("b3" if payload_label == "physics_only_baseline" else "current"),
                candidate_indices,
                pair["reference_indices"],
            )
        per_map.append(
            {
                "map_id": map_id,
                "samples_per_side": int(pair["candidate_indices"].numel()),
                "condition_distance_mean": float(pair["condition_distance"].mean()),
                "models": models,
            }
        )
        print(json.dumps({"event": "map_complete", "map": map_id, "complete": map_number, "total": len(pairs)}), flush=True)

    local_results: list[dict[str, Any]] = []
    for group_number, group in enumerate(local_groups, start=1):
        models = {}
        for output_label, payload_label in MODEL_KEYS.items():
            models[output_label] = evaluator.evaluate(
                "reference" if payload_label == "reference" else ("b3" if payload_label == "physics_only_baseline" else "current"),
                group["candidate_indices"],
                group["reference_indices"],
                local=True,
            )
        local_results.append(
            {
                key: value
                for key, value in group.items()
                if key not in {"candidate_indices", "reference_indices"}
            }
            | {"models": models}
        )
        if group_number % 25 == 0 or group_number == len(local_groups):
            print(json.dumps({"event": "local_group_progress", "complete": group_number, "total": len(local_groups)}), flush=True)

    all_candidate = torch.cat([item["candidate_indices"] for item in pairs.values()])
    all_reference = torch.cat([item["reference_indices"] for item in pairs.values()])
    overall = {
        label: evaluator.evaluate(
            "reference" if key == "reference" else ("b3" if key == "physics_only_baseline" else "current"),
            all_candidate,
            all_reference,
        )
        for label, key in MODEL_KEYS.items()
    }
    local_macro = _macro_local_records(local_results)
    report = {
        "format_version": 1,
        "evaluation": "csi_absolute_generation_quality",
        "artifact": str(Path(args.artifact).resolve()),
        "model": "B3 + DP Residual + Link-global AoD Residual",
        "samples_per_side": int(all_candidate.numel()),
        "maps_evaluated": len(per_map),
        "metric_direction": {
            "lower_is_better": sorted(LOWER_IS_BETTER),
            "higher_is_better": sorted(HIGHER_IS_BETTER),
        },
        "configuration": {
            "seed": args.seed,
            "minimum_pairs_per_map": args.minimum_pairs_per_map,
            "maximum_condition_distance": args.maximum_condition_distance,
            "local_minimum_samples": args.local_minimum_samples,
            "projection_dimension": args.projection_dimension,
            "projection_seeds": list(seeds),
            "swd_projections": args.swd_projections,
            "weighted_quantiles": args.weighted_quantiles,
            "coverage_k": args.coverage_k,
            "frequency_lags": list(FREQUENCY_LAGS),
            "bootstrap_samples": args.bootstrap_samples,
        },
        "pairing_audit": pairing_audit,
        "normalization_audit": evaluator.audit,
        "overall": overall,
        "per_map_summary": _summary_sections(per_map, bootstrap_samples=args.bootstrap_samples, seed=args.seed),
        "local_condition_summary": _summary_sections(local_macro, bootstrap_samples=args.bootstrap_samples, seed=args.seed + 10000),
        "per_map": per_map,
        "local_condition_groups": local_results,
        "interpretation": {
            "distance_floor": "For lower-is-better metrics, a model/Real-to-Real ratio near one means the generated distance is near finite-sample real variation under this protocol.",
            "floor_status": "Detectably above floor means the Map-bootstrap confidence interval of the oriented gap is entirely positive; inconclusive does not prove equality.",
            "coverage": "Precision measures plausibility, recall/coverage measure mode support, and rare-reference coverage is restricted to the sparsest 10% of real feature-space samples.",
        },
        "cautions": [
            "Real-to-Real uses independent halves matched within Map, LoS and K-validity on nearby continuous conditions; it is a finite-sample floor, not zero error.",
            "The Map bootstrap measures spatial consistency, not independent model-training-seed stability.",
            "The artifact contains one frozen B3 generation cache and one residual training seed; repeat with independently generated artifacts before a final paper claim.",
            "Local-condition groups use fewer samples and half the SW projection budget, so they are diagnostic rather than the primary headline estimate.",
        ],
    }
    atomic_json_dump(report, args.output)
    print(json.dumps({"event": "csi_absolute_quality_complete", "output": str(Path(args.output).resolve()), "samples_per_side": int(all_candidate.numel()), "maps": len(per_map), "local_groups": len(local_results)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
