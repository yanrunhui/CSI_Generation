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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.config import load_yaml
from physics_csi.data.physical_labels import compute_physical_conditions
from physics_csi.data.scene_id import load_seen_map_spatial_plan, make_scene_id_datasets
from physics_csi.evaluation.diffusion_comparison import compare_csi_distributions
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
from physics_csi.evaluation.scene_diagnostics import paired_map_summary
from physics_csi.physics.constraints import PhysicsConstraintLayer, first_active_path_mask
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


PATH_KEYS = (
    "path_mask",
    "path_delays_ns",
    "path_powers_linear",
    "path_phases_rad",
    "aoa_az_deg",
    "aoa_el_deg",
    "aod_az_deg",
    "aod_el_deg",
)

REPAIR_GROUPS: dict[str, tuple[str, ...]] = {
    "aod": ("aod_az_deg", "aod_el_deg"),
    "power": ("path_powers_linear",),
    "delay": ("path_delays_ns",),
    "phase": ("path_phases_rad",),
    "power_aod_joint": ("path_powers_linear", "aod_az_deg", "aod_el_deg"),
    "delay_power_joint": ("path_delays_ns", "path_powers_linear"),
    "delay_power_aod_joint": (
        "path_delays_ns",
        "path_powers_linear",
        "aod_az_deg",
        "aod_el_deg",
    ),
    "all_except_phase": (
        "path_delays_ns",
        "path_powers_linear",
        "aoa_az_deg",
        "aoa_el_deg",
        "aod_az_deg",
        "aod_el_deg",
    ),
    "all_path_parameters": (
        "path_delays_ns",
        "path_powers_linear",
        "path_phases_rad",
        "aoa_az_deg",
        "aoa_el_deg",
        "aod_az_deg",
        "aod_el_deg",
    ),
}

CORE_DISTANCE_PATHS = (
    "path_distribution.aod_direction_sliced_wasserstein",
    "path_distribution.power_aod_joint_sliced_wasserstein",
    "path_distribution.power_aod_link_path_count_conditioned_sliced_wasserstein",
    "path_distribution.delay_power_joint_sliced_wasserstein",
    "path_distribution.delay_power_link_path_count_conditioned_sliced_wasserstein",
    "path_distribution.delay_power_aod_joint_sliced_wasserstein",
    "path_distribution.delay_power_aod_link_path_count_conditioned_sliced_wasserstein",
    "path_distribution.full_multipath_path_count_conditioned_mmd_rbf",
    "path_distribution.full_multipath_path_count_conditioned_sliced_wasserstein",
    "csi_distribution.full_csi_projected_mmd_rbf",
    "csi_distribution.full_csi_projected_sliced_wasserstein",
    "csi_distribution.csi_summary_mmd_rbf",
    "csi_distribution.csi_summary_sliced_wasserstein",
    "csi_distribution.pdp_sliced_wasserstein",
)


def _subset(
    payload: Mapping[str, Any], indices: torch.Tensor, device: torch.device
) -> dict[str, Any]:
    count = int(
        next(value for value in payload.values() if isinstance(value, torch.Tensor)).shape[0]
    )
    return {
        key: (
            value[indices].to(device)
            if isinstance(value, torch.Tensor) and value.shape[:1] == (count,)
            else value
        )
        for key, value in payload.items()
    }


def _standardized(
    values: torch.Tensor, reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    mean = reference.mean(0, keepdim=True)
    raw_scale = reference.std(0, unbiased=False, keepdim=True)
    finite_positive = raw_scale[torch.isfinite(raw_scale) & (raw_scale > 0)]
    typical_scale = (
        finite_positive.median()
        if finite_positive.numel()
        else torch.ones((), dtype=reference.dtype, device=reference.device)
    )
    scale_floor = torch.maximum(
        typical_scale * 0.05,
        torch.tensor(1e-6, dtype=reference.dtype, device=reference.device),
    )
    scale = torch.where(torch.isfinite(raw_scale), raw_scale, torch.zeros_like(raw_scale))
    scale = scale.clamp_min(scale_floor)
    audit = {
        "dimensions": int(reference.shape[-1]),
        "reference_raw_scale_min": float(raw_scale.min()),
        "reference_raw_scale_median": float(raw_scale.median()),
        "reference_raw_scale_max": float(raw_scale.max()),
        "applied_scale_floor": float(scale_floor),
        "dimensions_using_scale_floor": int((raw_scale < scale_floor).sum()),
        "nonfinite_candidate_values": int((~torch.isfinite(values)).sum()),
        "nonfinite_reference_values": int((~torch.isfinite(reference)).sum()),
    }
    return (values - mean) / scale, (reference - mean) / scale, audit


def _standardized_swd(
    values: torch.Tensor, reference: torch.Tensor
) -> tuple[float, dict[str, Any]]:
    candidate, target, audit = _standardized(values, reference)
    return float(sliced_wasserstein(candidate, target)), audit


def _standardized_mmd(
    values: torch.Tensor, reference: torch.Tensor
) -> tuple[float, dict[str, Any]]:
    candidate, target, audit = _standardized(values, reference)
    return float(rbf_mmd(candidate, target, [0.5, 1.0, 2.0, 4.0])), audit


def _canonicalize_path_order(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Place active paths first and sort them by delay for link-level metrics."""
    mask = payload["path_mask"].bool()
    delay = payload["path_delays_ns"].float()
    sort_key = torch.where(mask, delay, torch.full_like(delay, torch.inf))
    direct = payload.get("direct_path_mask")
    if isinstance(direct, torch.Tensor):
        sort_key = torch.where(
            direct.bool() & mask,
            torch.full_like(sort_key, -torch.inf),
            sort_key,
        )
    order = torch.argsort(sort_key, dim=-1, stable=True)
    maximum_paths = mask.shape[1]
    result: dict[str, Any] = dict(payload)
    path_aligned_keys = set(PATH_KEYS) | {"direct_path_mask"}
    for key in path_aligned_keys:
        value = payload.get(key)
        if not isinstance(value, torch.Tensor):
            continue
        if value.ndim < 2 or value.shape[1] != maximum_paths:
            continue
        gather_index = order
        while gather_index.ndim < value.ndim:
            gather_index = gather_index.unsqueeze(-1)
        result[key] = value.gather(1, gather_index.expand_as(value))
    return result


def _aod_points(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask = payload["path_mask"].bool()
    return direction_vectors(payload["aod_az_deg"], payload["aod_el_deg"])[mask]


def _power_aod_points(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask = payload["path_mask"].bool()
    directions = direction_vectors(payload["aod_az_deg"], payload["aod_el_deg"])
    power_db = 10.0 * torch.log10(normalized_path_power(payload).clamp_min(1e-20))
    return torch.cat([directions, power_db.unsqueeze(-1)], dim=-1)[mask]


def _delay_power_points(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask = payload["path_mask"].bool()
    delays = payload["path_delays_ns"].float()
    first = torch.where(mask, delays, torch.full_like(delays, torch.inf)).amin(-1)
    relative = (delays - first.unsqueeze(-1)).clamp_min(0.0)
    power_db = 10.0 * torch.log10(normalized_path_power(payload).clamp_min(1e-20))
    return torch.stack([relative, power_db], dim=-1)[mask]


def _delay_power_aod_points(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask = payload["path_mask"].bool()
    delays = payload["path_delays_ns"].float()
    first = torch.where(mask, delays, torch.full_like(delays, torch.inf)).amin(-1)
    relative = (delays - first.unsqueeze(-1)).clamp_min(0.0)
    power_db = 10.0 * torch.log10(normalized_path_power(payload).clamp_min(1e-20))
    directions = direction_vectors(payload["aod_az_deg"], payload["aod_el_deg"])
    return torch.cat(
        [relative.unsqueeze(-1), power_db.unsqueeze(-1), directions], dim=-1
    )[mask]


def _delay_power_link_features(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask = payload["path_mask"].bool()
    delays = payload["path_delays_ns"].float()
    first = torch.where(mask, delays, torch.full_like(delays, torch.inf)).amin(-1)
    relative = (delays - first.unsqueeze(-1)).clamp_min(0.0) * mask
    fraction = normalized_path_power(payload) * mask
    return torch.cat([mask.float(), relative, fraction], dim=-1)


def _delay_power_aod_link_features(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask = payload["path_mask"].bool()
    numeric_mask = mask.float()
    delays = payload["path_delays_ns"].float()
    first = torch.where(mask, delays, torch.full_like(delays, torch.inf)).amin(-1)
    relative = (delays - first.unsqueeze(-1)).clamp_min(0.0) * numeric_mask
    fraction = normalized_path_power(payload) * numeric_mask
    direction = direction_vectors(payload["aod_az_deg"], payload["aod_el_deg"])
    direction = direction * numeric_mask.unsqueeze(-1)
    return torch.cat(
        [numeric_mask, relative, fraction, direction.flatten(1)], dim=-1
    )


def _path_integrity(payload: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    mask = payload["path_mask"].bool()
    delay = payload["path_delays_ns"].float()
    active_delay = delay[mask]
    inactive_delay = delay[~mask]
    seen_inactive = (~mask).to(torch.int64).cumsum(-1) > 0
    noncontiguous = (mask & seen_inactive).any(-1)
    adjacent_active = mask[:, 1:] & mask[:, :-1]
    delay_inversion = adjacent_active & (delay[:, 1:] < delay[:, :-1])
    path_count = mask.sum(-1).long()
    histogram = torch.bincount(path_count, minlength=mask.shape[1] + 1)
    inactive_nonzero: dict[str, int] = {}
    nonfinite_active: dict[str, int] = {}
    for key in PATH_KEYS[1:]:
        value = payload[key]
        if value.ndim != 2 or value.shape != mask.shape:
            continue
        inactive_nonzero[key] = int((value[~mask].abs() > 1e-12).sum())
        nonfinite_active[key] = int((~torch.isfinite(value[mask])).sum())
    quantiles = torch.tensor(
        [0.0, 0.5, 0.9, 0.99, 1.0], dtype=active_delay.dtype, device=active_delay.device
    )
    delay_quantiles = torch.quantile(active_delay, quantiles) if active_delay.numel() else torch.zeros_like(quantiles)
    return {
        "samples": int(mask.shape[0]),
        "active_paths": int(mask.sum()),
        "path_count_histogram": {
            str(index): int(value) for index, value in enumerate(histogram)
        },
        "noncontiguous_mask_samples": int(noncontiguous.sum()),
        "delay_order_violation_samples": int(delay_inversion.any(-1).sum()),
        "active_delay_ns": {
            "minimum": float(delay_quantiles[0]),
            "median": float(delay_quantiles[1]),
            "p90": float(delay_quantiles[2]),
            "p99": float(delay_quantiles[3]),
            "maximum": float(delay_quantiles[4]),
        },
        "inactive_delay_absolute_maximum": (
            float(inactive_delay.abs().max()) if inactive_delay.numel() else 0.0
        ),
        "inactive_slot_nonzero_values": inactive_nonzero,
        "active_slot_nonfinite_values": nonfinite_active,
    }


def _path_count_conditioned_metrics(
    candidate: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    feature_function: Any,
    *,
    include_mmd: bool,
) -> dict[str, Any]:
    candidate_count = candidate["path_mask"].sum(-1).long()
    reference_count = reference["path_mask"].sum(-1).long()
    candidate_features = feature_function(candidate)
    reference_features = feature_function(reference)
    groups: dict[str, Any] = {}
    weighted_swd = 0.0
    weighted_mmd = 0.0
    total_weight = 0.0
    for count in range(1, candidate["path_mask"].shape[1] + 1):
        candidate_keep = candidate_count == count
        reference_keep = reference_count == count
        candidate_samples = int(candidate_keep.sum())
        reference_samples = int(reference_keep.sum())
        if candidate_samples == 0 or reference_samples == 0:
            groups[str(count)] = {
                "candidate_samples": candidate_samples,
                "reference_samples": reference_samples,
                "status": "no_shared_support",
            }
            continue
        swd, swd_audit = _standardized_swd(
            candidate_features[candidate_keep], reference_features[reference_keep]
        )
        mmd: float | None = None
        mmd_audit: dict[str, Any] | None = None
        if include_mmd:
            mmd, mmd_audit = _standardized_mmd(
                candidate_features[candidate_keep], reference_features[reference_keep]
            )
        weight = 0.5 * (
            candidate_samples / max(candidate_count.numel(), 1)
            + reference_samples / max(reference_count.numel(), 1)
        )
        weighted_swd += weight * swd
        if mmd is not None:
            weighted_mmd += weight * mmd
        total_weight += weight
        groups[str(count)] = {
            "candidate_samples": candidate_samples,
            "reference_samples": reference_samples,
            "weight_before_shared_support_normalization": weight,
            "sliced_wasserstein": swd,
            "mmd_rbf": mmd,
            "normalization": swd_audit,
            "mmd_normalization": mmd_audit,
            "status": "ok",
        }
    if total_weight <= 0:
        raise ValueError("candidate and reference have no shared path-count support")
    return {
        "sliced_wasserstein": weighted_swd / total_weight,
        "mmd_rbf": weighted_mmd / total_weight if include_mmd else None,
        "shared_support_weight": total_weight,
        "groups": groups,
    }


def _full_csi_metrics(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    projection_dimension: int = 64,
    seed: int = 2026,
) -> dict[str, Any]:
    def features(csi: torch.Tensor) -> torch.Tensor:
        flat = csi.flatten(1)
        flat = flat / flat.abs().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-20)
        return torch.cat([flat.real, flat.imag], dim=-1).float()

    candidate_features = features(candidate)
    reference_features = features(reference)
    candidate_features, reference_features, normalization = _standardized(
        candidate_features, reference_features
    )
    generator = torch.Generator(device=reference.device).manual_seed(seed)
    projection = torch.randint(
        0,
        2,
        (reference_features.shape[1], projection_dimension),
        generator=generator,
        device=reference.device,
        dtype=torch.int8,
    ).float()
    projection = (projection * 2.0 - 1.0) / math.sqrt(reference_features.shape[1])
    candidate_projected = candidate_features @ projection
    reference_projected = reference_features @ projection
    return {
        "full_csi_projected_mmd_rbf": float(
            rbf_mmd(candidate_projected, reference_projected, [2.0, 4.0, 8.0, 16.0])
        ),
        "full_csi_projected_sliced_wasserstein": float(
            sliced_wasserstein(candidate_projected, reference_projected)
        ),
        "full_csi_random_projection_dimension": projection_dimension,
        "full_csi_normalization": normalization,
    }


def _path_count_total_variation(
    candidate: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]
) -> float:
    candidate_count = candidate["path_mask"].sum(-1).long()
    reference_count = reference["path_mask"].sum(-1).long()
    bins = max(candidate["path_mask"].shape[1], reference["path_mask"].shape[1]) + 1
    candidate_histogram = torch.bincount(candidate_count, minlength=bins).float()
    reference_histogram = torch.bincount(reference_count, minlength=bins).float()
    candidate_histogram /= candidate_histogram.sum().clamp_min(1.0)
    reference_histogram /= reference_histogram.sum().clamp_min(1.0)
    return float(0.5 * (candidate_histogram - reference_histogram).abs().sum())


@torch.inference_mode()
def _distance_metrics(
    candidate: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    candidate_canonical = _canonicalize_path_order(candidate)
    reference_canonical = _canonicalize_path_order(reference)
    candidate_full = full_multipath_features(candidate_canonical)
    reference_full = full_multipath_features(reference_canonical)
    candidate_power_aod_link = power_aod_joint_features(candidate)
    reference_power_aod_link = power_aod_joint_features(reference)
    normalization: dict[str, Any] = {}

    def swd(name: str, values: torch.Tensor, target: torch.Tensor) -> float:
        distance, audit = _standardized_swd(values, target)
        normalization[name] = audit
        return distance

    def mmd(name: str, values: torch.Tensor, target: torch.Tensor) -> float:
        distance, audit = _standardized_mmd(values, target)
        normalization[name] = audit
        return distance

    delay_power_conditioned = _path_count_conditioned_metrics(
        candidate_canonical,
        reference_canonical,
        _delay_power_link_features,
        include_mmd=False,
    )
    power_aod_conditioned = _path_count_conditioned_metrics(
        candidate_canonical,
        reference_canonical,
        power_aod_joint_features,
        include_mmd=False,
    )
    delay_power_aod_conditioned = _path_count_conditioned_metrics(
        candidate_canonical,
        reference_canonical,
        _delay_power_aod_link_features,
        include_mmd=False,
    )
    full_conditioned = _path_count_conditioned_metrics(
        candidate_canonical,
        reference_canonical,
        full_multipath_features,
        include_mmd=True,
    )
    path_metrics = {
        "path_count_total_variation": _path_count_total_variation(candidate, reference),
        "aod_direction_sliced_wasserstein": swd(
            "aod_direction",
            _aod_points(candidate), _aod_points(reference)
        ),
        "power_aod_joint_sliced_wasserstein": swd(
            "power_aod_path_points",
            _power_aod_points(candidate), _power_aod_points(reference)
        ),
        "power_aod_link_moments_sliced_wasserstein": swd(
            "power_aod_link_moments",
            candidate_power_aod_link, reference_power_aod_link
        ),
        "power_aod_link_path_count_conditioned_sliced_wasserstein": power_aod_conditioned[
            "sliced_wasserstein"
        ],
        "delay_power_joint_sliced_wasserstein": swd(
            "delay_power_path_points",
            _delay_power_points(candidate), _delay_power_points(reference)
        ),
        "delay_power_link_path_count_conditioned_sliced_wasserstein": delay_power_conditioned[
            "sliced_wasserstein"
        ],
        "delay_power_aod_joint_sliced_wasserstein": swd(
            "delay_power_aod_path_points",
            _delay_power_aod_points(candidate), _delay_power_aod_points(reference)
        ),
        "delay_power_aod_link_path_count_conditioned_sliced_wasserstein": delay_power_aod_conditioned[
            "sliced_wasserstein"
        ],
        "full_multipath_joint_mmd_rbf": mmd(
            "full_multipath_unconditioned_mmd",
            candidate_full, reference_full
        ),
        "full_multipath_joint_sliced_wasserstein": swd(
            "full_multipath_unconditioned_swd",
            candidate_full, reference_full
        ),
        "full_multipath_path_count_conditioned_mmd_rbf": full_conditioned["mmd_rbf"],
        "full_multipath_path_count_conditioned_sliced_wasserstein": full_conditioned[
            "sliced_wasserstein"
        ],
    }
    csi_metrics = compare_csi_distributions(candidate["csi"], reference["csi"])
    csi_metrics.update(_full_csi_metrics(candidate["csi"], reference["csi"]))
    return {
        "samples": int(candidate["path_mask"].shape[0]),
        "active_paths": int(candidate["path_mask"].sum()),
        "path_distribution": path_metrics,
        "path_integrity": {
            "candidate_before_metric_canonicalization": _path_integrity(candidate),
            "reference_before_metric_canonicalization": _path_integrity(reference),
            "candidate_after_metric_canonicalization": _path_integrity(
                candidate_canonical
            ),
            "reference_after_metric_canonicalization": _path_integrity(
                reference_canonical
            ),
        },
        "normalization_audit": normalization,
        "path_count_conditioned_details": {
            "delay_power_link": delay_power_conditioned,
            "power_aod_link": power_aod_conditioned,
            "delay_power_aod_link": delay_power_aod_conditioned,
            "full_multipath": full_conditioned,
        },
        "csi_distribution": csi_metrics,
    }


def _get_nested(payload: Mapping[str, Any], path: str) -> float:
    value: Any = payload
    for part in path.split("."):
        value = value[part]
    return float(value)


def _comparison(
    scene_metrics: Mapping[str, Any], baseline_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for path in CORE_DISTANCE_PATHS:
        scene = _get_nested(scene_metrics, path)
        baseline = _get_nested(baseline_metrics, path)
        improvement = baseline - scene
        values[path] = {
            "b3_scene_id": scene,
            "b3": baseline,
            "improvement_b3_minus_b3_scene_id": improvement,
            "relative_improvement": improvement / baseline if abs(baseline) > 1e-20 else None,
            "winner": "B3-S-ID" if improvement > 0 else ("B3" if improvement < 0 else "tie"),
        }
    return {
        "lower_is_better": True,
        "positive_improvement_favors": "B3-S-ID",
        "metrics": values,
    }


def _metric_sanity(metrics: Mapping[str, Any]) -> dict[str, Any]:
    values = {path: _get_nested(metrics, path) for path in CORE_DISTANCE_PATHS}
    nonfinite = [path for path, value in values.items() if not math.isfinite(value)]
    extreme = {path: value for path, value in values.items() if abs(value) >= 1e3}
    return {
        "all_core_distances_finite": not nonfinite,
        "nonfinite_core_distances": nonfinite,
        "core_distances_at_least_1e3": extreme,
        "maximum_absolute_core_distance": max(abs(value) for value in values.values()),
        "status": "ok" if not nonfinite and not extreme else "investigate",
    }


def _collect_training_support(dataset: Any, batch_size: int, num_workers: int) -> dict[str, torch.Tensor]:
    chunks: dict[str, list[torch.Tensor]] = {
        "conditions": [],
        "condition_validity": [],
        "map_id": [],
    }
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    for batch in loader:
        for key in chunks:
            chunks[key].append(batch[key].cpu())
    return {key: torch.cat(values) for key, values in chunks.items()}


def _condition_scale(conditions: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
    scale = conditions[:, :3].std(0, unbiased=False).clamp_min(1e-6)
    if bool(validity[:, 2].any()):
        scale[2] = conditions[validity[:, 2], 2].std(unbiased=False).clamp_min(1e-6)
    return scale


@torch.inference_mode()
def _support_statistics(
    reference: Mapping[str, torch.Tensor],
    train: Mapping[str, torch.Tensor],
    *,
    radius: float,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_conditions = train["conditions"].float().to(device)
    train_validity = train["condition_validity"].bool().to(device)
    train_map = train["map_id"].long().to(device)
    test_conditions = reference["conditions"].float()
    test_validity = reference["condition_validity"].bool()
    test_map = reference["map_id"].long()
    scale = _condition_scale(train_conditions, train_validity)
    train_normalized = train_conditions[:, :3] / scale
    test_normalized = test_conditions[:, :3] / scale
    train_normalized[:, 2] = torch.where(
        train_validity[:, 2], train_normalized[:, 2], torch.zeros_like(train_normalized[:, 2])
    )
    test_normalized[:, 2] = torch.where(
        test_validity[:, 2], test_normalized[:, 2], torch.zeros_like(test_normalized[:, 2])
    )
    counts = torch.zeros(test_conditions.shape[0], dtype=torch.long, device=device)
    nearest = torch.full(
        (test_conditions.shape[0],), torch.inf, dtype=torch.float32, device=device
    )
    test_los = test_conditions[:, 3] >= 0.5
    train_los = train_conditions[:, 3] >= 0.5
    groups = sorted(
        {
            (int(test_map[index]), bool(test_los[index]), bool(test_validity[index, 2]))
            for index in range(test_map.numel())
        }
    )
    for map_id, los, k_valid in groups:
        test_indices = torch.nonzero(
            (test_map == map_id)
            & (test_los == los)
            & (test_validity[:, 2] == k_valid),
            as_tuple=False,
        ).flatten()
        train_indices = torch.nonzero(
            (train_map == map_id)
            & (train_los == los)
            & (train_validity[:, 2] == k_valid),
            as_tuple=False,
        ).flatten()
        if train_indices.numel() == 0:
            continue
        for start in range(0, test_indices.numel(), chunk_size):
            selected = test_indices[start : start + chunk_size]
            distance = torch.cdist(test_normalized[selected], train_normalized[train_indices])
            counts[selected] = (distance <= radius).sum(-1)
            nearest[selected] = distance.min(-1).values
    return counts.cpu(), nearest.cpu()


def _density_labels(counts: torch.Tensor, nearest: torch.Tensor) -> torch.Tensor:
    count = counts.numel()
    ordered = sorted(
        range(count),
        key=lambda index: (
            int(counts[index]),
            -float(nearest[index]) if math.isfinite(float(nearest[index])) else -math.inf,
            index,
        ),
    )
    labels = torch.empty(count, dtype=torch.long)
    for rank, index in enumerate(ordered):
        labels[index] = min(2, 3 * rank // max(count, 1))
    return labels


def _density_summary(counts: torch.Tensor, nearest: torch.Tensor) -> dict[str, Any]:
    finite = nearest[torch.isfinite(nearest)]
    return {
        "samples": int(counts.numel()),
        "neighbor_count_min": int(counts.min()),
        "neighbor_count_median": float(counts.float().median()),
        "neighbor_count_mean": float(counts.float().mean()),
        "neighbor_count_max": int(counts.max()),
        "nearest_distance_mean": float(finite.mean()) if finite.numel() else None,
        "nearest_distance_p90": float(torch.quantile(finite, 0.9)) if finite.numel() else None,
        "samples_without_exact_map_los_kvalid_support": int((~torch.isfinite(nearest)).sum()),
    }


def _density_analysis(
    scene: Mapping[str, torch.Tensor],
    baseline: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    counts: torch.Tensor,
    nearest: torch.Tensor,
    *,
    minimum_group_samples: int,
    minimum_per_map_group: int,
    bootstrap_samples: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    labels = _density_labels(counts, nearest)
    group_names = ("sparse", "intermediate", "dense")
    groups: dict[str, Any] = {}
    for group_index, name in enumerate(group_names):
        keep_cpu = labels == group_index
        indices = torch.nonzero(keep_cpu, as_tuple=False).flatten().to(device)
        item: dict[str, Any] = {
            "support": _density_summary(counts[keep_cpu], nearest[keep_cpu]),
            "assignment": "global rank tercile by radius-neighbor count, then nearest distance",
        }
        if indices.numel() < minimum_group_samples:
            item["status"] = "insufficient_samples"
            groups[name] = item
            continue
        scene_group = _subset(scene, indices, device)
        baseline_group = _subset(baseline, indices, device)
        reference_group = _subset(reference, indices, device)
        scene_metrics = _distance_metrics(scene_group, reference_group)
        baseline_metrics = _distance_metrics(baseline_group, reference_group)
        item.update(
            {
                "status": "ok",
                "b3_scene_id": scene_metrics,
                "b3": baseline_metrics,
                "pooled_comparison": _comparison(scene_metrics, baseline_metrics),
            }
        )
        paired_values: dict[str, tuple[list[float], list[float]]] = {
            path: ([], []) for path in CORE_DISTANCE_PATHS
        }
        maps_used: list[int] = []
        for map_id in torch.unique(reference_group["map_id"].long(), sorted=True):
            map_keep = reference_group["map_id"].long() == map_id
            if int(map_keep.sum()) < minimum_per_map_group:
                continue
            map_indices = torch.nonzero(map_keep, as_tuple=False).flatten()
            scene_map = _distance_metrics(
                _subset(scene_group, map_indices, device),
                _subset(reference_group, map_indices, device),
            )
            baseline_map = _distance_metrics(
                _subset(baseline_group, map_indices, device),
                _subset(reference_group, map_indices, device),
            )
            maps_used.append(int(map_id))
            for path, (scene_values, baseline_values) in paired_values.items():
                scene_values.append(_get_nested(scene_map, path))
                baseline_values.append(_get_nested(baseline_map, path))
        item["paired_map_analysis"] = {
            "maps_used": maps_used,
            "minimum_samples_per_map_in_density_group": minimum_per_map_group,
            "metrics": {
                path: paired_map_summary(
                    scene_values,
                    baseline_values,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + metric_index + 101 * group_index,
                )
                for metric_index, (path, (scene_values, baseline_values)) in enumerate(
                    paired_values.items()
                )
                if scene_values
            },
        }
        groups[name] = item
    return {
        "support_radius_standardized": None,
        "density_assignment": (
            "Relative global terciles. Higher density means more same-Map, same-LoS and "
            "same-K-validity training samples inside the standardized condition radius."
        ),
        "groups": groups,
        "interpretation": (
            "Compare positive B3-minus-B3-S-ID improvements from sparse to dense groups. "
            "A monotone increase supports the condition-coverage hypothesis; pooled results "
            "must be checked against the paired-Map summaries."
        ),
    }


def _condition_view(payload: Mapping[str, torch.Tensor], generated: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if generated:
        return payload["actual_conditions"].float(), payload["actual_condition_validity"].bool()
    return payload["conditions"].float(), payload["condition_validity"].bool()


@torch.inference_mode()
def _match_reference_donors(
    candidate: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_conditions, candidate_validity = _condition_view(candidate, True)
    reference_conditions, reference_validity = _condition_view(reference, False)
    pooled = torch.cat([candidate_conditions[:, :3], reference_conditions[:, :3]])
    scale = pooled.std(0, unbiased=False).clamp_min(1e-6)
    candidate_count = candidate["path_mask"].sum(-1)
    reference_count = reference["path_mask"].sum(-1)
    candidate_los = candidate_conditions[:, 3] >= 0.5
    reference_los = reference_conditions[:, 3] >= 0.5
    candidate_normalized = candidate_conditions[:, :3] / scale
    reference_normalized = reference_conditions[:, :3] / scale
    candidate_normalized[:, 2] = torch.where(
        candidate_validity[:, 2], candidate_normalized[:, 2], torch.zeros_like(candidate_normalized[:, 2])
    )
    reference_normalized[:, 2] = torch.where(
        reference_validity[:, 2], reference_normalized[:, 2], torch.zeros_like(reference_normalized[:, 2])
    )
    matched_chunks: list[torch.Tensor] = []
    donor_chunks: list[torch.Tensor] = []
    distance_chunks: list[torch.Tensor] = []
    groups = sorted(
        {
            (int(candidate_count[index]), bool(candidate_los[index]), bool(candidate_validity[index, 2]))
            for index in range(candidate_count.numel())
        }
    )
    for path_count, los, k_valid in groups:
        candidate_indices = torch.nonzero(
            (candidate_count == path_count)
            & (candidate_los == los)
            & (candidate_validity[:, 2] == k_valid),
            as_tuple=False,
        ).flatten()
        reference_indices = torch.nonzero(
            (reference_count == path_count)
            & (reference_los == los)
            & (reference_validity[:, 2] == k_valid),
            as_tuple=False,
        ).flatten()
        if reference_indices.numel() == 0:
            continue
        distance = torch.cdist(
            candidate_normalized[candidate_indices], reference_normalized[reference_indices]
        )
        nearest_distance, nearest_local = distance.min(dim=1)
        matched_chunks.append(candidate_indices)
        donor_chunks.append(reference_indices[nearest_local])
        distance_chunks.append(nearest_distance)
    if not matched_chunks:
        empty = torch.empty(0, dtype=torch.long, device=candidate_conditions.device)
        return empty, empty, torch.empty(0, device=candidate_conditions.device)
    matched = torch.cat(matched_chunks)
    order = torch.argsort(matched)
    return matched[order], torch.cat(donor_chunks)[order], torch.cat(distance_chunks)[order]


def _path_match_features(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask = payload["path_mask"].bool()
    delay = payload["path_delays_ns"].float()
    first = torch.where(mask, delay, torch.full_like(delay, torch.inf)).amin(-1)
    relative = (delay - first.unsqueeze(-1)).clamp_min(0.0)
    count = mask.sum(-1).float().clamp_min(1.0)
    mean = (relative * mask).sum(-1) / count
    variance = (
        (relative - mean.unsqueeze(-1)).square() * mask
    ).sum(-1) / count
    relative = relative / variance.sqrt().clamp_min(1.0).unsqueeze(-1)
    log_power = torch.log10(normalized_path_power(payload).clamp_min(1e-20))
    aod = direction_vectors(payload["aod_az_deg"], payload["aod_el_deg"])
    aoa = direction_vectors(payload["aoa_az_deg"], payload["aoa_el_deg"])
    return torch.cat([relative.unsqueeze(-1), log_power.unsqueeze(-1), aod, aoa], dim=-1)


@torch.inference_mode()
def _align_donor_paths(
    candidate: Mapping[str, torch.Tensor], donor: Mapping[str, torch.Tensor]
) -> tuple[dict[str, Any], torch.Tensor]:
    batch, maximum_paths = candidate["path_mask"].shape
    orders = torch.arange(maximum_paths, device=candidate["path_mask"].device).repeat(batch, 1)
    costs = torch.zeros(batch, device=orders.device)
    los = candidate["target_conditions"][:, 3] >= 0.5
    path_count = candidate["path_mask"].sum(-1).long()
    candidate_features = _path_match_features(candidate)
    donor_features = _path_match_features(donor)
    groups = sorted(
        {(int(path_count[index]), bool(los[index])) for index in range(batch)}
    )
    for count, is_los in groups:
        selected = torch.nonzero(
            (path_count == count) & (los == is_los), as_tuple=False
        ).flatten()
        fixed = 1 if is_los else 0
        variable = tuple(range(fixed, count))
        permutations = torch.tensor(
            [tuple(range(fixed)) + order for order in itertools.permutations(variable)],
            dtype=torch.long,
            device=orders.device,
        )
        permuted_donor = donor_features[selected][:, permutations, :]
        match_cost = (
            candidate_features[selected, :count].unsqueeze(1) - permuted_donor
        ).square().mean(dim=(-1, -2))
        best_cost, best_index = match_cost.min(-1)
        orders[selected, :count] = permutations[best_index]
        costs[selected] = best_cost
    aligned: dict[str, Any] = dict(donor)
    for key in PATH_KEYS:
        value = donor[key]
        if value.ndim >= 2 and value.shape[1] == maximum_paths:
            gather_index = orders
            while gather_index.ndim < value.ndim:
                gather_index = gather_index.unsqueeze(-1)
            gather_index = gather_index.expand_as(value)
            aligned[key] = value.gather(1, gather_index)
    return aligned, costs


def _synthesize(
    payload: Mapping[str, torch.Tensor], synthesizer: MIMOOFDMSynthesizer, batch_size: int
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, payload["path_mask"].shape[0], batch_size):
        stop = min(start + batch_size, payload["path_mask"].shape[0])
        chunks.append(
            synthesizer(
                payload["path_delays_ns"][start:stop],
                payload["path_powers_linear"][start:stop],
                payload["path_phases_rad"][start:stop],
                payload["aoa_az_deg"][start:stop],
                payload["aoa_el_deg"][start:stop],
                payload["aod_az_deg"][start:stop],
                payload["aod_el_deg"][start:stop],
                payload["path_mask"][start:stop],
            )
        )
    return torch.cat(chunks)


def _projection_changes(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    *,
    replaced_parameters: tuple[str, ...],
) -> dict[str, Any]:
    mask = after["path_mask"].bool()
    before_power = before["path_powers_linear"].float() * mask
    after_power = after["path_powers_linear"].float() * mask
    before_fraction = before_power / before_power.sum(-1, keepdim=True).clamp_min(1e-20)
    after_fraction = after_power / after_power.sum(-1, keepdim=True).clamp_min(1e-20)
    delay_change = (
        after["path_delays_ns"] - before["path_delays_ns"]
    )[mask].abs()
    power_fraction_change = (after_fraction - before_fraction).abs().sum(-1)
    parameter_changes = {
        "path_delays_ns": {
            "active_value_mae": float(delay_change.mean()),
            "active_value_maximum": float(delay_change.max()),
        },
        "path_powers_linear": {
            "power_fraction_l1_mean": float(power_fraction_change.mean()),
            "power_fraction_l1_maximum": float(power_fraction_change.max()),
        },
    }
    projected_keys = {"path_delays_ns", "path_powers_linear"}
    replaced = set(replaced_parameters)
    unreplaced = {
        key: parameter_changes[key] for key in sorted(projected_keys - replaced)
    }
    coupling_flags = []
    if (
        "path_delays_ns" in unreplaced
        and unreplaced["path_delays_ns"]["active_value_mae"] > 0.1
    ):
        coupling_flags.append("unreplaced_delay_mae_exceeds_0.1_ns")
    if (
        "path_powers_linear" in unreplaced
        and unreplaced["path_powers_linear"]["power_fraction_l1_mean"] > 1e-3
    ):
        coupling_flags.append("unreplaced_power_fraction_l1_exceeds_1e-3")
    return {
        "parameter_changes": parameter_changes,
        "replaced_parameters": list(replaced_parameters),
        "unreplaced_variables_changed_by_projection": unreplaced,
        "single_group_attribution_guard": {
            "valid": not coupling_flags,
            "flags": coupling_flags,
            "meaning": (
                "When invalid, this repair cannot be interpreted as changing only the named "
                "parameter group because physical projection materially changed an unreplaced variable."
            ),
        },
        "projection_layer_does_not_modify": [
            "path_phases_rad",
            "aoa_az_deg",
            "aoa_el_deg",
            "aod_az_deg",
            "aod_el_deg",
        ],
        "interpretation": (
            "These values compare the donor-replaced tensor immediately before physical "
            "projection with the tensor after projection. A listed unreplaced variable "
            "therefore changed only because of constraint coupling."
        ),
    }


def _treatment_changes(
    control: Mapping[str, torch.Tensor], repaired: Mapping[str, torch.Tensor]
) -> dict[str, float]:
    mask = repaired["path_mask"].bool()
    control_power = control["path_powers_linear"].float() * mask
    repaired_power = repaired["path_powers_linear"].float() * mask
    control_fraction = control_power / control_power.sum(-1, keepdim=True).clamp_min(1e-20)
    repaired_fraction = repaired_power / repaired_power.sum(-1, keepdim=True).clamp_min(1e-20)
    control_aod = direction_vectors(control["aod_az_deg"], control["aod_el_deg"])
    repaired_aod = direction_vectors(repaired["aod_az_deg"], repaired["aod_el_deg"])
    return {
        "delay_active_mae_ns": float(
            (repaired["path_delays_ns"] - control["path_delays_ns"])[mask].abs().mean()
        ),
        "power_fraction_l1_mean": float(
            (repaired_fraction - control_fraction).abs().sum(-1).mean()
        ),
        "aod_direction_euclidean_mean": float(
            (repaired_aod - control_aod)[mask].norm(dim=-1).mean()
        ),
        "phase_circular_distance_mean": float(
            torch.angle(
                torch.exp(
                    1j
                    * (
                        repaired["path_phases_rad"][mask]
                        - control["path_phases_rad"][mask]
                    )
                )
            )
            .abs()
            .mean()
        ),
    }


def _condition_consistency(payload: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    actual, validity = compute_physical_conditions(
        payload["path_powers_linear"],
        payload["path_delays_ns"],
        payload["path_mask"],
        payload["direct_path_mask"],
    )
    target = payload["target_conditions"]
    target_validity = payload["target_condition_validity"].bool()
    fields = ("total_power_db", "delay_spread_ns", "k_factor_db", "los_status")
    errors: dict[str, Any] = {}
    for field_index, field in enumerate(fields):
        valid = target_validity[:, field_index] & validity[:, field_index]
        difference = (actual[:, field_index] - target[:, field_index]).abs()
        errors[field] = {
            "valid_samples": int(valid.sum()),
            "mae": float(difference[valid].mean()) if bool(valid.any()) else None,
            "maximum_absolute_error": float(difference[valid].max()) if bool(valid.any()) else None,
        }
    return {"fields": errors, "validity_match_rate": float((validity == target_validity).all(-1).float().mean())}


def _core_improvements(
    baseline: Mapping[str, Any], repaired: Mapping[str, Any]
) -> dict[str, float]:
    return {
        path: _get_nested(baseline, path) - _get_nested(repaired, path)
        for path in CORE_DISTANCE_PATHS
    }


def _interaction_effects(repairs: Mapping[str, Any]) -> dict[str, Any]:
    definitions = {
        "power_aod_beyond_singletons": ("power_aod_joint", ("power", "aod")),
        "delay_power_beyond_singletons": ("delay_power_joint", ("delay", "power")),
        "delay_power_aod_beyond_singletons": (
            "delay_power_aod_joint",
            ("delay", "power", "aod"),
        ),
    }
    result: dict[str, Any] = {}
    for label, (joint, singletons) in definitions.items():
        result[label] = {
            path: repairs[joint]["improvement_control_minus_repaired"][path]
            - sum(
                repairs[singleton]["improvement_control_minus_repaired"][path]
                for singleton in singletons
            )
            for path in CORE_DISTANCE_PATHS
        }
    return result


@torch.inference_mode()
def _counterfactual_analysis(
    scene: Mapping[str, torch.Tensor],
    baseline: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    *,
    data_config: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    matched, donor_indices, distances = _match_reference_donors(scene, reference)
    if matched.numel() < 2:
        raise ValueError("fewer than two exact path-count/LoS/K-validity donor matches")
    scene_matched = _subset(scene, matched, device)
    baseline_matched = _subset(baseline, matched, device)
    donor = _subset(reference, donor_indices, device)
    donor_aligned, path_costs = _align_donor_paths(scene_matched, donor)
    donor_direct = first_active_path_mask(donor_aligned["path_mask"]) & (
        donor_aligned["conditions"][:, 3:4] >= 0.5
    )
    donor_aligned["direct_path_mask"] = donor_direct

    data = load_yaml(data_config)["data"]
    system = SystemConfig.from_mapping(data["system"])
    synthesizer = MIMOOFDMSynthesizer(system).to(device).eval()
    constraint = PhysicsConstraintLayer(
        enforce_total_power=True,
        enforce_delay_spread=True,
        enforce_k_factor=True,
        enforce_los_direct_path=True,
        epsilon=1e-20,
        invalid_k_fill_db=float(data.get("k_factor", {}).get("invalid_fill_db", 0.0)),
    ).to(device).eval()
    scene_metrics = _distance_metrics(scene_matched, donor)
    baseline_metrics = _distance_metrics(baseline_matched, donor)

    control: dict[str, Any] = dict(scene_matched)
    control_before_projection = {
        key: control[key].clone() for key in PATH_KEYS
    }
    control_constrained = constraint(
        control["path_delays_ns"],
        control["path_powers_linear"].clamp_min(1e-20),
        control["path_mask"],
        control["target_conditions"],
        target_condition_validity=control["target_condition_validity"],
        direct_path_mask=control["direct_path_mask"],
    )
    control["path_delays_ns"] = control_constrained.delays_ns
    control["path_powers_linear"] = control_constrained.powers_linear
    control["direct_path_mask"] = control_constrained.direct_path_mask
    control["actual_conditions"] = control_constrained.actual_conditions
    control["actual_condition_validity"] = control_constrained.condition_validity
    control["csi"] = _synthesize(control, synthesizer, batch_size)
    control_metrics = _distance_metrics(control, donor)
    repairs: dict[str, Any] = {}
    for name, keys in REPAIR_GROUPS.items():
        repaired: dict[str, Any] = dict(scene_matched)
        for key in keys:
            repaired[key] = donor_aligned[key].clone()
        raw_replacement_integrity = _path_integrity(repaired)
        repaired = _canonicalize_path_order(repaired)
        before_projection = {
            key: repaired[key].clone() for key in PATH_KEYS
        }
        constrained = constraint(
            repaired["path_delays_ns"],
            repaired["path_powers_linear"].clamp_min(1e-20),
            repaired["path_mask"],
            repaired["target_conditions"],
            target_condition_validity=repaired["target_condition_validity"],
            direct_path_mask=repaired["direct_path_mask"],
        )
        repaired["path_delays_ns"] = constrained.delays_ns
        repaired["path_powers_linear"] = constrained.powers_linear
        repaired["direct_path_mask"] = constrained.direct_path_mask
        repaired["actual_conditions"] = constrained.actual_conditions
        repaired["actual_condition_validity"] = constrained.condition_validity
        repaired["csi"] = _synthesize(repaired, synthesizer, batch_size)
        metrics = _distance_metrics(repaired, donor)
        repairs[name] = {
            "replaced_parameters": list(keys),
            "metrics": metrics,
            "metric_sanity": _metric_sanity(metrics),
            "improvement_control_minus_repaired": _core_improvements(control_metrics, metrics),
            "constraint_feasible_rate": float(constrained.feasible.float().mean()),
            "condition_consistency": _condition_consistency(repaired),
            "integrity_after_replacement_before_canonicalization": raw_replacement_integrity,
            "integrity_before_projection": _path_integrity(before_projection),
            "integrity_after_projection": _path_integrity(repaired),
            "constraint_projection_change": _projection_changes(
                before_projection,
                repaired,
                replaced_parameters=keys,
            ),
            "total_treatment_change_from_projection_only_control": _treatment_changes(
                control, repaired
            ),
        }
    unique_donors, donor_reuse = torch.unique(
        donor_indices, return_counts=True
    )
    return {
        "matching": {
            "condition_rule": (
                "nearest real link with exact path count, LoS status and K-validity; donor reuse allowed"
            ),
            "path_rule": (
                "minimum joint delay/log-power/AoD/AoA cost; the first LoS path is fixed as direct"
            ),
            "matched_samples": int(matched.numel()),
            "match_rate": float(matched.numel() / scene["path_mask"].shape[0]),
            "condition_distance_mean": float(distances.mean()),
            "condition_distance_p90": float(torch.quantile(distances, 0.9)),
            "condition_distance_p99": float(torch.quantile(distances, 0.99)),
            "condition_distance_maximum": float(distances.max()),
            "condition_distance_above_0.1": int((distances > 0.1).sum()),
            "condition_distance_above_1.0": int((distances > 1.0).sum()),
            "unique_real_donors": int(unique_donors.numel()),
            "unique_donor_rate": float(unique_donors.numel() / matched.numel()),
            "maximum_donor_reuse_count": int(donor_reuse.max()),
            "path_alignment_cost_mean": float(path_costs.mean()),
            "path_alignment_cost_p90": float(torch.quantile(path_costs, 0.9)),
        },
        "b3_scene_id_unrepaired": scene_metrics,
        "b3_scene_id_unrepaired_metric_sanity": _metric_sanity(scene_metrics),
        "projection_only_control": {
            "metrics": control_metrics,
            "metric_sanity": _metric_sanity(control_metrics),
            "difference_unrepaired_minus_control": _core_improvements(
                scene_metrics, control_metrics
            ),
            "constraint_feasible_rate": float(control_constrained.feasible.float().mean()),
            "condition_consistency": _condition_consistency(control),
            "constraint_projection_change": _projection_changes(
                control_before_projection,
                control,
                replaced_parameters=(),
            ),
        },
        "b3_same_indices": baseline_metrics,
        "b3_same_indices_metric_sanity": _metric_sanity(baseline_metrics),
        "b3_vs_b3_scene_id_on_matched_subset": _comparison(scene_metrics, baseline_metrics),
        "repairs": repairs,
        "non_additive_interactions": _interaction_effects(repairs),
        "interpretation": (
            "Positive repair improvement means the repaired B3-S-ID distribution moved closer to "
            "condition-matched real data. Power and delay are projected back to the original target "
            "power, delay spread and K-factor before synthesis. Improvements are measured against "
            "the projection-only/resynthesis control, not the stored CSI tensor. Joint and singleton repairs share "
            "parameters, so improvements and interaction residuals are diagnostic, not independent "
            "causal effects."
        ),
    }


def _validate_payload(source: Mapping[str, Any]) -> None:
    required_top = {"reference", "correct_map", "physics_only_baseline"}
    if missing := sorted(required_top - source.keys()):
        raise ValueError(f"generated artifact is missing {missing}")
    if source["physics_only_baseline"] is None:
        raise ValueError(
            "generated artifact has no physics_only_baseline; rerun evaluate_scene_id_protocol.py "
            "with --baseline-checkpoint"
        )
    required_reference = set(PATH_KEYS) | {
        "csi",
        "conditions",
        "condition_validity",
        "map_id",
    }
    required_generated = set(PATH_KEYS) | {
        "csi",
        "actual_conditions",
        "actual_condition_validity",
        "direct_path_mask",
        "target_conditions",
        "target_condition_validity",
    }
    if missing := sorted(required_reference - source["reference"].keys()):
        raise ValueError(f"reference payload is missing {missing}")
    for label in ("correct_map", "physics_only_baseline"):
        if missing := sorted(required_generated - source[label].keys()):
            raise ValueError(f"{label} payload is missing {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose B3 versus B3-S-ID AoD/joint-path errors, stratify them by local "
            "Map-condition support, and repair parameter groups before full CSI evaluation."
        )
    )
    parser.add_argument("--generated", required=True)
    parser.add_argument("--scene-plan", required=True)
    parser.add_argument("--prepared-root", default="artifacts/prepared_d2los")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--maximum-samples", type=int, default=2048)
    parser.add_argument("--support-radius", type=float, default=1.0)
    parser.add_argument("--minimum-density-group", type=int, default=64)
    parser.add_argument("--minimum-per-map-density-group", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(
        args.maximum_samples,
        args.minimum_density_group,
        args.minimum_per_map_density_group,
        args.bootstrap_samples,
        args.batch_size,
    ) < 1:
        parser.error("sample, bootstrap and batch arguments must be positive")
    if args.support_radius <= 0:
        parser.error("--support-radius must be positive")

    device = resolve_device(args.device)
    source = torch.load(args.generated, map_location="cpu", weights_only=False)
    _validate_payload(source)
    plan = load_seen_map_spatial_plan(args.scene_plan)
    if source.get("scene_plan_fingerprint") not in {None, plan["fingerprint"]}:
        raise ValueError("generated artifact and --scene-plan fingerprints differ")
    sample_count = source["reference"]["csi"].shape[0]
    for label in ("correct_map", "physics_only_baseline"):
        if source[label]["csi"].shape[0] != sample_count:
            raise ValueError(f"{label} and reference sample counts differ")
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.arange(sample_count)
    if sample_count > args.maximum_samples:
        indices = torch.randperm(sample_count, generator=generator)[: args.maximum_samples].sort().values
    reference = _subset(source["reference"], indices, device)
    scene = _subset(source["correct_map"], indices, device)
    baseline = _subset(source["physics_only_baseline"], indices, device)

    print(json.dumps({"event": "distribution_decomposition", "samples": int(indices.numel())}), flush=True)
    scene_metrics = _distance_metrics(scene, reference)
    baseline_metrics = _distance_metrics(baseline, reference)

    print(json.dumps({"event": "loading_training_condition_support"}), flush=True)
    datasets = make_scene_id_datasets(args.prepared_root, plan)
    train = _collect_training_support(datasets["train"], args.batch_size, args.num_workers)
    counts, nearest = _support_statistics(
        reference,
        train,
        radius=args.support_radius,
        chunk_size=args.batch_size,
        device=device,
    )
    print(json.dumps({"event": "density_stratified_comparison"}), flush=True)
    density = _density_analysis(
        scene,
        baseline,
        reference,
        counts,
        nearest,
        minimum_group_samples=args.minimum_density_group,
        minimum_per_map_group=args.minimum_per_map_density_group,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        device=device,
    )
    density["support_radius_standardized"] = args.support_radius

    print(json.dumps({"event": "counterfactual_repairs"}), flush=True)
    counterfactual = _counterfactual_analysis(
        scene,
        baseline,
        reference,
        data_config=args.data_config,
        batch_size=args.batch_size,
        device=device,
    )

    report = {
        "format_version": 2,
        "diagnostic": "scene_joint_distribution_error_decomposition",
        "revision": {
            "name": "robust_path_order_and_projection_audit_v2",
            "purpose": (
                "Test whether extreme SW values came from padded near-zero-variance columns, "
                "path-order mismatch, invalid slots, or physical-projection coupling."
            ),
        },
        "source": str(Path(args.generated).resolve()),
        "scene_plan": str(Path(args.scene_plan).resolve()),
        "scene_plan_fingerprint": plan["fingerprint"],
        "samples": int(indices.numel()),
        "device": str(device),
        "A_distribution_decomposition": {
            "b3_scene_id": scene_metrics,
            "b3": baseline_metrics,
            "metric_sanity": {
                "b3_scene_id": _metric_sanity(scene_metrics),
                "b3": _metric_sanity(baseline_metrics),
            },
            "comparison": _comparison(scene_metrics, baseline_metrics),
            "metric_definitions": {
                "path_point_joint_metrics": (
                    "Active-path point clouds after reference standardization; circular angles "
                    "are represented as three-dimensional unit directions."
                ),
                "link_joint_metrics": (
                    "Paths are canonicalized by active mask and ascending Delay. Link-level "
                    "metrics are then computed within each path-count stratum and macro-combined "
                    "over shared support; path-count mismatch is reported separately by total variation."
                ),
                "robust_standardization": (
                    "Every feature uses a reference-derived scale with a floor equal to 5% of "
                    "the median positive reference scale (at least 1e-6). The report records "
                    "the floor and affected dimensions, preventing near-constant padded columns "
                    "from creating million-scale artifacts."
                ),
                "full_csi_projected_metrics": (
                    "MMD/SW on a fixed 64-dimensional Rademacher projection of the complete "
                    "energy-normalized complex CSI tensor (real and imaginary parts)."
                ),
                "csi_summary_metrics": (
                    "Existing project CSI-summary MMD/SW plus PDP SW, retained for continuity "
                    "with earlier B3 evaluations."
                ),
            },
            "question": (
                "Does B3-S-ID improve marginal AoD while worsening Power-AoD, Delay-Power "
                "or the complete link-level multipath joint distribution?"
            ),
        },
        "B_condition_density": density,
        "C_counterfactual_repairs": counterfactual,
        "cautions": [
            "Every reported distance is a distribution-level statistic; generated samples are not paired with real links by identity.",
            "Unconditioned padded-link distances are retained as secondary diagnostics only; conclusions use canonical path-count-conditioned link metrics plus a separate path-count TV distance.",
            "Density groups are relative terciles and do not turn observational coverage into a causal treatment.",
            "Counterfactual repairs use matched real donors and a physical projection layer; they are controlled diagnostics, not unique causal decompositions.",
            "A full-path improvement without CSI improvement points to phase/frequency/array coherence or metric sensitivity, not automatically to a synthesizer defect.",
        ],
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "diagnostic_complete",
                "output": str(Path(args.output).resolve()),
                "samples": int(indices.numel()),
                "counterfactual_matches": counterfactual["matching"]["matched_samples"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
