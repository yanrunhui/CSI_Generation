#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.data.scene_id import (
    load_seen_map_spatial_plan,
    make_scene_id_datasets,
)
from physics_csi.evaluation.scene_diagnostics import (
    rotation_aligned_scene_structure_distances,
    scene_structure_distances,
)
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


PAYLOAD_KEYS = (
    "csi",
    "source_id",
    "tx_xyz_m",
    "rx_xyz_m",
    "path_mask",
    "direct_path_mask",
    "path_delays_ns",
    "path_powers_linear",
    "path_phases_rad",
    "aoa_az_deg",
    "aoa_el_deg",
    "aod_az_deg",
    "aod_el_deg",
    "conditions",
    "condition_validity",
)


def _stack(records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([record[key] for record in records]) for key in PAYLOAD_KEYS}


def _to_device(
    payload: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in payload.items()}


def _condition_matrix(payload: dict[str, torch.Tensor], std: torch.Tensor) -> torch.Tensor:
    values = payload["conditions"].float()
    validity = payload["condition_validity"].bool()
    normalized = values[:, :3] / std[:3].clamp_min(1e-6)
    normalized[:, 2] = torch.where(validity[:, 2], normalized[:, 2], 0.0)
    return normalized


def _one_to_one_matched_indices(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
    std: torch.Tensor,
    maximum_distance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedy one-to-one physical-condition matching without replacement.

    Rows with the closest available neighbor are assigned first.  LoS and the
    K-factor validity mask must agree exactly.  Every returned index is unique
    on both sides.
    """
    first_values = _condition_matrix(first, std)
    second_values = _condition_matrix(second, std)
    distances = torch.cdist(first_values, second_values)
    first_los = first["conditions"][:, 3] >= 0.5
    second_los = second["conditions"][:, 3] >= 0.5
    first_k_valid = first["condition_validity"][:, 2].bool()
    second_k_valid = second["condition_validity"][:, 2].bool()
    compatible = (
        (first_los[:, None] == second_los[None, :])
        & (first_k_valid[:, None] == second_k_valid[None, :])
        & torch.isfinite(distances)
        & (distances <= maximum_distance)
    )
    distances = distances.masked_fill(~compatible, torch.inf)
    nearest = distances.amin(dim=1)
    row_order = torch.argsort(nearest).cpu()
    distances_cpu = distances.cpu()
    second_used = torch.zeros(distances.shape[1], dtype=torch.bool)
    matched_first: list[int] = []
    matched_second: list[int] = []
    matched_distance: list[float] = []
    for first_index in row_order.tolist():
        row = distances_cpu[first_index]
        for second_index in torch.argsort(row).tolist():
            distance = float(row[second_index])
            if not math.isfinite(distance) or distance > maximum_distance:
                break
            if not bool(second_used[second_index]):
                second_used[second_index] = True
                matched_first.append(first_index)
                matched_second.append(second_index)
                matched_distance.append(distance)
                break
    return (
        torch.tensor(matched_first, dtype=torch.long),
        torch.tensor(matched_second, dtype=torch.long),
        torch.tensor(matched_distance, dtype=torch.float32),
    )


def _condition_audit(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
    distances: torch.Tensor,
    std: torch.Tensor,
) -> dict[str, Any]:
    first_conditions = first["conditions"].float().cpu()
    second_conditions = second["conditions"].float().cpu()
    first_validity = first["condition_validity"].bool().cpu()
    second_validity = second["condition_validity"].bool().cpu()
    first_los = first_conditions[:, 3] >= 0.5
    second_los = second_conditions[:, 3] >= 0.5
    k_valid = first_validity[:, 2] & second_validity[:, 2]
    standardized_mean_gap = (
        (first_conditions[:, :2].mean(0) - second_conditions[:, :2].mean(0)).abs()
        / std[:2].cpu().clamp_min(1e-6)
    )
    if bool(k_valid.any()):
        k_mean_gap = float(
            (
                first_conditions[k_valid, 2].mean()
                - second_conditions[k_valid, 2].mean()
            ).abs()
            / std[2].cpu().clamp_min(1e-6)
        )
    else:
        k_mean_gap = None
    second_unique = int(torch.unique(second["_selected_indices"].cpu()).numel())
    first_unique = int(torch.unique(first["_selected_indices"].cpu()).numel())
    samples = int(first_conditions.shape[0])
    return {
        "samples": samples,
        "first_unique_samples": first_unique,
        "second_unique_samples": second_unique,
        "first_duplicate_rate": 1.0 - first_unique / max(samples, 1),
        "second_duplicate_rate": 1.0 - second_unique / max(samples, 1),
        "pair_distance_mean": float(distances.mean()),
        "pair_distance_maximum": float(distances.max()),
        "standardized_condition_mean_gap": {
            "total_power_db": float(standardized_mean_gap[0]),
            "delay_spread_ns": float(standardized_mean_gap[1]),
            "k_factor_db_valid_only": k_mean_gap,
            "los_prevalence": float((first_los.float().mean() - second_los.float().mean()).abs()),
        },
        "los_exact_for_every_pair": bool(torch.equal(first_los, second_los)),
        "k_validity_exact_for_every_pair": bool(
            torch.equal(first_validity[:, 2], second_validity[:, 2])
        ),
    }


def _selected_payload(
    payload: dict[str, torch.Tensor],
    indices: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    selected = {
        key: value[indices]
        for key, value in payload.items()
        if key in PAYLOAD_KEYS
    }
    selected["_selected_indices"] = indices.clone()
    return _to_device(selected, device)


def _mean_metrics(repeats: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    names = sorted(
        {
            name
            for repeat in repeats
            for name in repeat[key]
            if repeat[key][name] is not None
        }
    )
    result: dict[str, float | None] = {}
    for name in names:
        values = [repeat[key][name] for repeat in repeats if repeat[key][name] is not None]
        result[name] = float(torch.tensor(values, dtype=torch.float64).mean()) if values else None
    return result


def _frame_from_receiver_coordinates(payload: dict[str, torch.Tensor]) -> dict[str, float]:
    xy = payload["rx_xyz_m"][:, :2].double()
    centered = xy - xy.mean(0, keepdim=True)
    covariance = centered.T @ centered / max(xy.shape[0], 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    principal = eigenvectors[:, -1]
    displacement = (payload["rx_xyz_m"][:, :2] - payload["tx_xyz_m"][:, :2]).double()
    mean_displacement = displacement.mean(0)
    projection = float(torch.dot(principal, mean_displacement))
    if projection < 0 or (
        abs(projection) < 1e-12
        and (float(principal[0]) < 0 or (float(principal[0]) == 0 and float(principal[1]) < 0))
    ):
        principal = -principal
    yaw = math.degrees(math.atan2(float(principal[1]), float(principal[0])))
    denominator = float(eigenvalues.sum())
    anisotropy = (
        float((eigenvalues[-1] - eigenvalues[0]) / eigenvalues.sum())
        if denominator > 1e-20
        else 0.0
    )
    return {
        "receiver_pca_frame_yaw_deg": yaw,
        "receiver_pca_anisotropy": anisotropy,
        "receiver_pca_eigenvalue_min": float(eigenvalues[0]),
        "receiver_pca_eigenvalue_max": float(eigenvalues[-1]),
    }


def _circular_absolute_error_deg(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return torch.remainder(first - second + 180.0, 360.0).sub(180.0).abs()


def _direct_path_geometry_audit(payload: dict[str, torch.Tensor]) -> dict[str, Any]:
    direct = payload["direct_path_mask"].bool()
    rows, paths = torch.nonzero(direct, as_tuple=True)
    if rows.numel() == 0:
        return {"direct_paths": 0}
    displacement = payload["rx_xyz_m"].float() - payload["tx_xyz_m"].float()
    horizontal = torch.linalg.vector_norm(displacement[:, :2], dim=-1)
    forward_azimuth = torch.rad2deg(torch.atan2(displacement[:, 1], displacement[:, 0]))
    forward_elevation = torch.rad2deg(torch.atan2(displacement[:, 2], horizontal))
    aod_azimuth = payload["aod_az_deg"][rows, paths].float()
    aod_elevation = payload["aod_el_deg"][rows, paths].float()
    aoa_azimuth = payload["aoa_az_deg"][rows, paths].float()
    aoa_elevation = payload["aoa_el_deg"][rows, paths].float()
    forward_aod_error = _circular_absolute_error_deg(aod_azimuth, forward_azimuth[rows])
    reverse_aod_error = _circular_absolute_error_deg(
        aod_azimuth, forward_azimuth[rows] + 180.0
    )
    reverse_aoa_error = _circular_absolute_error_deg(
        aoa_azimuth, forward_azimuth[rows] + 180.0
    )
    forward_aoa_error = _circular_absolute_error_deg(aoa_azimuth, forward_azimuth[rows])
    aod_elevation_error = (aod_elevation - forward_elevation[rows]).abs()
    aoa_elevation_error = (aoa_elevation + forward_elevation[rows]).abs()

    def summary(values: torch.Tensor) -> dict[str, float]:
        return {
            "mean_deg": float(values.mean()),
            "median_deg": float(values.median()),
            "p90_deg": float(torch.quantile(values, 0.9)),
        }

    return {
        "direct_paths": int(rows.numel()),
        "aod_vs_tx_to_rx_azimuth": summary(forward_aod_error),
        "aod_vs_rx_to_tx_azimuth": summary(reverse_aod_error),
        "aoa_vs_rx_to_tx_azimuth": summary(reverse_aoa_error),
        "aoa_vs_tx_to_rx_azimuth": summary(forward_aoa_error),
        "aod_vs_tx_to_rx_elevation": summary(aod_elevation_error),
        "aoa_vs_rx_to_tx_elevation": summary(aoa_elevation_error),
        "inferred_aod_convention": (
            "tx_to_rx"
            if float(forward_aod_error.mean()) <= float(reverse_aod_error.mean())
            else "rx_to_tx_or_rotated"
        ),
        "inferred_aoa_convention": (
            "rx_to_tx"
            if float(reverse_aoa_error.mean()) <= float(forward_aoa_error.mean())
            else "tx_to_rx_or_rotated"
        ),
    }


def _tx_rx_bearing_audit(payload: dict[str, torch.Tensor]) -> dict[str, float]:
    displacement = payload["rx_xyz_m"].float() - payload["tx_xyz_m"].float()
    horizontal = torch.linalg.vector_norm(displacement[:, :2], dim=-1)
    azimuth = torch.atan2(displacement[:, 1], displacement[:, 0])
    elevation = torch.rad2deg(torch.atan2(displacement[:, 2], horizontal))
    mean_sine = azimuth.sin().mean()
    mean_cosine = azimuth.cos().mean()
    return {
        "tx_to_rx_bearing_circular_mean_deg": float(
            torch.rad2deg(torch.atan2(mean_sine, mean_cosine))
        ),
        "tx_to_rx_bearing_resultant": float(
            torch.stack([mean_cosine, mean_sine]).norm()
        ),
        "tx_to_rx_elevation_mean_deg": float(elevation.mean()),
        "tx_to_rx_horizontal_distance_mean_m": float(horizontal.mean()),
    }


def _coordinate_system_audit(
    payloads: dict[int, dict[str, torch.Tensor]],
    system: SystemConfig,
) -> tuple[dict[str, Any], dict[int, float]]:
    per_map: dict[str, Any] = {}
    frame_yaws: dict[int, float] = {}
    for map_id, payload in sorted(payloads.items()):
        frame = _frame_from_receiver_coordinates(payload)
        frame_yaws[map_id] = frame["receiver_pca_frame_yaw_deg"]
        tx = payload["tx_xyz_m"].float()
        rx = payload["rx_xyz_m"].float()
        per_map[str(map_id)] = {
            "samples": int(rx.shape[0]),
            "unique_source_ids": int(torch.unique(payload["source_id"]).numel()),
            "unique_tx_positions": int(torch.unique(tx, dim=0).shape[0]),
            "tx_xyz_min_m": tx.amin(0).tolist(),
            "tx_xyz_max_m": tx.amax(0).tolist(),
            "rx_xyz_min_m": rx.amin(0).tolist(),
            "rx_xyz_max_m": rx.amax(0).tolist(),
            **frame,
            "tx_rx_bearing": _tx_rx_bearing_audit(payload),
            "direct_path_geometry": _direct_path_geometry_audit(payload),
        }
    yaws = torch.deg2rad(torch.tensor(list(frame_yaws.values()), dtype=torch.float64))
    resultant = torch.stack([yaws.cos().mean(), yaws.sin().mean()]).norm()
    geometry_keys = (
        "tx_xyz_m",
        "rx_xyz_m",
        "direct_path_mask",
        "aod_az_deg",
        "aod_el_deg",
        "aoa_az_deg",
        "aoa_el_deg",
    )
    combined_geometry = {
        key: torch.cat([payload[key] for payload in payloads.values()])
        for key in geometry_keys
    }
    return (
        {
            "coordinate_fields_present": ["tx_xyz_m", "rx_xyz_m"],
            "explicit_tx_array_orientation_metadata_present": False,
            "explicit_rx_array_orientation_metadata_present": False,
            "configured_tx_array_plane": system.tx_array_plane,
            "configured_rx_array_plane": system.rx_array_plane,
            "configured_array_interpretation": (
                "The synthesizer uses one fixed global array plane and no per-map or "
                "per-transmitter yaw/pitch/roll. Therefore array orientation cannot be "
                "verified from the prepared records."
            ),
            "angle_convention_under_test": (
                "Global azimuth measured in the x-y plane from +x; direct-path AoD/AoA "
                "are compared with geometric Tx-Rx bearings."
            ),
            "rotation_alignment_method": (
                "Subtract each map's receiver-coordinate PCA yaw from AoD and AoA, then "
                "resynthesize CSI. This is a coordinate sensitivity analysis, not an "
                "estimate of physical antenna orientation."
            ),
            "map_frame_yaw_circular_resultant": float(resultant),
            "overall_tx_rx_bearing": _tx_rx_bearing_audit(combined_geometry),
            "overall_direct_path_geometry": _direct_path_geometry_audit(
                combined_geometry
            ),
            "per_map": per_map,
        },
        frame_yaws,
    )


def _repeat_scene_metrics(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
    *,
    first_yaw: float,
    second_yaw: float,
    synthesizer: MIMOOFDMSynthesizer,
    projections: int,
) -> dict[str, float | None]:
    result = scene_structure_distances(first, second, projections=projections)
    result.update(
        rotation_aligned_scene_structure_distances(
            first,
            second,
            candidate_frame_yaw_deg=first_yaw,
            reference_frame_yaw_deg=second_yaw,
            synthesizer=synthesizer,
            projections=projections,
        )
    )
    return result


def _aggregate_reports(
    reports: list[dict[str, Any]], section: str
) -> dict[str, dict[str, float]]:
    names = sorted(
        {
            name
            for report in reports
            for name, value in report.get(section, {}).items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
    )
    summary: dict[str, dict[str, float]] = {}
    for name in names:
        values = torch.tensor(
            [
                report[section][name]
                for report in reports
                if isinstance(report.get(section, {}).get(name), (int, float))
                and math.isfinite(float(report[section][name]))
            ],
            dtype=torch.float64,
        )
        summary[name] = {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "clusters": int(values.numel()),
        }
    return summary


def _matching_summary(
    cross_reports: list[dict[str, Any]],
    within_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    audits = [
        repeat["matching"]
        for report in [*cross_reports, *within_reports]
        for repeat in report["repeats"]
    ]
    if not audits:
        return {}
    return {
        "comparisons": len(audits),
        "sample_count_values": sorted({audit["samples"] for audit in audits}),
        "first_unique_sample_count_values": sorted(
            {audit["first_unique_samples"] for audit in audits}
        ),
        "second_unique_sample_count_values": sorted(
            {audit["second_unique_samples"] for audit in audits}
        ),
        "maximum_first_duplicate_rate": max(
            audit["first_duplicate_rate"] for audit in audits
        ),
        "maximum_second_duplicate_rate": max(
            audit["second_duplicate_rate"] for audit in audits
        ),
        "all_los_pairs_exact": all(
            audit["los_exact_for_every_pair"] for audit in audits
        ),
        "all_k_validity_pairs_exact": all(
            audit["k_validity_exact_for_every_pair"] for audit in audits
        ),
        "condition_pair_distance_mean": float(
            torch.tensor(
                [audit["pair_distance_mean"] for audit in audits],
                dtype=torch.float64,
            ).mean()
        ),
        "condition_pair_distance_maximum": max(
            audit["pair_distance_maximum"] for audit in audits
        ),
    }


def _cluster_bootstrap_contrast(
    cross_reports: list[dict[str, Any]],
    within_reports: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Pigeonhole cluster bootstrap over Map IDs.

    A resampled Map multiplicity weights a within-Map observation once and a
    cross-Map pair by the product of the two endpoint multiplicities.  This
    preserves dependence among Map pairs that share a Map.
    """
    map_ids = sorted(
        {report["first_map"] for report in cross_reports}
        | {report["second_map"] for report in cross_reports}
        | {report["map"] for report in within_reports}
    )
    map_to_index = {map_id: index for index, map_id in enumerate(map_ids)}
    metric_names = sorted(
        set.intersection(
            {
                name
                for report in cross_reports
                for name, value in report["scene_structure"].items()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            },
            {
                name
                for report in within_reports
                for name, value in report["scene_structure"].items()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            },
        )
    )
    generator = torch.Generator().manual_seed(seed)
    multiplicities = torch.zeros(
        bootstrap_samples, len(map_ids), dtype=torch.float64
    )
    draws = torch.randint(
        len(map_ids),
        (bootstrap_samples, len(map_ids)),
        generator=generator,
    )
    multiplicities.scatter_add_(
        1, draws, torch.ones_like(draws, dtype=torch.float64)
    )
    result: dict[str, dict[str, Any]] = {}
    quantiles = torch.tensor([0.025, 0.975], dtype=torch.float64)
    for metric in metric_names:
        metric_cross_reports = [
            report
            for report in cross_reports
            if isinstance(report["scene_structure"].get(metric), (int, float))
            and math.isfinite(float(report["scene_structure"][metric]))
        ]
        metric_within_reports = [
            report
            for report in within_reports
            if isinstance(report["scene_structure"].get(metric), (int, float))
            and math.isfinite(float(report["scene_structure"][metric]))
        ]
        cross_values = torch.tensor(
            [report["scene_structure"][metric] for report in metric_cross_reports],
            dtype=torch.float64,
        )
        within_values = torch.tensor(
            [report["scene_structure"][metric] for report in metric_within_reports],
            dtype=torch.float64,
        )
        cross_first = torch.tensor(
            [map_to_index[report["first_map"]] for report in metric_cross_reports]
        )
        cross_second = torch.tensor(
            [map_to_index[report["second_map"]] for report in metric_cross_reports]
        )
        within_map = torch.tensor(
            [map_to_index[report["map"]] for report in metric_within_reports]
        )
        cross_weight = (
            multiplicities[:, cross_first] * multiplicities[:, cross_second]
        )
        within_weight = multiplicities[:, within_map]
        valid = (cross_weight.sum(1) > 0) & (within_weight.sum(1) > 0)
        cross_bootstrap = (
            (cross_weight[valid] * cross_values).sum(1)
            / cross_weight[valid].sum(1)
        )
        within_bootstrap = (
            (within_weight[valid] * within_values).sum(1)
            / within_weight[valid].sum(1)
        )
        difference = cross_bootstrap - within_bootstrap
        ratio = cross_bootstrap / within_bootstrap.clamp_min(1e-20)
        difference_interval = torch.quantile(difference, quantiles)
        ratio_interval = torch.quantile(ratio, quantiles)
        observed_cross = float(cross_values.mean())
        observed_within = float(within_values.mean())
        p_value = min(
            1.0,
            2.0
            * min(
                float((difference <= 0).double().mean()),
                float((difference >= 0).double().mean()),
            ),
        )
        result[metric] = {
            "between_map_mean": observed_cross,
            "within_map_null_mean": observed_within,
            "between_minus_within": observed_cross - observed_within,
            "between_to_within_ratio": (
                observed_cross / observed_within
                if abs(observed_within) > 1e-20
                else None
            ),
            "difference_cluster_bootstrap_95_percent_ci": difference_interval.tolist(),
            "ratio_cluster_bootstrap_95_percent_ci": ratio_interval.tolist(),
            "cluster_bootstrap_two_sided_p": p_value,
            "valid_bootstrap_replicates": int(valid.sum()),
            "evidence": (
                "positive_cross_map_residual"
                if float(difference_interval[0]) > 0
                else "negative"
                if float(difference_interval[1]) < 0
                else "inconclusive"
            ),
        }
    return result


def _coordinate_alignment_effect(
    contrast: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for global_metric, global_values in contrast.items():
        aligned_metric = f"rotation_aligned_{global_metric}"
        if aligned_metric not in contrast:
            continue
        aligned_values = contrast[aligned_metric]
        global_ratio = global_values["between_to_within_ratio"]
        aligned_ratio = aligned_values["between_to_within_ratio"]
        result[global_metric] = {
            "global_coordinate": global_values,
            "receiver_pca_aligned": aligned_values,
            "aligned_minus_global_between_within_ratio": (
                aligned_ratio - global_ratio
                if aligned_ratio is not None and global_ratio is not None
                else None
            ),
            "interpretation": (
                "A substantially smaller aligned ratio suggests that global map rotation "
                "or Tx/Rx layout orientation contributes to the apparent Map separation."
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strict real-Map separation audit with unique condition matching, repeated "
            "sampling, Map-cluster bootstrap, and coordinate-frame sensitivity metrics."
        )
    )
    parser.add_argument("--scene-plan", required=True)
    parser.add_argument("--prepared-root", default="artifacts/prepared_d2los")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--condition-std-json", default=None)
    parser.add_argument("--maximum-condition-distance", type=float, default=1.0)
    parser.add_argument("--samples-per-comparison", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--maximum-repeat-attempts", type=int, default=200)
    parser.add_argument("--maximum-map-pairs", type=int, default=100)
    parser.add_argument("--maximum-samples-per-map", type=int, default=256)
    parser.add_argument(
        "--map-pair-selection",
        choices=["random", "largest_support"],
        default="random",
    )
    parser.add_argument("--cluster-bootstrap-samples", type=int, default=2000)
    parser.add_argument("--diagnostic-projections", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every-map-pairs", type=int, default=100)
    parser.add_argument("--log-every-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.samples_per_comparison < 2:
        raise ValueError("--samples-per-comparison must be at least 2")
    if args.repeats < 20:
        raise ValueError("--repeats must be at least 20 for the strict audit")
    if args.maximum_repeat_attempts < args.repeats:
        raise ValueError("--maximum-repeat-attempts must be at least --repeats")
    if args.cluster_bootstrap_samples < 100:
        raise ValueError("--cluster-bootstrap-samples must be at least 100")
    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    torch.set_grad_enabled(False)
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
    payloads = {map_id: _stack(records) for map_id, records in records_by_map.items()}
    del records_by_map

    with (Path(args.prepared_root) / "manifest.json").open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    system = SystemConfig.from_mapping(manifest["system"])
    synthesizer = MIMOOFDMSynthesizer(system).to(device).eval()
    coordinate_audit, frame_yaws = _coordinate_system_audit(payloads, system)

    if args.condition_std_json:
        with Path(args.condition_std_json).open("r", encoding="utf-8") as stream:
            std = torch.tensor(json.load(stream)["condition_std"], dtype=torch.float32)
        std_source = str(Path(args.condition_std_json))
    else:
        all_conditions = torch.cat(
            [payload["conditions"].float() for payload in payloads.values()]
        )
        std = all_conditions.std(0, unbiased=False).clamp_min(1e-6)
        std_source = "current capped audit split"
    matching_std = std.to(device)
    matching_payloads = {
        map_id: {
            "conditions": payload["conditions"].to(device),
            "condition_validity": payload["condition_validity"].to(device),
        }
        for map_id, payload in payloads.items()
    }

    map_pairs = list(itertools.combinations(sorted(payloads), 2))
    candidates: list[tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for pair_index, (first_map, second_map) in enumerate(map_pairs, start=1):
        first_index, second_index, distances = _one_to_one_matched_indices(
            matching_payloads[first_map],
            matching_payloads[second_map],
            matching_std,
            args.maximum_condition_distance,
        )
        if first_index.numel() >= args.samples_per_comparison:
            candidates.append(
                (
                    int(first_index.numel()),
                    first_map,
                    second_map,
                    first_index,
                    second_index,
                    distances,
                )
            )
        if pair_index % max(args.log_every_map_pairs, 1) == 0 or pair_index == len(map_pairs):
            print(
                json.dumps(
                    {
                        "event": "unique_condition_matching",
                        "pairs_complete": pair_index,
                        "pairs_total": len(map_pairs),
                        "eligible_pairs": len(candidates),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if args.map_pair_selection == "largest_support":
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    else:
        order = torch.randperm(len(candidates), generator=generator).tolist()
        candidates = [candidates[index] for index in order]
    selected_pairs = candidates[: args.maximum_map_pairs]

    cross_reports: list[dict[str, Any]] = []
    for pair_number, (
        available_matches,
        first_map,
        second_map,
        first_pool,
        second_pool,
        distance_pool,
    ) in enumerate(selected_pairs, start=1):
        repeat_reports: list[dict[str, Any]] = []
        repeat_subsets: set[tuple[int, ...]] = set()
        for repeat in range(args.repeats):
            selection = torch.randperm(available_matches, generator=generator)[
                : args.samples_per_comparison
            ]
            repeat_subsets.add(tuple(sorted(int(value) for value in selection.tolist())))
            selected_first = first_pool[selection]
            selected_second = second_pool[selection]
            selected_distances = distance_pool[selection]
            first = _selected_payload(payloads[first_map], selected_first, device)
            second = _selected_payload(payloads[second_map], selected_second, device)
            metrics = _repeat_scene_metrics(
                first,
                second,
                first_yaw=frame_yaws[first_map],
                second_yaw=frame_yaws[second_map],
                synthesizer=synthesizer,
                projections=args.diagnostic_projections,
            )
            repeat_reports.append(
                {
                    "repeat": repeat,
                    "scene_structure": metrics,
                    "matching": _condition_audit(
                        first, second, selected_distances, std
                    ),
                }
            )
            del first, second
            if (repeat + 1) % max(args.log_every_repeats, 1) == 0 or repeat + 1 == args.repeats:
                print(
                    json.dumps(
                        {
                            "event": "cross_map_repeat_progress",
                            "pair": pair_number,
                            "pairs_total": len(selected_pairs),
                            "first_map": first_map,
                            "second_map": second_map,
                            "repeats_complete": repeat + 1,
                            "repeats_total": args.repeats,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        cross_reports.append(
            {
                "first_map": first_map,
                "second_map": second_map,
                "available_unique_matches": available_matches,
                "matched_samples": args.samples_per_comparison,
                "successful_repeats": len(repeat_reports),
                "unique_repeat_subsets": len(repeat_subsets),
                "scene_structure": _mean_metrics(repeat_reports, "scene_structure"),
                "repeats": repeat_reports,
            }
        )
        print(
            json.dumps(
                {
                    "event": "cross_map_repeats_complete",
                    "pair": pair_number,
                    "pairs_total": len(selected_pairs),
                    "first_map": first_map,
                    "second_map": second_map,
                    "repeats": len(repeat_reports),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    within_reports: list[dict[str, Any]] = []
    excluded_within_maps: list[dict[str, Any]] = []
    for map_number, map_id in enumerate(sorted(payloads), start=1):
        payload = payloads[map_id]
        sample_count = int(payload["conditions"].shape[0])
        if sample_count < 2 * args.samples_per_comparison:
            excluded_within_maps.append(
                {"map": map_id, "reason": "fewer than two disjoint sample groups"}
            )
            continue
        repeat_reports = []
        repeat_subsets: set[tuple[int, ...]] = set()
        attempts = 0
        while len(repeat_reports) < args.repeats and attempts < args.maximum_repeat_attempts:
            attempts += 1
            permutation = torch.randperm(sample_count, generator=generator)
            midpoint = sample_count // 2
            first_half_indices = permutation[:midpoint]
            second_half_indices = permutation[midpoint:]
            first_half_matching = {
                "conditions": payload["conditions"][first_half_indices].to(device),
                "condition_validity": payload["condition_validity"][first_half_indices].to(device),
            }
            second_half_matching = {
                "conditions": payload["conditions"][second_half_indices].to(device),
                "condition_validity": payload["condition_validity"][second_half_indices].to(device),
            }
            first_local, second_local, distances = _one_to_one_matched_indices(
                first_half_matching,
                second_half_matching,
                matching_std,
                args.maximum_condition_distance,
            )
            available_matches = int(first_local.numel())
            if available_matches < args.samples_per_comparison:
                continue
            selection = torch.randperm(available_matches, generator=generator)[
                : args.samples_per_comparison
            ]
            selected_first = first_half_indices[first_local[selection]]
            selected_second = second_half_indices[second_local[selection]]
            repeat_subsets.add(
                tuple(sorted(int(value) for value in selected_first.tolist()))
            )
            selected_distances = distances[selection]
            first = _selected_payload(payload, selected_first, device)
            second = _selected_payload(payload, selected_second, device)
            repeat_reports.append(
                {
                    "repeat": len(repeat_reports),
                    "attempt": attempts,
                    "available_unique_matches": available_matches,
                    "scene_structure": _repeat_scene_metrics(
                        first,
                        second,
                        first_yaw=frame_yaws[map_id],
                        second_yaw=frame_yaws[map_id],
                        synthesizer=synthesizer,
                        projections=args.diagnostic_projections,
                    ),
                    "matching": _condition_audit(
                        first, second, selected_distances, std
                    ),
                }
            )
            del first, second
            if len(repeat_reports) % max(args.log_every_repeats, 1) == 0:
                print(
                    json.dumps(
                        {
                            "event": "within_map_repeat_progress",
                            "map": map_id,
                            "map_index": map_number,
                            "maps_total": len(payloads),
                            "repeats_complete": len(repeat_reports),
                            "repeats_total": args.repeats,
                            "attempts": attempts,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if len(repeat_reports) < args.repeats:
            excluded_within_maps.append(
                {
                    "map": map_id,
                    "reason": "insufficient successful unique matched splits",
                    "successful_repeats": len(repeat_reports),
                    "attempts": attempts,
                }
            )
            continue
        within_reports.append(
            {
                "map": map_id,
                "matched_samples": args.samples_per_comparison,
                "successful_repeats": len(repeat_reports),
                "unique_repeat_subsets": len(repeat_subsets),
                "attempts": attempts,
                "scene_structure": _mean_metrics(repeat_reports, "scene_structure"),
                "repeats": repeat_reports,
            }
        )
        print(
            json.dumps(
                {
                    "event": "within_map_repeats_complete",
                    "map": map_id,
                    "map_index": map_number,
                    "maps_total": len(payloads),
                    "repeats": len(repeat_reports),
                    "attempts": attempts,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    cross_summary = _aggregate_reports(cross_reports, "scene_structure")
    within_summary = _aggregate_reports(within_reports, "scene_structure")
    contrast = _cluster_bootstrap_contrast(
        cross_reports,
        within_reports,
        bootstrap_samples=args.cluster_bootstrap_samples,
        seed=args.seed + 7919,
    )
    report = {
        "format_version": 2,
        "purpose": (
            "Strict real-map residual-structure audit after one-to-one physical-condition "
            "matching, repeated sampling, Map-cluster bootstrap, and coordinate alignment."
        ),
        "scene_plan_fingerprint": plan["fingerprint"],
        "split": args.split,
        "device": str(device),
        "maps": len(payloads),
        "all_map_pairs": len(map_pairs),
        "eligible_unique_matched_map_pairs": len(candidates),
        "reported_map_pairs": len(cross_reports),
        "reported_within_maps": len(within_reports),
        "excluded_within_maps": excluded_within_maps,
        "matching": {
            "method": "greedy one-to-one nearest available neighbor without replacement",
            "maximum_standardized_condition_distance": args.maximum_condition_distance,
            "samples_per_comparison": args.samples_per_comparison,
            "repeats": args.repeats,
            "maximum_samples_per_map": args.maximum_samples_per_map,
            "map_pair_selection": args.map_pair_selection,
            "condition_std": std.tolist(),
            "condition_std_source": std_source,
            "los_matched_exactly": True,
            "k_factor_validity_matched_exactly": True,
            "second_side_reuse_allowed": False,
            "observed_audit": _matching_summary(cross_reports, within_reports),
        },
        "inference": {
            "method": "Map-cluster pigeonhole bootstrap",
            "bootstrap_samples": args.cluster_bootstrap_samples,
            "cluster_unit": "Map ID",
            "shared_map_pair_dependence_accounted_for": True,
            "note": (
                "Cross-pair weights are products of endpoint Map multiplicities; within-Map "
                "weights use the corresponding Map multiplicity."
            ),
        },
        "coordinate_system_audit": coordinate_audit,
        "scene_structure_summary": cross_summary,
        "within_map_scene_structure_null": within_summary,
        "between_vs_within_scene_structure": contrast,
        "global_vs_rotation_aligned": _coordinate_alignment_effect(contrast),
        "map_pairs": cross_reports,
        "within_map_nulls": within_reports,
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "audit_complete",
                "output": str(Path(args.output).resolve()),
                "reported_map_pairs": len(cross_reports),
                "reported_within_maps": len(within_reports),
                "metrics": len(contrast),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
