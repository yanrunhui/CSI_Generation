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

from physics_csi.physics.aod_control import (
    AoDControlProjector,
    circular_difference_degrees,
    select_nearest_aod_candidates,
)
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump
from scripts.diagnose_path_count_control import (
    _fidelity_metrics,
    _power_aod_path,
    _real_real_floor,
    _standardized_swd,
    _tx_spatial_relative_frobenius,
)


ROTATION_BINS = (
    ("0_to_15", 0.0, 15.0),
    ("15_to_45", 15.0, 45.0),
    ("45_to_90", 45.0, 90.0),
    ("90_to_180", 90.0, 180.0001),
)


def _float_list(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not result or any(not math.isfinite(item) for item in result):
        raise argparse.ArgumentTypeError("the list must contain finite numbers")
    return result


def _int_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("the list cannot be empty")
    return result


def _validate_artifact(artifact: Mapping[str, Any]) -> None:
    if artifact.get("model_kind") != "b3_plus_dp_plus_aod_residual":
        raise ValueError("artifact is not a B3 + DP + AoD residual evaluation")
    if artifact.get("aod_architecture") != "link_global":
        raise ValueError("artifact must use the Link-global AoD architecture")
    for section in ("reference", "correct_map"):
        if section not in artifact:
            raise ValueError(f"artifact is missing {section}")
        for key in (
            "csi",
            "path_mask",
            "path_delays_ns",
            "path_powers_linear",
            "path_phases_rad",
            "aoa_az_deg",
            "aoa_el_deg",
            "aod_az_deg",
            "aod_el_deg",
        ):
            if key not in artifact[section]:
                raise ValueError(f"artifact is missing {section}.{key}")
    for key in ("map_id", "conditions", "condition_validity"):
        if key not in artifact["reference"]:
            raise ValueError(f"artifact is missing reference.{key}")


def _resolve_system(
    artifact: Mapping[str, Any],
    *,
    link_checkpoint_path: str | None,
    cache_path: str | None,
) -> tuple[SystemConfig, str]:
    candidates: list[tuple[str, Any]] = [("artifact.system", artifact.get("system"))]
    if link_checkpoint_path:
        checkpoint = torch.load(
            link_checkpoint_path, map_location="cpu", weights_only=False
        )
        if checkpoint.get("model_kind") != "scene_link_global_aod_residual":
            raise ValueError("--link-checkpoint is not a Link-global AoD residual")
        candidates.append(("link_checkpoint.system", checkpoint.get("system")))
    if cache_path:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache.get("kind") != "b3_dp_residual_cache":
            raise ValueError("--cache is not a B3 DP-residual cache")
        candidates.append(("cache.system", cache.get("system")))
    for source, values in candidates:
        if values is not None:
            system = SystemConfig.from_mapping(values)
            csi = artifact["reference"]["csi"]
            expected = (
                system.rx_array_shape[0] * system.rx_array_shape[1],
                system.tx_array_shape[0] * system.tx_array_shape[1],
                system.num_subcarriers,
            )
            actual = tuple(int(value) for value in csi.shape[1:])
            if actual != expected:
                raise ValueError(
                    f"{source} expects CSI [Rx,Tx,F]={expected}, but artifact uses {actual}"
                )
            return system, source
    raise ValueError(
        "No synthesizer system configuration is available. Supply "
        "--link-checkpoint or --cache for a legacy artifact without artifact.system."
    )


def _subset(
    payload: Mapping[str, Any], indices: torch.Tensor, device: torch.device
) -> dict[str, torch.Tensor]:
    count = int(payload["path_mask"].shape[0])
    return {
        key: value[indices].to(device)
        for key, value in payload.items()
        if isinstance(value, torch.Tensor) and value.shape[:1] == (count,)
    }


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


def _angle_audit(
    payload: Mapping[str, torch.Tensor],
    projector: AoDControlProjector,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    measured = projector(payload, 0.0, mode=mode)  # rotation is not applied here
    audit = {
        "mean_concentration_mean": float(
            measured.mean_resultant_concentration.mean()
        ),
        "mean_valid_rate": float(measured.control_valid.float().mean()),
        "strongest_near_tie_rate": float(measured.strongest_near_tie.float().mean()),
    }
    return measured.reference_before_deg, measured.control_valid, audit


def _cap(indices: torch.Tensor, maximum: int, seed: int) -> torch.Tensor:
    if indices.numel() <= maximum:
        return indices
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(indices.numel(), generator=generator)[:maximum]
    return indices[order]


def _condition_features(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    values = payload["conditions"][:, :3].float().clone()
    valid_k = payload["condition_validity"][:, 2].bool()
    values[:, 2] = torch.where(valid_k, values[:, 2], torch.zeros_like(values[:, 2]))
    return values


def _greedy_condition_match(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    maximum: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    first = _condition_features(reference).cpu()
    second = _condition_features(candidate).cpu()
    combined = torch.cat([first, second], dim=0)
    scale = combined.std(0, unbiased=False).clamp_min(1e-6)
    distance = torch.cdist(first / scale, second / scale)
    matches = min(maximum, first.shape[0], second.shape[0])
    first_selected: list[int] = []
    second_selected: list[int] = []
    selected_distance: list[float] = []
    for _ in range(matches):
        flat = int(distance.argmin())
        row = flat // distance.shape[1]
        column = flat % distance.shape[1]
        value = float(distance[row, column])
        if not math.isfinite(value):
            break
        first_selected.append(row)
        second_selected.append(column)
        selected_distance.append(value)
        distance[row, :] = torch.inf
        distance[:, column] = torch.inf
    distances = torch.tensor(selected_distance)
    audit = {
        "pairs": len(selected_distance),
        "standardized_distance_mean": float(distances.mean()) if distances.numel() else float("nan"),
        "standardized_distance_p95": (
            float(torch.quantile(distances, 0.95)) if distances.numel() else float("nan")
        ),
        "standardized_distance_maximum": (
            float(distances.max()) if distances.numel() else float("nan")
        ),
    }
    return (
        torch.tensor(first_selected, dtype=torch.long),
        torch.tensor(second_selected, dtype=torch.long),
        audit,
    )


def _beamforming_gain(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    # With one receive antenna, maximum-ratio transmit beamforming has gain ||h||^2.
    csi = payload["csi"]
    return 10.0 * torch.log10(
        csi.abs().square().sum(dim=2).mean(dim=(1, 2)).clamp_min(1e-20)
    ).unsqueeze(-1)


def _metrics(
    candidate: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    *,
    projections: int,
    projection_dimension: int,
    projection_seeds: tuple[int, ...],
    seed: int,
) -> dict[str, float]:
    result = _fidelity_metrics(
        candidate,
        reference,
        swd_projections=projections,
        projection_dimension=projection_dimension,
        projection_seeds=projection_seeds,
        seed=seed,
    )
    result["beamforming_gain_swd"] = _standardized_swd(
        _beamforming_gain(candidate),
        _beamforming_gain(reference),
        projections=min(projections, 64),
        seed=seed + 71,
    )
    return result


def _rotation_bins(
    candidate: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    rotation: torch.Tensor,
    *,
    minimum_samples: int,
    projections: int,
    seed: int,
) -> dict[str, Any]:
    absolute = rotation.abs()
    result: dict[str, Any] = {}
    for number, (name, lower, upper) in enumerate(ROTATION_BINS):
        selected = torch.nonzero(
            (absolute >= lower) & (absolute < upper), as_tuple=False
        ).flatten()
        entry: dict[str, Any] = {"samples": int(selected.numel())}
        if selected.numel() >= minimum_samples:
            current = {key: value[selected] for key, value in candidate.items()}
            real = {key: value[selected] for key, value in reference.items()}
            entry["path_power_aod_swd"] = _standardized_swd(
                _power_aod_path(current),
                _power_aod_path(real),
                projections=projections,
                seed=seed + number,
            )
            entry["tx_spatial_correlation_relative_frobenius"] = (
                _tx_spatial_relative_frobenius(current["csi"], real["csi"])
            )
            entry["beamforming_gain_swd"] = _standardized_swd(
                _beamforming_gain(current),
                _beamforming_gain(real),
                projections=min(projections, 64),
                seed=seed + 100 + number,
            )
        else:
            entry["status"] = "insufficient_samples"
        result[name] = entry
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Angle-matched RayVerse fidelity test for dual-mode AoD control."
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--link-checkpoint",
        default=None,
        help="Fallback source of system configuration for legacy artifacts.",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="Second fallback source of system configuration.",
    )
    parser.add_argument("--map-ids", default="10", help="Comma-separated IDs or 'all'.")
    parser.add_argument("--path-counts", type=_int_list, default=_int_list("5"))
    parser.add_argument(
        "--target-angles",
        type=_float_list,
        default=_float_list("-150,-90,-30,30,90,150"),
    )
    parser.add_argument("--angle-half-width-deg", type=float, default=15.0)
    parser.add_argument("--minimum-group-samples", type=int, default=24)
    parser.add_argument("--maximum-group-samples", type=int, default=128)
    parser.add_argument("--minimum-rotation-bin-samples", type=int, default=12)
    parser.add_argument("--minimum-mean-concentration", type=float, default=0.10)
    parser.add_argument("--strongest-near-tie-margin", type=float, default=0.02)
    parser.add_argument("--small-projection-threshold-deg", type=float, default=45.0)
    parser.add_argument("--swd-projections", type=int, default=128)
    parser.add_argument("--projection-dimension", type=int, default=64)
    parser.add_argument("--projection-seeds", type=_int_list, default=_int_list("0,1,2"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.angle_half_width_deg <= 0 or args.angle_half_width_deg > 180:
        parser.error("angle-half-width-deg must lie in (0, 180]")
    if args.minimum_group_samples < 8:
        parser.error("minimum-group-samples must be at least 8")
    if args.maximum_group_samples < args.minimum_group_samples:
        parser.error("maximum-group-samples cannot be smaller than the minimum")
    if not 0.0 < args.small_projection_threshold_deg <= 180.0:
        parser.error("small-projection-threshold-deg must lie in (0, 180]")

    device = resolve_device(args.device)
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    _validate_artifact(artifact)
    system, system_source = _resolve_system(
        artifact,
        link_checkpoint_path=args.link_checkpoint,
        cache_path=args.cache,
    )
    reference_cpu = artifact["reference"]
    natural_cpu = artifact["correct_map"]
    available_maps = sorted(int(value) for value in torch.unique(reference_cpu["map_id"]))
    map_ids = (
        available_maps
        if args.map_ids.strip().lower() == "all"
        else list(_int_list(args.map_ids))
    )
    missing_maps = sorted(set(map_ids) - set(available_maps))
    if missing_maps:
        raise ValueError(f"artifact has no requested Maps: {missing_maps}")
    projector = AoDControlProjector(
        minimum_mean_concentration=args.minimum_mean_concentration,
        strongest_near_tie_margin=args.strongest_near_tie_margin,
    )
    synthesizer = MIMOOFDMSynthesizer(system).to(device).eval()
    path_count = reference_cpu["path_mask"].bool().sum(-1)
    los = reference_cpu["conditions"][:, 3] >= 0.5
    k_valid = reference_cpu["condition_validity"][:, 2].bool()
    groups: list[dict[str, Any]] = []
    with torch.inference_mode():
        for map_id in map_ids:
            for requested_l in args.path_counts:
                for los_value in (False, True):
                    for k_valid_value in (False, True):
                        stratum = torch.nonzero(
                            (reference_cpu["map_id"] == map_id)
                            & (path_count == requested_l)
                            & (los == los_value)
                            & (k_valid == k_valid_value),
                            as_tuple=False,
                        ).flatten()
                        if stratum.numel() < args.minimum_group_samples:
                            continue
                        real_stratum = _subset(reference_cpu, stratum, device)
                        natural_stratum = _subset(natural_cpu, stratum, device)
                        # Generated artifact sections are index-aligned to the reference,
                        # but older artifacts store target metadata only in reference.
                        natural_stratum["conditions"] = real_stratum["conditions"]
                        natural_stratum["condition_validity"] = real_stratum[
                            "condition_validity"
                        ]
                        natural_stratum["map_id"] = real_stratum["map_id"]
                        for mode_number, mode in enumerate(("mean", "strongest")):
                            real_angle, real_valid, real_audit = _angle_audit(
                                real_stratum, projector, mode
                            )
                            natural_angle, natural_valid, natural_audit = _angle_audit(
                                natural_stratum, projector, mode
                            )
                            for target_number, target in enumerate(args.target_angles):
                                seed = (
                                    args.seed
                                    + map_id * 100003
                                    + requested_l * 1009
                                    + int(los_value) * 101
                                    + mode_number * 31
                                    + target_number
                                )
                                real_local = torch.nonzero(
                                    real_valid
                                    & (
                                        circular_difference_degrees(real_angle, torch.tensor(target, device=device)).abs()
                                        <= args.angle_half_width_deg
                                    ),
                                    as_tuple=False,
                                ).flatten().cpu()
                                natural_local = torch.nonzero(
                                    natural_valid
                                    & (
                                        circular_difference_degrees(natural_angle, torch.tensor(target, device=device)).abs()
                                        <= args.angle_half_width_deg
                                    ),
                                    as_tuple=False,
                                ).flatten().cpu()
                                entry: dict[str, Any] = {
                                    "map_id": map_id,
                                    "path_count": requested_l,
                                    "los_status": int(los_value),
                                    "k_factor_valid": k_valid_value,
                                    "mode": mode,
                                    "target_az_deg": target,
                                    "angle_half_width_deg": args.angle_half_width_deg,
                                    "stratum_samples": int(stratum.numel()),
                                    "real_angle_support": int(real_local.numel()),
                                    "natural_angle_support": int(natural_local.numel()),
                                    "real_angle_audit": real_audit,
                                    "natural_angle_audit": natural_audit,
                                }
                                if real_local.numel() < args.minimum_group_samples:
                                    entry["status"] = "insufficient_real_angle_support"
                                    groups.append(entry)
                                    continue
                                # Projection evaluation is exactly condition-paired: the candidate
                                # and RayVerse reference use the same artifact indices. Mean-mode
                                # samples must have a stable direction in both distributions.
                                projection_candidates = real_local[
                                    natural_valid[
                                        real_local.to(natural_valid.device)
                                    ].cpu()
                                ]
                                entry["projection_valid_support"] = int(
                                    projection_candidates.numel()
                                )
                                if (
                                    projection_candidates.numel()
                                    < args.minimum_group_samples
                                ):
                                    entry["status"] = (
                                        "insufficient_joint_control_valid_support"
                                    )
                                    groups.append(entry)
                                    continue
                                projection_local = _cap(
                                    projection_candidates,
                                    args.maximum_group_samples,
                                    seed,
                                )
                                real_projection = {
                                    key: value[projection_local.to(device)]
                                    for key, value in real_stratum.items()
                                }
                                natural_projection = {
                                    key: value[projection_local.to(device)]
                                    for key, value in natural_stratum.items()
                                }
                                control = projector(natural_projection, target, mode=mode)
                                projected = projector.apply_to_payload(
                                    natural_projection, control
                                )
                                projected["csi"] = _synthesize(projected, synthesizer)
                                entry["status"] = "ok"
                                entry["projection_samples"] = int(projection_local.numel())
                                entry["projection_control_valid_rate"] = float(
                                    control.control_valid.float().mean()
                                )
                                entry["projection_control_error_maximum_deg"] = float(
                                    control.circular_error_deg[control.control_valid].max()
                                ) if bool(control.control_valid.any()) else None
                                entry["projection_rotation_absolute_deg"] = {
                                    "mean": float(control.rotation_deg.abs().mean()),
                                    "p95": float(torch.quantile(control.rotation_deg.abs(), 0.95)),
                                    "maximum": float(control.rotation_deg.abs().max()),
                                }
                                entry["projection_vs_angle_matched_real"] = _metrics(
                                    projected,
                                    real_projection,
                                    projections=args.swd_projections,
                                    projection_dimension=args.projection_dimension,
                                    projection_seeds=args.projection_seeds,
                                    seed=seed,
                                )
                                entry["paired_unprojected_vs_angle_matched_real"] = _metrics(
                                    natural_projection,
                                    real_projection,
                                    projections=args.swd_projections,
                                    projection_dimension=args.projection_dimension,
                                    projection_seeds=args.projection_seeds,
                                    seed=seed,
                                )
                                entry["real_vs_real"] = _real_real_floor(
                                    real_projection,
                                    swd_projections=args.swd_projections,
                                    projection_dimension=args.projection_dimension,
                                    projection_seeds=args.projection_seeds,
                                    seed=seed + 7,
                                )
                                entry["projection_by_rotation_magnitude"] = _rotation_bins(
                                    projected,
                                    real_projection,
                                    control.rotation_deg,
                                    minimum_samples=args.minimum_rotation_bin_samples,
                                    projections=args.swd_projections,
                                    seed=seed + 13,
                                )
                                # Resampling uses unique natural links from the complete
                                # Map/L/LoS/K-validity stratum, ranks them by required
                                # circular rotation, then condition-matches them to the
                                # angle-matched real set before applying the residual turn.
                                resampling_samples = min(
                                    args.maximum_group_samples,
                                    int(real_local.numel()),
                                    int(natural_valid.sum()),
                                )
                                if resampling_samples >= args.minimum_group_samples:
                                    selection = select_nearest_aod_candidates(
                                        natural_stratum,
                                        target,
                                        samples=resampling_samples,
                                        mode=mode,
                                        projector=projector,
                                    )
                                    resampled_candidates = {
                                        key: value[selection.indices.to(value.device)]
                                        for key, value in natural_stratum.items()
                                    }
                                    real_candidates_for_resampling = {
                                        key: value[real_local.to(device)]
                                        for key, value in real_stratum.items()
                                    }
                                    real_match, resampled_match, resampling_match_audit = (
                                        _greedy_condition_match(
                                            real_candidates_for_resampling,
                                            resampled_candidates,
                                            resampling_samples,
                                        )
                                    )
                                    if real_match.numel() >= args.minimum_group_samples:
                                        resampling_real = {
                                            key: value[real_match.to(device)]
                                            for key, value in real_candidates_for_resampling.items()
                                        }
                                        resampled_only = {
                                            key: value[resampled_match.to(device)]
                                            for key, value in resampled_candidates.items()
                                        }
                                        resampled_control = projector(
                                            resampled_only, target, mode=mode
                                        )
                                        resampled_projected = projector.apply_to_payload(
                                            resampled_only, resampled_control
                                        )
                                        resampled_projected["csi"] = _synthesize(
                                            resampled_projected, synthesizer
                                        )
                                        selected_rotation = resampled_control.rotation_deg.abs()
                                        entry["resampling_selection_audit"] = {
                                            "candidate_pool_samples": selection.candidate_samples,
                                            "eligible_pool_samples": selection.eligible_samples,
                                            "selected_unique_samples": int(
                                                torch.unique(selection.indices).numel()
                                            ),
                                            "condition_matched_samples": int(real_match.numel()),
                                            "pre_match_selected_distance_mean_deg": float(
                                                selection.angular_distance_deg.mean()
                                            ),
                                            "pre_match_selected_distance_p95_deg": float(
                                                torch.quantile(
                                                    selection.angular_distance_deg, 0.95
                                                )
                                            ),
                                            "condition_match": resampling_match_audit,
                                        }
                                        entry[
                                            "resampling_small_projection_rotation_absolute_deg"
                                        ] = {
                                            "mean": float(selected_rotation.mean()),
                                            "p95": float(
                                                torch.quantile(selected_rotation, 0.95)
                                            ),
                                            "maximum": float(selected_rotation.max()),
                                            "threshold": args.small_projection_threshold_deg,
                                            "fraction_within_threshold": float(
                                                (
                                                    selected_rotation
                                                    <= args.small_projection_threshold_deg
                                                ).float().mean()
                                            ),
                                        }
                                        entry[
                                            "resampled_only_vs_angle_matched_real"
                                        ] = _metrics(
                                            resampled_only,
                                            resampling_real,
                                            projections=args.swd_projections,
                                            projection_dimension=args.projection_dimension,
                                            projection_seeds=args.projection_seeds,
                                            seed=seed + 23,
                                        )
                                        entry[
                                            "resampling_small_projection_vs_angle_matched_real"
                                        ] = _metrics(
                                            resampled_projected,
                                            resampling_real,
                                            projections=args.swd_projections,
                                            projection_dimension=args.projection_dimension,
                                            projection_seeds=args.projection_seeds,
                                            seed=seed + 23,
                                        )
                                        entry["resampling_real_vs_real"] = _real_real_floor(
                                            resampling_real,
                                            swd_projections=args.swd_projections,
                                            projection_dimension=args.projection_dimension,
                                            projection_seeds=args.projection_seeds,
                                            seed=seed + 29,
                                        )
                                        entry[
                                            "resampling_projection_by_rotation_magnitude"
                                        ] = _rotation_bins(
                                            resampled_projected,
                                            resampling_real,
                                            resampled_control.rotation_deg,
                                            minimum_samples=args.minimum_rotation_bin_samples,
                                            projections=args.swd_projections,
                                            seed=seed + 31,
                                        )
                                if natural_local.numel() >= args.minimum_group_samples:
                                    real_candidates = {
                                        key: value[real_local.to(device)]
                                        for key, value in real_stratum.items()
                                    }
                                    natural_candidates = {
                                        key: value[natural_local.to(device)]
                                        for key, value in natural_stratum.items()
                                    }
                                    real_match, natural_match, match_audit = (
                                        _greedy_condition_match(
                                            real_candidates,
                                            natural_candidates,
                                            args.maximum_group_samples,
                                        )
                                    )
                                    if real_match.numel() >= args.minimum_group_samples:
                                        matched_real = {
                                            key: value[real_match.to(device)]
                                            for key, value in real_candidates.items()
                                        }
                                        matched_natural = {
                                            key: value[natural_match.to(device)]
                                            for key, value in natural_candidates.items()
                                        }
                                        entry["natural_condition_match_audit"] = match_audit
                                        entry["natural_filtered_vs_angle_matched_real"] = _metrics(
                                            matched_natural,
                                            matched_real,
                                            projections=args.swd_projections,
                                            projection_dimension=args.projection_dimension,
                                            projection_seeds=args.projection_seeds,
                                            seed=seed + 19,
                                        )
                                groups.append(entry)
                                print(
                                    json.dumps(
                                        {
                                            "event": "dual_mode_aod_fidelity_group_complete",
                                            "map_id": map_id,
                                            "L": requested_l,
                                            "mode": mode,
                                            "target": target,
                                            "samples": int(projection_local.numel()),
                                        },
                                        sort_keys=True,
                                    ),
                                    flush=True,
                                )
    completed = [item for item in groups if item.get("status") == "ok"]
    report = {
        "format_version": 1,
        "experiment": "dual_mode_aod_angle_matched_fidelity",
        "artifact": str(Path(args.artifact).resolve()),
        "system_configuration_source": system_source,
        "system": system.as_dict(),
        "split": artifact.get("split"),
        "maps_requested": map_ids,
        "path_counts_requested": list(args.path_counts),
        "target_angles_deg": list(args.target_angles),
        "angle_half_width_deg": args.angle_half_width_deg,
        "groups_completed": len(completed),
        "groups": groups,
        "metric_direction": "lower_is_better_except_diversity_ratios",
        "interpretation": [
            "Mean and Strongest groups use separate angle-matched RayVerse references.",
            "Projection comparisons are index-paired in Map, L, LoS, K validity and all four physical target conditions.",
            "Paired unprojected and direct-projection rows use the same generated links and real references, isolating the direct rotation effect.",
            "Natural filtering is matched greedily in standardized physical-condition space and reports the residual distance.",
            "Resampling selects unique nearest-angle links without replacement, condition-matches them to real references, and evaluates before and after the remaining small projection.",
            "No result with insufficient real angle support establishes Map-conditional realism.",
            "Rotation bins diagnose, but do not by themselves define, out-of-distribution control.",
        ],
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "dual_mode_aod_fidelity_complete",
                "output": str(Path(args.output).resolve()),
                "groups_completed": len(completed),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
