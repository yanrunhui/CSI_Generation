#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.models.diffusion.multipath_diffusion import MultipathDiffusionModel
from physics_csi.models.scene_delay_power_residual import SceneConditionedDelayPowerResidual
from physics_csi.models.scene_link_global_aod_residual import (
    SceneConditionedLinkGlobalAoDResidual,
)
from physics_csi.physics.aod_control import (
    AoDControlProjector,
    select_aod_candidates_within_tolerance,
    select_nearest_aod_candidates,
    select_payload_samples,
)
from physics_csi.physics.constraints import check_condition_feasibility
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    return value


def _append_payload(
    chunks: dict[str, list[torch.Tensor]], payload: Mapping[str, torch.Tensor]
) -> None:
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            chunks.setdefault(key, []).append(value.detach().cpu())


def _finish_payload(
    chunks: Mapping[str, list[torch.Tensor]],
) -> dict[str, torch.Tensor]:
    return {key: torch.cat(values, dim=0) for key, values in chunks.items()}


def _to_device(
    payload: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in payload.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _map_id(value: str) -> int:
    normalized = value.strip().lower()
    if normalized.startswith("map_"):
        normalized = normalized[4:]
    try:
        return int(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Map ID must look like 10 or map_10") from error


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


def _audit(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().flatten()
    return {
        "mean": float(values.mean()),
        "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
        "p99": float(torch.quantile(values, 0.99)),
        "maximum": float(values.max()),
    }


def _load_models(
    b3_path: Path,
    dp_path: Path,
    link_path: Path,
    device: torch.device,
) -> tuple[
    MultipathDiffusionModel,
    SceneConditionedDelayPowerResidual,
    SceneConditionedLinkGlobalAoDResidual,
    Mapping[str, Any],
]:
    b3_checkpoint = torch.load(b3_path, map_location="cpu", weights_only=False)
    dp_checkpoint = torch.load(dp_path, map_location="cpu", weights_only=False)
    link_checkpoint = torch.load(link_path, map_location="cpu", weights_only=False)
    if dp_checkpoint.get("model_kind") != "scene_delay_power_residual":
        raise ValueError("--dp-checkpoint is not a DP residual")
    if link_checkpoint.get("model_kind") != "scene_link_global_aod_residual":
        raise ValueError("--link-checkpoint is not a Link-global AoD residual")
    fingerprint = _sha256(b3_path)
    for checkpoint in (dp_checkpoint, link_checkpoint):
        if checkpoint["cache_contract"]["b3_checkpoint_sha256"] != fingerprint:
            raise ValueError("residual checkpoint was trained with a different B3 backbone")
    config = b3_checkpoint["config"]
    system = SystemConfig.from_mapping(config["data"]["system"])
    b3 = MultipathDiffusionModel(
        config["multipath_diffusion"], system, b3_checkpoint["statistics"]
    ).to(device)
    if b3.scene_aware_enabled:
        raise ValueError("dual-mode pipeline requires the Map-independent B3 backbone")
    b3.load_state_dict(
        b3_checkpoint.get("ema_state_dict", b3_checkpoint["model_state_dict"])
    )
    b3.requires_grad_(False).eval()
    dp = SceneConditionedDelayPowerResidual(**dp_checkpoint["model_init"]).to(device)
    dp.load_state_dict(dp_checkpoint.get("ema_state_dict", dp_checkpoint["model_state_dict"]))
    dp.requires_grad_(False).eval()
    link = SceneConditionedLinkGlobalAoDResidual(**link_checkpoint["model_init"]).to(device)
    link.load_state_dict(
        link_checkpoint.get("ema_state_dict", link_checkpoint["model_state_dict"])
    )
    link.requires_grad_(False).eval()
    if dp.map_id_to_index != link.map_id_to_index:
        raise ValueError("DP and Link-global checkpoints use different Map mappings")
    return b3, dp, link, b3_checkpoint


def _support_audit(
    cache_path: str | None,
    *,
    map_id: int,
    conditions: torch.Tensor,
    validity: torch.Tensor,
    path_count: int,
) -> dict[str, Any] | None:
    if cache_path is None:
        return None
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if "train" not in cache["splits"]:
        return {"status": "unavailable", "reason": "cache has no train split"}
    reference = cache["splits"]["train"]["reference"]
    values = reference["conditions"].float()
    valid = reference["condition_validity"].bool()
    count = reference["path_mask"].bool().sum(-1)
    maps = reference["map_id"].long()
    exact = (
        (maps == map_id)
        & (count == path_count)
        & ((values[:, 3] >= 0.5) == (conditions[0, 3] >= 0.5))
        & (valid[:, 2] == validity[0, 2])
    )
    candidates = torch.nonzero(exact, as_tuple=False).flatten()
    if not candidates.numel():
        return {
            "status": "out_of_support",
            "exact_structural_candidates": 0,
            "map_id": map_id,
            "path_count": path_count,
        }
    scale = values.std(0, unbiased=False).clamp_min(1e-6)
    dimensions = torch.tensor([0, 1, 2]) if bool(validity[0, 2]) else torch.tensor([0, 1])
    distance = (
        (values[candidates][:, dimensions] - conditions[0, dimensions])
        / scale[dimensions]
    ).square().sum(-1).sqrt()
    nearest = float(distance.min())
    within_one = int((distance <= 1.0).sum())
    within_two = int((distance <= 2.0).sum())
    status = "supported" if within_one >= 12 else "low_support" if within_two >= 12 else "out_of_support"
    return {
        "status": status,
        "exact_structural_candidates": int(candidates.numel()),
        "standardized_condition_distance_minimum": nearest,
        "neighbors_within_1": within_one,
        "neighbors_within_2": within_two,
        "map_id": map_id,
        "path_count": path_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CSI with B3 + DP + Link-global and optional dual-mode AoD control."
    )
    parser.add_argument("--b3-checkpoint", required=True)
    parser.add_argument("--dp-checkpoint", required=True)
    parser.add_argument("--link-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--support-cache", default=None)
    parser.add_argument("--map-id", type=_map_id, required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--total-power-db", type=float, required=True)
    parser.add_argument("--delay-spread-ns", type=float, required=True)
    parser.add_argument("--k-factor-db", type=float, default=0.0)
    parser.add_argument("--los-status", type=int, choices=[0, 1], required=True)
    parser.add_argument("--path-count", type=int, required=True)
    parser.add_argument("--target-aod-az-deg", type=float, default=None)
    parser.add_argument(
        "--aod-control-mode",
        choices=["none", "mean", "strongest"],
        default="mean",
        help=(
            "Reference used by the optional target angle. 'none' explicitly disables "
            "the projector and cannot be combined with --target-aod-az-deg."
        ),
    )
    parser.add_argument("--minimum-mean-concentration", type=float, default=0.10)
    parser.add_argument("--strongest-near-tie-margin", type=float, default=0.02)
    parser.add_argument(
        "--selection-strategy",
        choices=["direct", "resample_project"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--control-policy",
        choices=["direct", "realism_first", "exact_control"],
        default=None,
        help=(
            "realism_first accepts natural links inside a fixed tolerance without "
            "rotation; exact_control requires the selected residual rotation to stay "
            "inside its safety threshold. When omitted, a supplied target uses "
            "realism_first and generation without a target remains uncontrolled."
        ),
    )
    parser.add_argument("--candidate-multiplier", type=int, default=20)
    parser.add_argument("--candidate-generation-batch-size", type=int, default=512)
    parser.add_argument("--natural-angle-tolerance-deg", type=float, default=10.0)
    parser.add_argument("--maximum-small-projection-deg", type=float, default=10.0)
    parser.add_argument(
        "--require-small-projection", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--exclude-strongest-near-ties", action="store_true")
    parser.add_argument(
        "--k-factor-validity",
        choices=["auto", "valid", "invalid"],
        default="auto",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    args = parser.parse_args()
    if args.control_policy is not None:
        control_policy = args.control_policy
    elif args.selection_strategy == "resample_project":
        control_policy = "exact_control"
    elif args.selection_strategy == "direct":
        control_policy = "direct"
    elif args.target_aod_az_deg is not None:
        control_policy = "realism_first"
    else:
        control_policy = "direct"
    if args.num_samples < 1:
        parser.error("num-samples must be positive")
    if args.aod_control_mode == "none" and args.target_aod_az_deg is not None:
        parser.error("--aod-control-mode none cannot be combined with a target angle")
    if control_policy in {"realism_first", "exact_control"} and args.target_aod_az_deg is None:
        parser.error(f"{control_policy} requires --target-aod-az-deg")
    if args.candidate_multiplier < 1:
        parser.error("candidate-multiplier must be positive")
    if args.candidate_generation_batch_size < 1:
        parser.error("candidate-generation-batch-size must be positive")
    if not 0.0 < args.maximum_small_projection_deg <= 180.0:
        parser.error("maximum-small-projection-deg must lie in (0, 180]")
    if not 0.0 <= args.natural_angle_tolerance_deg <= 180.0:
        parser.error("natural-angle-tolerance-deg must lie in [0, 180]")
    k_valid = (
        bool(args.los_status and args.path_count >= 2)
        if args.k_factor_validity == "auto"
        else args.k_factor_validity == "valid"
    )
    condition_dict = {
        "total_power_db": args.total_power_db,
        "delay_spread_ns": args.delay_spread_ns,
        "k_factor_db": args.k_factor_db,
        "los_status": args.los_status,
    }
    check_condition_feasibility(
        condition_dict, path_count=args.path_count, k_factor_valid=k_valid
    )

    device = resolve_device(args.device)
    b3, dp, link, b3_checkpoint = _load_models(
        Path(args.b3_checkpoint),
        Path(args.dp_checkpoint),
        Path(args.link_checkpoint),
        device,
    )
    if not 1 <= args.path_count <= b3.max_paths:
        parser.error(f"path-count must lie in [1, {b3.max_paths}]")
    candidate_samples = args.num_samples * (
        args.candidate_multiplier
        if control_policy in {"realism_first", "exact_control"}
        else 1
    )
    condition_row = torch.tensor(
        [args.total_power_db, args.delay_spread_ns, args.k_factor_db, args.los_status],
        dtype=torch.float32,
    )
    sampling = b3_checkpoint["config"]["sampling"]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    started = time.perf_counter()
    base_chunks: dict[str, list[torch.Tensor]] = {}
    dp_chunks: dict[str, list[torch.Tensor]] = {}
    natural_chunks: dict[str, list[torch.Tensor]] = {}
    with torch.inference_mode():
        for start in range(0, candidate_samples, args.candidate_generation_batch_size):
            batch_samples = min(
                args.candidate_generation_batch_size, candidate_samples - start
            )
            conditions_batch = condition_row.to(device).expand(batch_samples, -1)
            validity_batch = torch.ones_like(conditions_batch, dtype=torch.bool)
            validity_batch[:, 2] = k_valid
            external_map = torch.full(
                (batch_samples,), args.map_id, device=device, dtype=torch.long
            )
            map_index = link.map_indices(external_map)
            if not torch.equal(map_index, dp.map_indices(external_map)):
                raise RuntimeError("DP and Link-global resolved the Map ID differently")
            base_batch = b3.generate(
                conditions_batch,
                batch_samples,
                condition_validity=validity_batch,
                sampler=args.sampler or sampling["sampler"],
                sampling_steps=(
                    args.sampling_steps
                    if args.sampling_steps is not None
                    else int(sampling["sampling_steps"])
                ),
                eta=float(sampling.get("eta", 0.0)),
                guidance_scale=(
                    args.guidance_scale
                    if args.guidance_scale is not None
                    else float(sampling["guidance_scale"])
                ),
                forced_path_count=args.path_count,
                generator=generator,
            )
            dp_output = dp(
                base_batch, conditions_batch, validity_batch, map_index=map_index
            )
            dp_batch = dp.apply_to_payload(base_batch, dp_output)
            link_output = link(
                dp_batch, conditions_batch, validity_batch, map_index=map_index
            )
            natural_batch = link.apply_to_payload(dp_batch, link_output)
            _append_payload(base_chunks, base_batch)
            _append_payload(dp_chunks, dp_batch)
            _append_payload(natural_chunks, natural_batch)
            print(
                json.dumps(
                    {
                        "event": "dual_mode_aod_candidate_progress",
                        "complete": start + batch_samples,
                        "total": candidate_samples,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        base = _finish_payload(base_chunks)
        dp_payload = _finish_payload(dp_chunks)
        natural = _finish_payload(natural_chunks)
        selection = None
        selection_report: dict[str, Any] | None = None
        selection_projector = AoDControlProjector(
            minimum_mean_concentration=args.minimum_mean_concentration,
            strongest_near_tie_margin=args.strongest_near_tie_margin,
        )
        if control_policy in {"realism_first", "exact_control"}:
            if control_policy == "realism_first":
                selection = select_aod_candidates_within_tolerance(
                    natural,
                    args.target_aod_az_deg,
                    samples=args.num_samples,
                    tolerance_deg=args.natural_angle_tolerance_deg,
                    mode=args.aod_control_mode,
                    projector=selection_projector,
                    exclude_strongest_near_ties=args.exclude_strongest_near_ties,
                )
            else:
                selection = select_nearest_aod_candidates(
                    natural,
                    args.target_aod_az_deg,
                    samples=args.num_samples,
                    mode=args.aod_control_mode,
                    projector=selection_projector,
                    exclude_strongest_near_ties=args.exclude_strongest_near_ties,
                )
            base = select_payload_samples(base, selection.indices)
            dp_payload = select_payload_samples(dp_payload, selection.indices)
            natural = select_payload_samples(natural, selection.indices)
            selected_distance = selection.angular_distance_deg
            selection_report = {
                "policy": control_policy,
                "candidate_samples": selection.candidate_samples,
                "eligible_samples": selection.eligible_samples,
                "acceptance_rate": (
                    selection.eligible_samples / selection.candidate_samples
                ),
                "selected_samples": selection.requested_samples,
                "unique_selected_samples": int(torch.unique(selection.indices).numel()),
                "candidate_multiplier": args.candidate_multiplier,
                "natural_angle_tolerance_deg": (
                    args.natural_angle_tolerance_deg
                    if control_policy == "realism_first"
                    else None
                ),
                "pre_projection_angular_distance_deg": _audit(selected_distance),
                "fraction_within_15_deg": float((selected_distance <= 15.0).float().mean()),
                "fraction_within_45_deg": float((selected_distance <= 45.0).float().mean()),
                "small_projection_threshold_deg": args.maximum_small_projection_deg,
                "fraction_within_small_projection_threshold": float(
                    (selected_distance <= args.maximum_small_projection_deg).float().mean()
                ),
                "selected_candidate_indices": selection.indices.detach().cpu().tolist(),
            }
            if control_policy == "exact_control" and bool(
                (selected_distance > args.maximum_small_projection_deg).any()
            ):
                raise RuntimeError(
                    "candidate pool cannot satisfy the required small-projection threshold; "
                    "increase --candidate-multiplier or relax the threshold"
                )
        else:
            selection_report = {
                "policy": "direct",
                "candidate_samples": candidate_samples,
                "selected_samples": args.num_samples,
            }
        conditions = condition_row.expand(args.num_samples, -1).clone()
        validity = torch.ones_like(conditions, dtype=torch.bool)
        validity[:, 2] = k_valid
        synthesizer = MIMOOFDMSynthesizer(b3.synthesizer.config).to(device).eval()
        natural_device = _to_device(natural, device)
        natural_device["csi"] = _synthesize(natural_device, synthesizer)
        natural = _cpu(natural_device)
        control_output = None
        projection_applied = False
        if args.target_aod_az_deg is not None:
            projector = AoDControlProjector(
                minimum_mean_concentration=args.minimum_mean_concentration,
                strongest_near_tie_margin=args.strongest_near_tie_margin,
            )
            control_output = projector(
                natural_device,
                args.target_aod_az_deg,
                mode=args.aod_control_mode,
            )
            if control_policy == "realism_first":
                controlled = dict(natural)
            else:
                projection_applied = True
                controlled_device = projector.apply_to_payload(
                    natural_device, control_output
                )
                controlled_device["csi"] = _synthesize(controlled_device, synthesizer)
                controlled = _cpu(controlled_device)
        else:
            controlled = dict(natural)
    elapsed = time.perf_counter() - started

    preserve_keys = (
        "path_mask",
        "path_delays_ns",
        "path_powers_linear",
        "path_phases_rad",
        "aoa_az_deg",
        "aoa_el_deg",
        "aod_el_deg",
    )
    preservation = {
        key: (
            float((controlled[key].float() - natural[key].float()).abs().max())
            if controlled[key].is_floating_point()
            else float((controlled[key] != natural[key]).float().max())
        )
        for key in preserve_keys
    }
    support = _support_audit(
        args.support_cache,
        map_id=args.map_id,
        conditions=conditions.detach().cpu(),
        validity=validity.detach().cpu(),
        path_count=args.path_count,
    )
    control_report: dict[str, Any] | None = None
    control_tensors: dict[str, torch.Tensor] | None = None
    if control_output is not None:
        final_error = (
            control_output.circular_error_deg
            if projection_applied
            else control_output.rotation_deg.abs()
        )
        control_report = {
            "mode": control_output.mode,
            "policy": control_policy,
            "projection_applied": projection_applied,
            "target_az_deg": args.target_aod_az_deg,
            "natural_angle_error_deg": _audit(control_output.rotation_deg.abs()),
            "final_angle_error_deg": _audit(final_error),
            "rotation_absolute_deg": _audit(control_output.rotation_deg.abs()),
            "valid_rate": float(control_output.control_valid.float().mean()),
            "mean_resultant_concentration": _audit(
                control_output.mean_resultant_concentration
            ),
            "strongest_near_tie_rate": float(
                control_output.strongest_near_tie.float().mean()
            ),
            "strongest_power_margin_fraction": _audit(
                control_output.strongest_power_margin_fraction
            ),
        }
        control_tensors = {
            "target_az_deg": control_output.target_az_deg,
            "reference_before_deg": control_output.reference_before_deg,
            "reference_after_deg": (
                control_output.reference_after_deg
                if projection_applied
                else control_output.reference_before_deg
            ),
            "requested_rotation_deg": control_output.rotation_deg,
            "applied_rotation_deg": (
                control_output.rotation_deg
                if projection_applied
                else torch.zeros_like(control_output.rotation_deg)
            ),
            "final_circular_error_deg": final_error,
            "control_valid": control_output.control_valid,
            "mean_resultant_concentration": control_output.mean_resultant_concentration,
            "strongest_path_index": control_output.strongest_path_index,
            "strongest_power_margin_fraction": control_output.strongest_power_margin_fraction,
            "strongest_near_tie": control_output.strongest_near_tie,
        }
    artifact = {
        "format_version": 1,
        "model_kind": "b3_dp_link_global_dual_mode_aod_control",
        "conditions": condition_dict,
        "condition_validity": validity[0].detach().cpu(),
        "map_id": args.map_id,
        "path_count": args.path_count,
        "seed": args.seed,
        "control_mode": args.aod_control_mode if control_output is not None else "natural",
        "control_policy": control_policy,
        "target_aod_az_deg": args.target_aod_az_deg,
        "base": _cpu(base),
        "dp_only": _cpu(dp_payload),
        "natural": _cpu(natural),
        "controlled": _cpu(controlled),
        "control_audit": _cpu(control_tensors),
        "selection_audit": selection_report,
        "checkpoint_paths": {
            "b3": str(Path(args.b3_checkpoint).resolve()),
            "dp": str(Path(args.dp_checkpoint).resolve()),
            "link_global": str(Path(args.link_checkpoint).resolve()),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(artifact, temporary)
    temporary.replace(output)
    report = {
        "format_version": 1,
        "experiment": "dual_mode_aod_control_generation",
        "artifact": str(output.resolve()),
        "samples": args.num_samples,
        "candidate_samples": candidate_samples,
        "seconds": elapsed,
        "samples_per_second": args.num_samples / max(elapsed, 1e-20),
        "conditions": condition_dict,
        "condition_validity": validity[0].detach().cpu().tolist(),
        "map_id": args.map_id,
        "path_count": args.path_count,
        "selection": selection_report,
        "control": control_report,
        "preservation_maximum_absolute_error": preservation,
        "scene_condition_support": support,
        "interpretation": (
            "Numerical control accuracy does not establish Map-conditional realism. Treat "
            "low/out-of-support conditions as controllable extrapolations only."
        ),
    }
    report_path = Path(args.report) if args.report else output.with_suffix(".json")
    atomic_json_dump(report, report_path)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
