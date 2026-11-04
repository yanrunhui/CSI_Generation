#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.data.delay_power_residual import DelayPowerResidualCacheDataset
from physics_csi.evaluation.scene_diagnostics import paired_map_summary
from physics_csi.models.scene_aod_residual import SceneConditionedAoDResidual
from physics_csi.models.scene_delay_power_residual import SceneConditionedDelayPowerResidual
from physics_csi.models.scene_link_global_aod_residual import (
    SceneConditionedLinkGlobalAoDResidual,
)
from physics_csi.models.scene_relative_aod_residual import (
    SceneConditionedRelativeAoDResidual,
)
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig
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
        if isinstance(value, torch.Tensor) and value.shape[:1] == (sample_count,)
    }


def _append(chunks: dict[str, list[torch.Tensor]], payload: Mapping[str, torch.Tensor]) -> None:
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            chunks.setdefault(key, []).append(value.detach().cpu())


def _finish(chunks: Mapping[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat(values, dim=0) for key, values in chunks.items()}


def _synthesize(payload: Mapping[str, torch.Tensor], model: MIMOOFDMSynthesizer) -> torch.Tensor:
    return model(
        payload["path_delays_ns"],
        payload["path_powers_linear"],
        payload["path_phases_rad"],
        payload["aoa_az_deg"],
        payload["aoa_el_deg"],
        payload["aod_az_deg"],
        payload["aod_el_deg"],
        payload["path_mask"],
    )


def _selected_indices(map_ids: torch.Tensor, maximum_per_map: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    result: list[torch.Tensor] = []
    for map_id in torch.unique(map_ids, sorted=True):
        indices = torch.nonzero(map_ids == map_id, as_tuple=False).flatten()
        if indices.numel() > maximum_per_map:
            indices = indices[
                torch.randperm(indices.numel(), generator=generator)[:maximum_per_map]
            ]
        result.append(indices)
    return torch.cat(result).sort().values


def _different_map_indices(correct: torch.Tensor, map_count: int) -> torch.Tensor:
    if map_count < 2:
        raise ValueError("shuffled-Map evaluation requires at least two Maps")
    offset = 1 + torch.remainder(correct * 37, map_count - 1)
    return (correct + offset) % map_count


def _load_dp(checkpoint: Mapping[str, Any], device: torch.device) -> SceneConditionedDelayPowerResidual:
    model = SceneConditionedDelayPowerResidual(**checkpoint["model_init"]).to(device)
    model.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))
    return model.requires_grad_(False).eval()


def _load_aod(checkpoint: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    model_kind = checkpoint.get("model_kind")
    if model_kind == "scene_aod_residual":
        model_class = SceneConditionedAoDResidual
    elif model_kind == "scene_link_global_aod_residual":
        model_class = SceneConditionedLinkGlobalAoDResidual
    elif model_kind == "scene_relative_aod_residual":
        model_class = SceneConditionedRelativeAoDResidual
    else:
        raise ValueError("checkpoint is not a supported AoD residual")
    model = model_class(**checkpoint["model_init"]).to(device)
    model.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))
    return model.requires_grad_(False).eval()


def _quantile_audit(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().abs().flatten()
    return {
        "mean": float(values.mean()),
        "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
        "p99": float(torch.quantile(values, 0.99)),
        "maximum": float(values.max()),
    }


def _power_weighted_direction_center(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask = payload["path_mask"].bool()
    power = payload["path_powers_linear"].float().clamp_min(0.0) * mask
    fraction = power / power.sum(-1, keepdim=True).clamp_min(1e-20)
    azimuth = torch.deg2rad(payload["aod_az_deg"].float())
    elevation = torch.deg2rad(payload["aod_el_deg"].float())
    horizontal = elevation.cos()
    direction = torch.stack(
        [horizontal * azimuth.cos(), horizontal * azimuth.sin(), elevation.sin()],
        dim=-1,
    )
    center = (fraction.unsqueeze(-1) * direction).sum(1)
    return center / center.norm(dim=-1, keepdim=True).clamp_min(1e-20)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate B3 + DP + AoD residual hierarchy.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dp-checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-artifact", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default=None)
    parser.add_argument("--maximum-samples-per-map", type=int, default=None)
    parser.add_argument("--minimum-samples-per-map", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dp_checkpoint = torch.load(args.dp_checkpoint, map_location="cpu", weights_only=False)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if checkpoint.get("model_kind") not in {
        "scene_aod_residual",
        "scene_link_global_aod_residual",
        "scene_relative_aod_residual",
    }:
        raise ValueError("checkpoint is not a supported AoD residual")
    if dp_checkpoint.get("model_kind") != "scene_delay_power_residual":
        raise ValueError("--dp-checkpoint is not a DP residual")
    for contract in (checkpoint["cache_contract"], dp_checkpoint["cache_contract"]):
        if contract["scene_plan_fingerprint"] != cache["scene_plan_fingerprint"]:
            raise ValueError("checkpoint and cache scene plans differ")
        if contract["b3_checkpoint_sha256"] != cache["b3_checkpoint_sha256"]:
            raise ValueError("checkpoint and cache frozen B3 backbones differ")

    evaluation = checkpoint["config"]["evaluation"]
    split = args.split or str(evaluation.get("split", "test"))
    maximum_per_map = args.maximum_samples_per_map or int(
        evaluation.get("maximum_samples_per_map", 64)
    )
    minimum_per_map = args.minimum_samples_per_map or int(
        evaluation.get("minimum_samples_per_map", 12)
    )
    bootstrap_samples = args.bootstrap_samples or int(
        evaluation.get("bootstrap_samples", 10000)
    )
    batch_size = args.batch_size or int(evaluation.get("batch_size", 256))
    dataset = DelayPowerResidualCacheDataset(cache, split)
    indices = _selected_indices(dataset.map_ids, maximum_per_map, args.seed)

    dp_model = _load_dp(dp_checkpoint, device)
    model = _load_aod(checkpoint, device)
    if model.map_id_to_index != dp_model.map_id_to_index:
        raise ValueError("AoD and DP checkpoints use different Map mappings")
    link_model: SceneConditionedLinkGlobalAoDResidual | None = None
    if checkpoint.get("model_kind") == "scene_relative_aod_residual":
        link_path = checkpoint.get("parent_link_global_checkpoint")
        if not link_path:
            raise ValueError("Relative-AoD checkpoint has no Link-global parent")
        link_checkpoint = torch.load(link_path, map_location="cpu", weights_only=False)
        if link_checkpoint.get("model_kind") != "scene_link_global_aod_residual":
            raise ValueError("Relative-AoD parent is not a Link-global checkpoint")
        link_model = SceneConditionedLinkGlobalAoDResidual(
            **link_checkpoint["model_init"]
        ).to(device)
        link_model.load_state_dict(
            link_checkpoint.get("ema_state_dict", link_checkpoint["model_state_dict"])
        )
        link_model.requires_grad_(False).eval()
        if link_model.map_id_to_index != model.map_id_to_index:
            raise ValueError("Relative and Link-global checkpoints use different Map mappings")
    loss_init = dict(checkpoint["loss_init"])
    weights = AoDLossWeights(**loss_init.pop("weights"))
    loss_function = AoDDistributionLoss(**loss_init, weights=weights).to(device).eval()
    synthesizer = MIMOOFDMSynthesizer(
        SystemConfig.from_mapping(checkpoint["system"])
    ).to(device).eval()

    labels = (
        "reference",
        "physics_only_baseline",
        "dp_only",
        "link_global_only",
        "correct_map",
        "shuffled_map",
    )
    chunks: dict[str, dict[str, list[torch.Tensor]]] = {label: {} for label in labels}
    azimuth_residuals: list[torch.Tensor] = []
    elevation_residuals: list[torch.Tensor] = []
    global_center_errors_deg: list[torch.Tensor] = []
    wrong_map_ids: list[torch.Tensor] = []
    dp_preservation_maximum = {
        "path_delays_ns": 0.0,
        "path_powers_linear": 0.0,
        "path_phases_rad": 0.0,
        "aoa_az_deg": 0.0,
        "aoa_el_deg": 0.0,
    }
    with torch.inference_mode():
        for start in range(0, indices.numel(), batch_size):
            selected = indices[start : start + batch_size]
            reference = move_to_device(_subset(dataset.reference, selected), device)
            base = move_to_device(_subset(dataset.base, selected), device)
            conditions = reference["conditions"]
            validity = reference["condition_validity"]
            correct_index = model.map_indices(reference["map_id"]).to(device)
            wrong_index = _different_map_indices(correct_index, len(model.map_id_to_index))
            dp_output = dp_model(base, conditions, validity, map_index=correct_index)
            dp_payload = dp_model.apply_to_payload(base, dp_output)
            if link_model is not None:
                link_output = link_model(
                    dp_payload, conditions, validity, map_index=correct_index
                )
                aod_base = link_model.apply_to_payload(dp_payload, link_output)
            else:
                aod_base = dp_payload
            correct_output = model(
                aod_base, conditions, validity, map_index=correct_index
            )
            shuffled_output = model(
                aod_base, conditions, validity, map_index=wrong_index
            )
            correct = model.apply_to_payload(aod_base, correct_output)
            shuffled = model.apply_to_payload(aod_base, shuffled_output)
            for key in dp_preservation_maximum:
                dp_preservation_maximum[key] = max(
                    dp_preservation_maximum[key],
                    float((correct[key] - dp_payload[key]).abs().max()),
                    float((shuffled[key] - dp_payload[key]).abs().max()),
                )
            for payload in (base, dp_payload, aod_base, correct, shuffled):
                payload["map_id"] = reference["map_id"]
                payload["csi"] = _synthesize(payload, synthesizer)
            reference["csi"] = _synthesize(reference, synthesizer)
            correct["conditioned_map_id"] = reference["map_id"]
            shuffled_external = model.external_map_ids(wrong_index)
            shuffled["conditioned_map_id"] = shuffled_external
            wrong_map_ids.append(shuffled_external.cpu())
            for label, payload in zip(
                labels, (reference, base, dp_payload, aod_base, correct, shuffled)
            ):
                _append(chunks[label], payload)
            active = dp_payload["path_mask"].bool()
            azimuth_residuals.append(correct_output.azimuth_residual_deg[active].cpu())
            elevation_residuals.append(correct_output.elevation_residual_deg[active].cpu())
            before_center = _power_weighted_direction_center(aod_base)
            after_center = _power_weighted_direction_center(correct)
            global_center_errors_deg.append(
                torch.rad2deg(
                    torch.acos(
                        (before_center * after_center).sum(-1).clamp(-1.0, 1.0)
                    )
                ).cpu()
            )

    artifact = {label: _finish(chunks[label]) for label in labels}
    artifact.update(
        {
            "format_version": 1,
            "model_kind": "b3_plus_dp_plus_aod_residual",
            "aod_architecture": checkpoint.get("aod_architecture", "per_path"),
            "scene_plan_fingerprint": cache["scene_plan_fingerprint"],
            "split": split,
            "seed": args.seed,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "dp_checkpoint": str(Path(args.dp_checkpoint).resolve()),
            "cache": str(Path(args.cache).resolve()),
            "system": checkpoint["system"],
            "shuffled_conditioned_map_id": torch.cat(wrong_map_ids),
            "shuffled_scope": (
                "Relative-AoD Map only; DP and Link-global remain conditioned on the correct target Map."
                if link_model is not None
                else "AoD Map only; DP remains conditioned on the correct target Map."
            ),
        }
    )
    artifact_path = Path(args.output_artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    torch.save(artifact, temporary)
    temporary.replace(artifact_path)

    per_map: dict[str, Any] = {}
    metric_names = (
        "path_power_aod_swd",
        "delay_power_aod_swd",
        "link_power_aod_swd",
        "power_weighted_aod_swd",
        "tx_spatial_correlation_relative_frobenius",
        "tx_near_spatial_correlation_relative_l2",
        "distribution_score",
    )
    maps = sorted(int(value) for value in torch.unique(artifact["reference"]["map_id"]))
    with torch.inference_mode():
        for map_id in maps:
            map_indices = torch.nonzero(
                artifact["reference"]["map_id"] == map_id, as_tuple=False
            ).flatten()
            if map_indices.numel() < minimum_per_map:
                per_map[f"map_{map_id}"] = {
                    "map_id": map_id,
                    "samples": int(map_indices.numel()),
                    "status": "insufficient_samples",
                }
                continue
            reference = move_to_device(_subset(artifact["reference"], map_indices), device)
            models: dict[str, Any] = {}
            for label in (
                "physics_only_baseline",
                "dp_only",
                "link_global_only",
                "correct_map",
                "shuffled_map",
            ):
                candidate = move_to_device(_subset(artifact[label], map_indices), device)
                terms = loss_function.distribution_terms(candidate, reference)
                models[label] = {key: float(value) for key, value in terms.items()}
            per_map[f"map_{map_id}"] = {
                "map_id": map_id,
                "samples": int(map_indices.numel()),
                "status": "ok",
                "models": models,
            }

    valid_maps = [value for value in per_map.values() if value.get("status") == "ok"]
    if not valid_maps:
        raise RuntimeError("no Map has enough AoD evaluation samples")
    summaries: dict[str, Any] = {}
    for comparison_number, comparison in enumerate(
        ("link_global_only", "dp_only", "shuffled_map", "physics_only_baseline")
    ):
        section: dict[str, Any] = {}
        for metric_number, metric in enumerate(metric_names):
            values = paired_map_summary(
                [item["models"]["correct_map"][metric] for item in valid_maps],
                [item["models"][comparison][metric] for item in valid_maps],
                bootstrap_samples=bootstrap_samples,
                seed=args.seed + comparison_number * 1000 + metric_number,
            )
            values["map_ids"] = [item["map_id"] for item in valid_maps]
            section[metric] = values
        summaries[f"correct_map_vs_{comparison}"] = section

    azimuth = torch.cat(azimuth_residuals)
    elevation = torch.cat(elevation_residuals)
    report = {
        "format_version": 1,
        "experiment": "b3_dp_scene_conditioned_aod_residual",
        "aod_architecture": checkpoint.get("aod_architecture", "per_path"),
        "metric_direction": "lower_is_better",
        "artifact": str(artifact_path.resolve()),
        "samples": int(indices.numel()),
        "maps_evaluated": len(valid_maps),
        "shuffled_scope": artifact["shuffled_scope"],
        "summaries": summaries,
        "per_map": per_map,
        "residual_audit": {
            "azimuth_residual_deg": _quantile_audit(azimuth),
            "elevation_residual_deg": _quantile_audit(elevation),
            "azimuth_gate": float(torch.sigmoid(model.azimuth_gate_logit)),
            "elevation_gate": float(torch.sigmoid(model.elevation_gate_logit)),
            "power_weighted_global_center_error_deg": _quantile_audit(
                torch.cat(global_center_errors_deg)
            ),
        },
        "dp_preservation_maximum_absolute_error": dp_preservation_maximum,
        "success_rule": (
            (
                "Relative-AoD must beat frozen Link-global-only and shuffled-Relative Map on "
                "near-neighbor Tx correlation, power-weighted and link-level Power--AoD across "
                "Maps. Delay--Power and the Link-global input are preserved exactly by "
                "architecture; PDP and complex CSI remain external non-regression gates."
            )
            if link_model is not None
            else (
                "Correct AoD Map must beat DP-only and shuffled-AoD Map on power-weighted and "
                "link-level Power--AoD across Maps. Delay--Power is preserved exactly by "
                "architecture; PDP and complex CSI remain external non-regression gates."
            )
        ),
        "cautions": [
            "This report uses one B3 generation cache and one AoD training seed.",
            "Frozen Phase does not freeze CSI because AoD changes the transmit array response.",
            "Run the independent Power--AoD, Delay--Power, joint CSI/PDP and phase-invariant diagnostics on the emitted artifact.",
        ],
    }
    atomic_json_dump(report, args.output_report)
    print(
        json.dumps(
            {
                "event": "aod_residual_evaluation_complete",
                "artifact": str(artifact_path.resolve()),
                "report": str(Path(args.output_report).resolve()),
                "samples": int(indices.numel()),
                "maps_evaluated": len(valid_maps),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
