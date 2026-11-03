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
from physics_csi.models.scene_delay_power_residual import SceneConditionedDelayPowerResidual
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig
from physics_csi.training.delay_power_residual import (
    DelayPowerDistributionLoss,
    DelayPowerLossWeights,
)
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


def _different_map_indices(correct: torch.Tensor, map_count: int) -> torch.Tensor:
    # Deterministic per-target-Map derangement: all samples from one target Map
    # receive the same wrong Map, avoiding an artificially heterogeneous foil.
    if map_count < 2:
        raise ValueError("shuffled-Map evaluation requires at least two Maps")
    offset = 1 + torch.remainder(correct * 37, map_count - 1)
    return (correct + offset) % map_count


def _selected_indices(
    map_ids: torch.Tensor, maximum_per_map: int, seed: int
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    selected: list[torch.Tensor] = []
    for map_id in torch.unique(map_ids, sorted=True):
        indices = torch.nonzero(map_ids == map_id, as_tuple=False).flatten()
        if indices.numel() > maximum_per_map:
            indices = indices[
                torch.randperm(indices.numel(), generator=generator)[:maximum_per_map]
            ]
        selected.append(indices)
    return torch.cat(selected).sort().values


def _condition_audit(
    payload: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    target = payload.get("target_conditions", reference["conditions"]).float()
    validity = reference["condition_validity"].bool()
    error = (payload["actual_conditions"].float() - target).abs()
    valid_error = error[validity]
    return {
        "feasible_rate": float(payload["constraint_feasible"].float().mean()),
        "valid_condition_mae": float(valid_error.mean()) if valid_error.numel() else 0.0,
        "valid_condition_maximum_absolute_error": (
            float(valid_error.max()) if valid_error.numel() else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate B3 + scene-conditioned Delay--Power residual."
    )
    parser.add_argument("--checkpoint", required=True)
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
    if checkpoint.get("model_kind") != "scene_delay_power_residual":
        raise ValueError("checkpoint is not a Delay--Power residual")
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
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache["scene_plan_fingerprint"] != checkpoint["cache_contract"]["scene_plan_fingerprint"]:
        raise ValueError("cache and residual checkpoint use different scene plans")
    if cache["b3_checkpoint_sha256"] != checkpoint["cache_contract"]["b3_checkpoint_sha256"]:
        raise ValueError("cache and residual checkpoint use different B3 backbones")

    dataset = DelayPowerResidualCacheDataset(cache, split)
    indices = _selected_indices(dataset.map_ids, maximum_per_map, args.seed)
    model = SceneConditionedDelayPowerResidual(**checkpoint["model_init"]).to(device)
    model.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))
    model.requires_grad_(False).eval()
    loss_init = dict(checkpoint["loss_init"])
    weights = DelayPowerLossWeights(**loss_init.pop("weights"))
    loss_function = DelayPowerDistributionLoss(**loss_init, weights=weights).to(device).eval()
    synthesizer = MIMOOFDMSynthesizer(
        SystemConfig.from_mapping(checkpoint["system"])
    ).to(device).eval()

    chunks: dict[str, dict[str, list[torch.Tensor]]] = {
        "reference": {},
        "physics_only_baseline": {},
        "correct_map": {},
        "shuffled_map": {},
    }
    shuffled_external_chunks: list[torch.Tensor] = []
    residual_audit: dict[str, list[torch.Tensor]] = {
        "delay_log_gap_residual": [],
        "power_logit_residual": [],
        "projection_delay_change_ns": [],
        "projection_power_fraction_l1": [],
    }
    with torch.inference_mode():
        for start in range(0, indices.numel(), batch_size):
            selected = indices[start : start + batch_size]
            reference = move_to_device(_subset(dataset.reference, selected), device)
            base = move_to_device(_subset(dataset.base, selected), device)
            conditions = reference["conditions"]
            validity = reference["condition_validity"]
            correct_index = model.map_indices(reference["map_id"]).to(device)
            shuffled_index = _different_map_indices(correct_index, len(model.map_id_to_index))
            correct_output = model(
                base, conditions, validity, map_index=correct_index
            )
            shuffled_output = model(
                base, conditions, validity, map_index=shuffled_index
            )
            correct = model.apply_to_payload(base, correct_output)
            shuffled = model.apply_to_payload(base, shuffled_output)
            for payload in (base, correct, shuffled):
                payload["csi"] = _synthesize(payload, synthesizer)
            reference["csi"] = _synthesize(reference, synthesizer)
            # Keep the true target Map on every candidate; record the actually
            # conditioned wrong Map separately to prevent ambiguous analysis.
            for payload in (base, correct, shuffled):
                payload["map_id"] = reference["map_id"]
            correct["conditioned_map_id"] = reference["map_id"]
            shuffled_external = model.external_map_ids(shuffled_index)
            shuffled["conditioned_map_id"] = shuffled_external
            shuffled_external_chunks.append(shuffled_external.cpu())
            _append(chunks["reference"], reference)
            _append(chunks["physics_only_baseline"], base)
            _append(chunks["correct_map"], correct)
            _append(chunks["shuffled_map"], shuffled)
            residual_audit["delay_log_gap_residual"].append(
                correct_output.delay_log_gap_residual.cpu()
            )
            residual_audit["power_logit_residual"].append(
                correct_output.power_logit_residual.cpu()
            )
            residual_audit["projection_delay_change_ns"].append(
                (
                    correct_output.constrained.delays_ns
                    - correct_output.pre_projection_delays_ns
                ).abs().sum(-1).cpu()
            )
            pre_fraction = correct_output.pre_projection_power_fractions
            post_fraction = correct_output.constrained.powers_linear
            post_fraction = post_fraction / post_fraction.sum(-1, keepdim=True).clamp_min(1e-20)
            residual_audit["projection_power_fraction_l1"].append(
                (post_fraction - pre_fraction).abs().sum(-1).cpu()
            )

    artifact = {key: _finish(value) for key, value in chunks.items()}
    artifact.update(
        {
            "format_version": 1,
            "model_kind": "b3_plus_scene_delay_power_residual",
            "scene_plan_fingerprint": cache["scene_plan_fingerprint"],
            "split": split,
            "seed": args.seed,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "cache": str(Path(args.cache).resolve()),
            "shuffled_conditioned_map_id": torch.cat(shuffled_external_chunks),
            "path_count_control": "B3 was forced to the reference path count; residual preserves it.",
        }
    )
    artifact_path = Path(args.output_artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    torch.save(artifact, temporary)
    temporary.replace(artifact_path)

    per_map: dict[str, Any] = {}
    metric_names = (
        "path_delay_power_swd",
        "link_delay_power_swd",
        "power_weighted_delay_cdf",
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
            conditions = reference["conditions"]
            models: dict[str, Any] = {}
            for label in ("physics_only_baseline", "correct_map", "shuffled_map"):
                candidate = move_to_device(_subset(artifact[label], map_indices), device)
                terms = loss_function.distribution_terms(candidate, reference, conditions)
                models[label] = {key: float(value) for key, value in terms.items()}
            per_map[f"map_{map_id}"] = {
                "map_id": map_id,
                "samples": int(map_indices.numel()),
                "status": "ok",
                "models": models,
            }

    summaries: dict[str, Any] = {}
    valid_maps = [values for values in per_map.values() if values.get("status") == "ok"]
    for comparison in ("physics_only_baseline", "shuffled_map"):
        summaries[f"correct_map_vs_{comparison}"] = {}
        for metric_number, metric in enumerate(metric_names):
            summary = paired_map_summary(
                [values["models"]["correct_map"][metric] for values in valid_maps],
                [values["models"][comparison][metric] for values in valid_maps],
                bootstrap_samples=bootstrap_samples,
                seed=args.seed + metric_number + (0 if comparison == "physics_only_baseline" else 1000),
            )
            summary["map_ids"] = [values["map_id"] for values in valid_maps]
            summaries[f"correct_map_vs_{comparison}"][metric] = summary

    report = {
        "format_version": 1,
        "experiment": "b3_scene_conditioned_delay_power_residual",
        "metric_direction": "lower_is_better",
        "artifact": str(artifact_path.resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": split,
        "samples": int(indices.numel()),
        "maps_evaluated": len(valid_maps),
        "path_count_controlled": True,
        "condition_audit": {
            label: _condition_audit(artifact[label], artifact["reference"])
            for label in ("physics_only_baseline", "correct_map", "shuffled_map")
        },
        "residual_audit": {
            key: {
                "mean": float(torch.cat(values).float().mean()),
                "maximum": float(torch.cat(values).float().abs().max()),
            }
            for key, values in residual_audit.items()
        },
        "summaries": summaries,
        "per_map": per_map,
        "success_rule": (
            "Correct Map should beat both B3 and shuffled Map on held-out per-Map DP metrics; "
            "physical feasibility must remain near 1. Existing Power-AoD/CSI diagnostics should "
            "be run on the emitted artifact as non-regression checks."
        ),
        "cautions": [
            "One B3 generation cache seed is evaluated here; repeat cache seeds for final inference.",
            "Path count is controlled by construction, so this experiment isolates Delay--Power transport.",
            "The smooth power-weighted-delay CDF is a training/evaluation proxy; retain the existing quantile diagnostic for the paper table.",
        ],
    }
    atomic_json_dump(report, args.output_report)
    print(
        json.dumps(
            {
                "event": "dp_residual_evaluation_complete",
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
