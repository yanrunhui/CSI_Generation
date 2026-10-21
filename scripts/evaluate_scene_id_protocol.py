#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.data.scene_id import make_scene_id_datasets
from physics_csi.evaluation.diffusion_comparison import evaluate_generated_payload
from physics_csi.models.diffusion.multipath_diffusion import MultipathDiffusionModel
from physics_csi.physics.synthesizer import SystemConfig
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


REFERENCE_KEYS = (
    "csi",
    "map_id",
    "path_count",
    "path_mask",
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


def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    return value


def _reference(records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        key: torch.stack([record[key] for record in records])
        for key in REFERENCE_KEYS
    }


def _concatenate(chunks: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat([chunk[key] for chunk in chunks]) for key in chunks[0]}


def _subset(payload: dict[str, torch.Tensor], keep: torch.Tensor) -> dict[str, torch.Tensor]:
    count = int(keep.numel())
    return {
        key: value[keep]
        if isinstance(value, torch.Tensor) and value.shape[:1] == (count,)
        else value
        for key, value in payload.items()
    }


def _path_count_report(
    generated: torch.Tensor, reference: torch.Tensor, classes: int
) -> dict[str, Any]:
    generated_histogram = torch.bincount(generated.long(), minlength=classes + 1)[1:]
    reference_histogram = torch.bincount(reference.long(), minlength=classes + 1)[1:]
    generated_frequency = generated_histogram.float() / generated_histogram.sum().clamp_min(1)
    reference_frequency = reference_histogram.float() / reference_histogram.sum().clamp_min(1)
    flat = (reference.long() - 1) * classes + (generated.long() - 1)
    matrix = torch.bincount(flat, minlength=classes * classes).reshape(classes, classes)
    support = matrix.sum(1)
    recall = matrix.diag().float() / support.clamp_min(1)
    return {
        "generated_histogram": generated_histogram.tolist(),
        "reference_histogram": reference_histogram.tolist(),
        "total_variation_distance": float(
            0.5 * (generated_frequency - reference_frequency).abs().sum()
        ),
        "confusion_matrix": matrix.tolist(),
        "recall": recall.tolist(),
        "accuracy": float(matrix.diag().sum() / matrix.sum().clamp_min(1)),
        "balanced_accuracy": float(recall[support > 0].mean()),
    }


def _load_model(
    checkpoint: dict[str, Any], device: torch.device
) -> MultipathDiffusionModel:
    config = checkpoint["config"]
    model = MultipathDiffusionModel(
        config["multipath_diffusion"],
        SystemConfig.from_mapping(config["data"]["system"]),
        checkpoint["statistics"],
    ).to(device)
    model.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))
    return model.eval()


def _wrong_map_ids(map_ids: torch.Tensor, known_maps: list[int]) -> torch.Tensor:
    if len(known_maps) < 2:
        raise ValueError("Map Shuffle Ablation requires at least two learned maps")
    shifted = {
        current: known_maps[(index + 1) % len(known_maps)]
        for index, current in enumerate(known_maps)
    }
    return torch.tensor([shifted[int(value)] for value in map_ids], dtype=torch.long)


@torch.no_grad()
def _generate(
    model: MultipathDiffusionModel,
    reference: dict[str, torch.Tensor],
    map_ids: torch.Tensor | None,
    *,
    batch_size: int,
    sampling: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    chunks: list[dict[str, torch.Tensor]] = []
    sample_count = reference["conditions"].shape[0]
    for start in range(0, sample_count, batch_size):
        stop = min(start + batch_size, sample_count)
        generated = model.generate(
            reference["conditions"][start:stop].to(device),
            stop - start,
            condition_validity=reference["condition_validity"][start:stop].to(device),
            map_id=(map_ids[start:stop] if map_ids is not None else None),
            sampler=sampling["sampler"],
            sampling_steps=int(sampling["sampling_steps"]),
            eta=float(sampling["eta"]),
            guidance_scale=float(sampling["guidance_scale"]),
            path_count_temperature=float(sampling["path_count_temperature"]),
            generator=generator,
        )
        chunks.append(_cpu(generated))
    return _concatenate(chunks)


def _evaluate(
    generated: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    max_paths: int,
) -> dict[str, Any]:
    report = evaluate_generated_payload("multipath_diffusion", generated, reference)
    report["path_count"] = _path_count_report(
        generated["path_count"], reference["path_count"], max_paths
    )
    return report


def _distance_value(report: dict[str, Any], section: str, key: str) -> float | None:
    value = report.get(section, {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified B3 versus B3-S-ID per-map and map-shuffle evaluation."
    )
    parser.add_argument("--scene-checkpoint", required=True)
    parser.add_argument(
        "--baseline-checkpoint",
        default=None,
        help="Matched physics-only checkpoint trained with the same scene plan.",
    )
    parser.add_argument("--prepared-root", default=None)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--minimum-per-map", type=int, default=8)
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--path-count-temperature", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--generated-output", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = resolve_device(args.device)
    scene_checkpoint = torch.load(
        args.scene_checkpoint, map_location="cpu", weights_only=False
    )
    if scene_checkpoint.get("model_kind") != "multipath_diffusion_scene_id":
        raise ValueError("--scene-checkpoint is not a B3-S-ID checkpoint")
    scene_config = scene_checkpoint["config"]
    plan = scene_config.get("scene_id_experiment")
    if plan is None:
        raise ValueError("scene checkpoint does not contain its split plan")
    baseline_checkpoint: dict[str, Any] | None = None
    if args.baseline_checkpoint:
        baseline_checkpoint = torch.load(
            args.baseline_checkpoint, map_location="cpu", weights_only=False
        )
        if baseline_checkpoint.get("model_kind") != "multipath_diffusion_seen_map_baseline":
            raise ValueError("baseline must be trained by train_scene_id_diffusion.py")
        baseline_plan = baseline_checkpoint["config"].get("scene_id_experiment")
        if baseline_plan is None or baseline_plan.get("fingerprint") != plan["fingerprint"]:
            raise ValueError("B3 and B3-S-ID checkpoints were not trained on the same split")

    prepared_root = args.prepared_root or scene_config["data"]["prepared_root"]
    dataset = make_scene_id_datasets(prepared_root, plan)["test"]
    indices = list(range(len(dataset)))
    selection_generator = torch.Generator().manual_seed(args.seed)
    if len(indices) > args.max_samples:
        selected = torch.randperm(len(indices), generator=selection_generator)[: args.max_samples]
        indices = sorted(int(index) for index in selected)
    records = [dataset[index] for index in indices]
    if not records:
        raise ValueError("test split is empty")
    reference = _reference(records)
    known_maps = sorted(int(value) for value in plan["map_id_to_index"])
    wrong_map_ids = _wrong_map_ids(reference["map_id"], known_maps)
    sampling_config = scene_config["sampling"]
    sampling = {
        "sampler": args.sampler or sampling_config["sampler"],
        "sampling_steps": (
            args.sampling_steps
            if args.sampling_steps is not None
            else int(sampling_config["sampling_steps"])
        ),
        "eta": float(sampling_config.get("eta", 0.0)),
        "guidance_scale": (
            args.guidance_scale
            if args.guidance_scale is not None
            else float(sampling_config["guidance_scale"])
        ),
        "path_count_temperature": (
            args.path_count_temperature
            if args.path_count_temperature is not None
            else float(sampling_config.get("path_count_temperature", 1.0))
        ),
    }
    baseline_sampling = dict(sampling)
    if baseline_checkpoint is not None and args.path_count_temperature is None:
        baseline_sampling["path_count_temperature"] = float(
            baseline_checkpoint["config"]["sampling"].get(
                "path_count_temperature", 1.0
            )
        )
    scene_model = _load_model(scene_checkpoint, device)
    correct = _generate(
        scene_model,
        reference,
        reference["map_id"],
        batch_size=args.batch_size,
        sampling=sampling,
        seed=args.seed,
        device=device,
    )
    shuffled = _generate(
        scene_model,
        reference,
        wrong_map_ids,
        batch_size=args.batch_size,
        sampling=sampling,
        seed=args.seed,
        device=device,
    )
    baseline = (
        _generate(
            _load_model(baseline_checkpoint, device),
            reference,
            None,
            batch_size=args.batch_size,
            sampling=baseline_sampling,
            seed=args.seed,
            device=device,
        )
        if baseline_checkpoint is not None
        else None
    )
    overall = {
        "correct_map": _evaluate(correct, reference, scene_model.max_paths),
        "shuffled_map": _evaluate(shuffled, reference, scene_model.max_paths),
    }
    if baseline is not None:
        overall["physics_only_baseline"] = _evaluate(
            baseline, reference, scene_model.max_paths
        )
    paired_difference = (
        (correct["csi"] - shuffled["csi"]).abs().square().flatten(1).sum(-1)
        / (
            0.5
            * (
                correct["csi"].abs().square().flatten(1).sum(-1)
                + shuffled["csi"].abs().square().flatten(1).sum(-1)
            )
        ).clamp_min(1e-20)
    )

    per_map: dict[str, Any] = {}
    scene_better_than_shuffle = 0
    scene_better_than_baseline = 0
    evaluated_maps = 0
    for map_id in known_maps:
        keep = reference["map_id"] == map_id
        sample_count = int(keep.sum())
        if sample_count < args.minimum_per_map:
            per_map[f"map_{map_id}"] = {
                "samples": sample_count,
                "status": "insufficient_reference_samples",
            }
            continue
        ref_map = _subset(reference, keep)
        correct_report = _evaluate(
            _subset(correct, keep), ref_map, scene_model.max_paths
        )
        shuffled_report = _evaluate(
            _subset(shuffled, keep), ref_map, scene_model.max_paths
        )
        item: dict[str, Any] = {
            "samples": sample_count,
            "training_samples": plan["support"][str(map_id)]["train"]["samples"],
            "correct_map": correct_report,
            "shuffled_map": shuffled_report,
        }
        correct_mmd = _distance_value(
            correct_report, "distribution", "csi_summary_mmd_rbf"
        )
        shuffled_mmd = _distance_value(
            shuffled_report, "distribution", "csi_summary_mmd_rbf"
        )
        if correct_mmd is not None and shuffled_mmd is not None:
            item["correct_map_improves_csi_mmd_over_shuffle"] = correct_mmd < shuffled_mmd
            scene_better_than_shuffle += int(correct_mmd < shuffled_mmd)
        if baseline is not None:
            baseline_report = _evaluate(
                _subset(baseline, keep), ref_map, scene_model.max_paths
            )
            item["physics_only_baseline"] = baseline_report
            baseline_mmd = _distance_value(
                baseline_report, "distribution", "csi_summary_mmd_rbf"
            )
            if correct_mmd is not None and baseline_mmd is not None:
                item["scene_id_improves_csi_mmd_over_baseline"] = correct_mmd < baseline_mmd
                scene_better_than_baseline += int(correct_mmd < baseline_mmd)
        per_map[f"map_{map_id}"] = item
        evaluated_maps += 1

    report: dict[str, Any] = {
        "format_version": 1,
        "protocol": "seen_map_per_condition_with_map_shuffle",
        "scene_checkpoint": str(Path(args.scene_checkpoint).resolve()),
        "baseline_checkpoint": (
            str(Path(args.baseline_checkpoint).resolve())
            if args.baseline_checkpoint
            else None
        ),
        "scene_plan_fingerprint": plan["fingerprint"],
        "samples": len(records),
        "maps_with_metrics": evaluated_maps,
        "sampling": {
            "scene_id": sampling,
            "physics_only_baseline": (
                baseline_sampling if baseline_checkpoint is not None else None
            ),
            "note": (
                "Sampler, step count and guidance are matched. Independently calibrated "
                "path-count temperatures are retained unless explicitly overridden."
            ),
        },
        "overall": overall,
        "map_embedding_effect": {
            "paired_correct_vs_shuffled_csi_relative_distance_mean": float(
                paired_difference.mean()
            ),
            "path_count_changed_rate": float(
                (correct["path_count"] != shuffled["path_count"]).float().mean()
            ),
            "maps_where_correct_id_beats_shuffled_id_by_csi_mmd": scene_better_than_shuffle,
            "maps_where_scene_id_beats_physics_only_by_csi_mmd": (
                scene_better_than_baseline if baseline is not None else None
            ),
            "evaluated_maps": evaluated_maps,
            "interpretation": (
                "A changed output is not sufficient: correct Map ID should also reduce the "
                "per-map distance to held-out real samples."
            ),
        },
        "per_map": per_map,
    }
    atomic_json_dump(report, args.output)
    if args.generated_output:
        output = Path(args.generated_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "reference": reference,
                "correct_map": correct,
                "shuffled_map": shuffled,
                "physics_only_baseline": baseline,
                "wrong_map_id": wrong_map_ids,
                "scene_plan_fingerprint": plan["fingerprint"],
            },
            output,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
