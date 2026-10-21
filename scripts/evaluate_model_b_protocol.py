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

from physics_csi.data.conditional_experiments import load_condition_experiment_plan
from physics_csi.data.dataset import PreparedShardDataset
from physics_csi.evaluation.diffusion_comparison import evaluate_generated_payload
from physics_csi.models.diffusion.multipath_diffusion import MultipathDiffusionModel
from physics_csi.physics.synthesizer import SystemConfig
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


REFERENCE_KEYS = (
    "csi",
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


def _concatenate(chunks: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat([chunk[key] for chunk in chunks]) for key in chunks[0]}


def _subset(payload: dict[str, torch.Tensor], keep: torch.Tensor) -> dict[str, torch.Tensor]:
    count = int(keep.numel())
    return {
        key: value[keep] if isinstance(value, torch.Tensor) and value.shape[:1] == (count,) else value
        for key, value in payload.items()
    }


def _reference(records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        key: torch.stack([record[key] for record in records])
        for key in REFERENCE_KEYS
        if key in records[0]
    }


def _path_count_report(
    predicted: torch.Tensor, target: torch.Tensor, classes: int, *, paired: bool
) -> dict[str, Any]:
    generated_histogram = torch.bincount(predicted.long(), minlength=classes + 1)[1:]
    reference_histogram = torch.bincount(target.long(), minlength=classes + 1)[1:]
    generated_frequency = generated_histogram.float() / generated_histogram.sum().clamp_min(1)
    reference_frequency = reference_histogram.float() / reference_histogram.sum().clamp_min(1)
    result: dict[str, Any] = {
        "generated_histogram": generated_histogram.tolist(),
        "reference_histogram": reference_histogram.tolist(),
        "total_variation_distance": float(
            0.5 * (generated_frequency - reference_frequency).abs().sum()
        ),
    }
    if not paired:
        result["pairing"] = "unpaired fixed-condition distribution"
        return result
    flat = (target.long() - 1) * classes + (predicted.long() - 1)
    matrix = torch.bincount(flat, minlength=classes * classes).reshape(classes, classes)
    support = matrix.sum(1)
    recall = matrix.diag().float() / support.clamp_min(1)
    accuracy = matrix.diag().sum().float() / matrix.sum().clamp_min(1)
    result.update({
        "rows": "reference path count",
        "columns": "sampled path count",
        "confusion_matrix": matrix.tolist(),
        "support": support.tolist(),
        "recall": recall.tolist(),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "pairing": "same input condition; CSI realizations remain unpaired",
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Condition-matched ID/sparse/OOD evaluation for Model B."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--experiment-plan", default=None)
    parser.add_argument("--protocol", choices=["per_condition", "fixed_region"], required=True)
    parser.add_argument("--prepared-root", default=None)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--path-count-temperature", type=float, default=None)
    parser.add_argument("--force-reference-path-count", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--generated-output", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("model_kind") != "multipath_diffusion":
        raise ValueError("checkpoint is not Model B")
    config = checkpoint["config"]
    embedded_plan = config.get("condition_experiment")
    external_plan = (
        load_condition_experiment_plan(args.experiment_plan)
        if args.experiment_plan
        else None
    )
    if (
        embedded_plan is not None
        and external_plan is not None
        and embedded_plan.get("fingerprint") != external_plan.get("fingerprint")
    ):
        raise ValueError("external experiment plan does not match the checkpoint")
    plan = external_plan or embedded_plan
    prepared_root = args.prepared_root or config["data"]["prepared_root"]
    dataset = PreparedShardDataset(prepared_root, "test")
    candidate_indices = (
        list(plan["indices"]["test_reference"]) if plan is not None else list(range(len(dataset)))
    )
    if not candidate_indices:
        raise ValueError("evaluation plan contains no test reference samples")
    random = torch.Generator().manual_seed(args.seed)
    if len(candidate_indices) > args.max_samples:
        selection = torch.randperm(len(candidate_indices), generator=random)[: args.max_samples]
        candidate_indices = sorted(candidate_indices[int(index)] for index in selection)
    records = [dataset[index] for index in candidate_indices]
    reference = _reference(records)

    model = MultipathDiffusionModel(
        config["multipath_diffusion"],
        SystemConfig.from_mapping(config["data"]["system"]),
        checkpoint["statistics"],
    ).to(device)
    model.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))
    model.eval()
    sampling = config["sampling"]
    sampler = args.sampler or sampling["sampler"]
    sampling_steps = args.sampling_steps or int(sampling["sampling_steps"])
    guidance = (
        args.guidance_scale
        if args.guidance_scale is not None
        else float(sampling["guidance_scale"])
    )
    count_temperature = (
        args.path_count_temperature
        if args.path_count_temperature is not None
        else float(sampling.get("path_count_temperature", 1.0))
    )
    cuda_generator = torch.Generator(device=device).manual_seed(args.seed)

    condition_distance: torch.Tensor
    if args.protocol == "fixed_region":
        values = reference["conditions"].float()
        median = values.median(0).values
        scale = values.std(0, unbiased=False).clamp_min(1e-6)
        validity = reference["condition_validity"].bool()
        comparable = validity & validity.any(0).unsqueeze(0)
        score = torch.where(
            comparable, (values - median) / scale, torch.zeros_like(values)
        ).square().sum(-1)
        center_index = int(score.argmin())
        target = values[center_index]
        target_validity = validity[center_index]
        compatible = (values[:, 3] >= 0.5) == (target[3] >= 0.5)
        if bool(target_validity[2]):
            compatible &= validity[:, 2]
        distance = torch.where(
            validity & target_validity.unsqueeze(0),
            (values - target) / scale,
            torch.zeros_like(values),
        ).square().sum(-1).sqrt()
        compatible_indices = torch.nonzero(compatible, as_tuple=False).flatten()
        order = compatible_indices[distance[compatible].argsort()]
        reference = {key: value[order] for key, value in reference.items()}
        condition_distance = distance[order]
        target_conditions = target.unsqueeze(0).expand(order.numel(), -1)
        target_validities = target_validity.unsqueeze(0).expand(order.numel(), -1)
    else:
        target_conditions = reference["conditions"].float()
        target_validities = reference["condition_validity"].bool()
        condition_distance = torch.zeros(target_conditions.shape[0])

    generated_chunks: list[dict[str, torch.Tensor]] = []
    for start in range(0, target_conditions.shape[0], args.batch_size):
        stop = min(start + args.batch_size, target_conditions.shape[0])
        forced = (
            reference["path_count"][start:stop].to(device)
            if args.force_reference_path_count
            else None
        )
        generated = model.generate(
            target_conditions[start:stop].to(device),
            stop - start,
            condition_validity=target_validities[start:stop].to(device),
            sampler=sampler,
            sampling_steps=sampling_steps,
            eta=float(sampling.get("eta", 0.0)),
            guidance_scale=guidance,
            path_count_temperature=count_temperature,
            forced_path_count=forced,
            generator=cuda_generator,
        )
        generated_chunks.append(_cpu(generated))
    generated = _concatenate(generated_chunks)
    experiment_summary = (
        {key: value for key, value in plan.items() if key != "indices"}
        if plan is not None
        else None
    )

    report = evaluate_generated_payload("multipath_diffusion", generated, reference)
    report.update(
        {
            "protocol": args.protocol,
            "experiment": experiment_summary,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "reference_split": "test",
            "reference_samples": int(reference["csi"].shape[0]),
            "condition_distance": {
                "mean": float(condition_distance.mean()),
                "median": float(condition_distance.median()),
                "maximum": float(condition_distance.max()),
            },
            "paired_nmse": {
                "status": "not_computed",
                "reason": "conditional samples are distributional draws, not paired reconstructions",
            },
            "path_count": _path_count_report(
                generated["path_count"],
                reference["path_count"],
                model.max_paths,
                paired=args.protocol == "per_condition",
            ),
            "sampling": {
                "sampler": sampler,
                "sampling_steps": sampling_steps,
                "guidance_scale": guidance,
                "path_count_temperature": count_temperature,
                "force_reference_path_count": args.force_reference_path_count,
            },
        }
    )

    strata: dict[str, Any] = {}
    for label, keep in (
        ("los", reference["conditions"][:, 3] >= 0.5),
        ("nlos", reference["conditions"][:, 3] < 0.5),
    ):
        if int(keep.sum()) >= 8:
            strata[label] = evaluate_generated_payload(
                "multipath_diffusion", _subset(generated, keep), _subset(reference, keep)
            )
    for count in range(1, model.max_paths + 1):
        keep = reference["path_count"] == count
        if int(keep.sum()) >= 8:
            strata[f"reference_path_count_{count}"] = evaluate_generated_payload(
                "multipath_diffusion", _subset(generated, keep), _subset(reference, keep)
            )
    report["stratified"] = strata

    atomic_json_dump(report, args.output)
    if args.generated_output:
        output = Path(args.generated_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 3,
                "model_kind": "multipath_diffusion",
                "generated": generated,
                "protocol": args.protocol,
                "experiment": experiment_summary,
                "checkpoint": str(Path(args.checkpoint).resolve()),
            },
            output,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
