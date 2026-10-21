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

from physics_csi.config import load_project_config
from physics_csi.data.conditional_experiments import (
    load_condition_experiment_plan,
    make_experiment_datasets,
)
from physics_csi.evaluation.diffusion_comparison import evaluate_generated_payload
from physics_csi.physics.constraints import PhysicsConstraintLayer
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig
from physics_csi.utils.io import atomic_json_dump


PATH_KEYS = (
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
    "direct_path_mask",
    "conditions",
    "condition_validity",
)


def _stack(records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        key: torch.stack([record[key] for record in records])
        for key in PATH_KEYS
        if key in records[0]
    }


def _sample_sources(
    reference_conditions: torch.Tensor,
    reference_validity: torch.Tensor,
    targets: torch.Tensor,
    target_validity: torch.Tensor,
    bandwidth: torch.Tensor,
    generator: torch.Generator,
    batch_size: int = 128,
) -> torch.Tensor:
    selected: list[torch.Tensor] = []
    for start in range(0, targets.shape[0], batch_size):
        stop = min(start + batch_size, targets.shape[0])
        target = targets[start:stop]
        validity = target_validity[start:stop]
        comparable = reference_validity.unsqueeze(0) & validity.unsqueeze(1)
        difference = (
            reference_conditions.unsqueeze(0) - target.unsqueeze(1)
        ) / bandwidth.reshape(1, 1, -1).clamp_min(1e-6)
        difference = torch.where(comparable, difference, torch.zeros_like(difference))
        log_weight = -0.5 * difference.square().sum(-1)
        los_match = (
            (reference_conditions[:, 3] >= 0.5).unsqueeze(0)
            == (target[:, 3] >= 0.5).unsqueeze(1)
        )
        eligible = los_match
        probabilities = torch.softmax(log_weight.masked_fill(~eligible, -torch.inf), dim=-1)
        if not torch.isfinite(probabilities).all():
            raise ValueError("conditional baseline has degenerate kernel weights")
        selected.append(
            torch.multinomial(probabilities, 1, replacement=True, generator=generator).squeeze(-1)
        )
    return torch.cat(selected)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate conditional resampling/statistical baselines under an experiment plan."
    )
    parser.add_argument("--experiment-plan", required=True)
    parser.add_argument("--baseline", choices=["resampling", "statistical"], required=True)
    parser.add_argument("--protocol", choices=["per_condition", "fixed_region"], required=True)
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--max-train-reference", type=int, default=50000)
    parser.add_argument("--angle-jitter-std-deg", type=float, default=2.0)
    parser.add_argument("--delay-jitter-std-ns", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_project_config(args.config_dir)
    plan = load_condition_experiment_plan(args.experiment_plan)
    prepared_root = plan["prepared_root"]
    datasets = make_experiment_datasets(prepared_root, plan)
    generator = torch.Generator().manual_seed(args.seed)

    test_count = min(len(datasets["test"]), args.max_samples)
    test_order = torch.randperm(len(datasets["test"]), generator=generator)[:test_count]
    test_order = test_order.sort().values.tolist()
    test_records = [datasets["test"][index] for index in test_order]
    reference = _stack(test_records)

    if args.protocol == "fixed_region":
        values = reference["conditions"].float()
        target = values.median(0).values
        scale = values.std(0, unbiased=False).clamp_min(1e-6)
        center = ((values - target) / scale).square().sum(-1).argmin()
        target = values[center]
        validity = reference["condition_validity"][center]
        compatible = (values[:, 3] >= 0.5) == (target[3] >= 0.5)
        if bool(validity[2]):
            compatible &= reference["condition_validity"][:, 2]
        distance = ((values - target) / scale).square().sum(-1)
        compatible_indices = torch.nonzero(compatible, as_tuple=False).flatten()
        order = compatible_indices[distance[compatible].argsort()]
        reference = {key: value[order] for key, value in reference.items()}
        test_count = int(order.numel())
        targets = target.unsqueeze(0).expand(test_count, -1)
        target_validity = validity.unsqueeze(0).expand(test_count, -1)
    else:
        targets = reference["conditions"].float()
        target_validity = reference["condition_validity"].bool()

    train_count = min(len(datasets["train"]), args.max_train_reference)
    train_order = torch.randperm(len(datasets["train"]), generator=generator)[:train_count]
    train_order = train_order.sort().values.tolist()
    train_conditions: list[torch.Tensor] = []
    train_validity: list[torch.Tensor] = []
    for index in train_order:
        record = datasets["train"][index]
        train_conditions.append(record["conditions"])
        train_validity.append(record["condition_validity"])
    train_conditions_tensor = torch.stack(train_conditions).float()
    train_validity_tensor = torch.stack(train_validity).bool()
    bandwidth = torch.tensor(config["evaluation"]["resampling"]["kernel_bandwidth"])
    selected_local = _sample_sources(
        train_conditions_tensor,
        train_validity_tensor,
        targets,
        target_validity,
        bandwidth,
        generator,
    )
    selected_records = [datasets["train"][train_order[int(index)]] for index in selected_local]
    generated = _stack(selected_records)
    generated["target_conditions"] = targets
    generated["target_condition_validity"] = target_validity

    if args.baseline == "resampling":
        generated["actual_conditions"] = generated.pop("conditions")
        generated["actual_condition_validity"] = generated.pop("condition_validity")
    else:
        system = SystemConfig.from_mapping(config["data"]["system"])
        constraints = PhysicsConstraintLayer(
            **dict(config["model"]["constraints"])
        )
        mask = generated["path_mask"].bool()
        delays = (
            generated["path_delays_ns"]
            + torch.randn(generated["path_delays_ns"].shape, generator=generator)
            * args.delay_jitter_std_ns
        ).clamp_min(0.0)
        for key in ("aoa_az_deg", "aoa_el_deg", "aod_az_deg", "aod_el_deg"):
            generated[key] = generated[key] + torch.randn(
                generated[key].shape, generator=generator
            ) * args.angle_jitter_std_deg
        generated["path_phases_rad"] = torch.remainder(
            generated["path_phases_rad"]
            + 2.0 * torch.pi * torch.rand(generated["path_phases_rad"].shape, generator=generator)
            + torch.pi,
            2.0 * torch.pi,
        ) - torch.pi
        constrained = constraints(
            delays,
            generated["path_powers_linear"],
            mask,
            targets,
            target_condition_validity=target_validity,
        )
        generated["path_delays_ns"] = constrained.delays_ns
        generated["path_powers_linear"] = constrained.powers_linear
        generated["direct_path_mask"] = constrained.direct_path_mask
        generated["actual_conditions"] = constrained.actual_conditions
        generated["actual_condition_validity"] = constrained.condition_validity
        generated["constraint_feasible"] = constrained.feasible
        generated["csi"] = MIMOOFDMSynthesizer(system).eval()(
            generated["path_delays_ns"],
            generated["path_powers_linear"],
            generated["path_phases_rad"],
            generated["aoa_az_deg"],
            generated["aoa_el_deg"],
            generated["aod_az_deg"],
            generated["aod_el_deg"],
            mask,
        )
        generated.pop("conditions", None)
        generated.pop("condition_validity", None)

    report = evaluate_generated_payload("multipath_diffusion", generated, reference)
    experiment_summary = {key: value for key, value in plan.items() if key != "indices"}
    report.update(
        {
            "baseline": args.baseline,
            "protocol": args.protocol,
            "experiment": experiment_summary,
            "training_reference_samples": train_count,
            "reference_samples": test_count,
            "paired_nmse": {
                "status": "not_computed",
                "reason": "conditional distribution samples are not paired reconstructions",
            },
        }
    )
    atomic_json_dump(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
