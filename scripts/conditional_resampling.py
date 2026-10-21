#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.baselines.conditional_resampling import ConditionalResampler
from physics_csi.config import load_project_config
from physics_csi.data.dataset import PreparedShardDataset


FIELDS = (
    "csi",
    "path_count",
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


def reservoir(dataset: PreparedShardDataset, count: int, seed: int) -> dict[str, torch.Tensor]:
    if count <= 0 or len(dataset) == 0:
        raise ValueError("Training split and reservoir size must both be non-empty")
    rng = random.Random(seed)
    selected: list[dict] = []
    for index in range(len(dataset)):
        row = dataset[index]
        if len(selected) < count:
            selected.append(row)
            continue
        replacement = rng.randint(0, index)
        if replacement < count:
            selected[replacement] = row
    return {
        key: torch.stack([row[key] for row in selected])
        for key in FIELDS
        if key in selected[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Training-only conditional resampling baseline.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=["nearest", "threshold", "kernel"], default="kernel")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--max-reference-samples", type=int, default=50000)
    parser.add_argument("--total-power-db", type=float, required=True)
    parser.add_argument("--delay-spread-ns", type=float, required=True)
    parser.add_argument("--k-factor-db", type=float, default=0.0)
    parser.add_argument("--los-status", type=int, choices=[0, 1], required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    config = load_project_config(args.config_dir)
    train = PreparedShardDataset(config["data"]["prepared_root"], "train")
    reference = reservoir(train, min(args.max_reference_samples, len(train)), args.seed)
    baseline = ConditionalResampler(reference)
    target = torch.tensor(
        [args.total_power_db, args.delay_spread_ns, args.k_factor_db, args.los_status]
    )
    result = baseline.sample(
        target,
        args.num_samples,
        method=args.method,
        threshold=torch.tensor(config["evaluation"]["resampling"]["threshold"]),
        bandwidth=torch.tensor(config["evaluation"]["resampling"]["kernel_bandwidth"]),
        generator=torch.Generator().manual_seed(args.seed),
    )
    generated = dict(result.samples)
    generated["target_conditions"] = target.unsqueeze(0).expand(args.num_samples, -1)
    generated["target_condition_validity"] = torch.ones(
        args.num_samples, 4, dtype=torch.bool
    )
    generated["target_condition_validity"][:, 2] = args.los_status == 1
    generated["actual_conditions"] = generated["conditions"]
    generated["actual_condition_validity"] = generated["condition_validity"]
    generated["condition_distance"] = result.condition_distance
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "baseline": f"conditional_resampling:{args.method}",
            "reference_split": "train",
            "reference_reservoir_size": int(reference["conditions"].shape[0]),
            "generated": generated,
            "conditions": dict(zip(PhysicsConditionedFields, target.tolist())),
            "seed": args.seed,
        },
        output,
    )


PhysicsConditionedFields = (
    "total_power_db",
    "delay_spread_ns",
    "k_factor_db",
    "los_status",
)


if __name__ == "__main__":
    main()

