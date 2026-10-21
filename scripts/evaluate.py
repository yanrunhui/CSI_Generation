#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.config import load_project_config
from physics_csi.data.dataset import PreparedShardDataset
from physics_csi.evaluation.conditions import condition_accuracy, condition_tolerance_rates
from physics_csi.evaluation.distribution import distribution_metrics
from physics_csi.evaluation.diversity import diversity_metrics
from physics_csi.utils.io import atomic_json_dump


FIELDS = (
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


def collect_reference(dataset: PreparedShardDataset, target: torch.Tensor, count: int) -> dict:
    scales = torch.tensor([20.0, 100.0, 10.0, 1.0])
    best: list[tuple[float, int]] = []
    for index in range(len(dataset)):
        row = dataset[index]
        condition = row["conditions"]
        if bool((condition[3] >= 0.5) != (target[3] >= 0.5)):
            continue
        if target[3] >= 0.5 and not bool(row["condition_validity"][2]):
            continue
        comparable = row["condition_validity"].bool().clone()
        if target[3] < 0.5:
            comparable[2] = False
        difference = torch.where(
            comparable, (condition - target) / scales, torch.zeros_like(condition)
        )
        distance = float(difference.square().sum())
        candidate = (-distance, index)
        if len(best) < count:
            heapq.heappush(best, candidate)
        elif candidate > best[0]:
            heapq.heapreplace(best, candidate)
    indices = [index for _, index in sorted(best, reverse=True)]
    if not indices:
        raise ValueError("No test samples match the requested LoS/NLoS status")
    rows = [dataset[index] for index in indices]
    return {
        key: torch.stack([row[key] for row in rows])
        for key in FIELDS
        if key in rows[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated conditional CSI.")
    parser.add_argument("--generated", required=True)
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--output", default="artifacts/evaluation.json")
    args = parser.parse_args()
    config = load_project_config(args.config_dir)
    payload = torch.load(args.generated, map_location="cpu", weights_only=False)
    generated = payload["generated"]
    target = generated["target_conditions"]
    validity = generated["target_condition_validity"]
    reference_dataset = PreparedShardDataset(config["data"]["prepared_root"], "test")
    reference = collect_reference(reference_dataset, target[0], len(target))
    generated_eval = dict(generated)
    generated_eval["conditions"] = generated["actual_conditions"]
    metrics = {
        "condition_accuracy": condition_accuracy(
            target, generated["actual_conditions"], validity
        ),
        "condition_tolerance": condition_tolerance_rates(
            target,
            generated["actual_conditions"],
            validity,
            config["evaluation"]["condition_tolerances"],
        ),
        "distribution": distribution_metrics(
            generated_eval,
            reference,
            list(config["evaluation"]["mmd_kernel_bandwidths"]),
        ),
        "diversity": diversity_metrics(generated_eval, reference),
        "reference_split": "test",
        "reference_count": int(reference["path_mask"].shape[0]),
    }
    atomic_json_dump(metrics, args.output)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

