#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.data.dataset import PreparedShardDataset
from physics_csi.evaluation.diffusion_comparison import evaluate_generated_payload
from physics_csi.utils.io import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved diffusion generation artifact.")
    parser.add_argument("--generated", required=True)
    parser.add_argument("--prepared-root", default="artifacts/prepared_d2los")
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--max-reference-samples", type=int, default=4096)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    artifact = torch.load(args.generated, map_location="cpu", weights_only=False)
    dataset = PreparedShardDataset(args.prepared_root, args.split)
    count = min(len(dataset), args.max_reference_samples)
    records = [dataset[index] for index in range(count)]
    reference_csi = torch.stack([record["csi"] for record in records])
    reference_conditions = torch.stack([record["conditions"] for record in records]).float()
    reference_validity = torch.stack([record["condition_validity"] for record in records]).bool()
    target = artifact["generated"]["target_conditions"].float().mean(0)
    target_validity = artifact["generated"]["target_condition_validity"].bool().all(0)
    scale = reference_conditions.std(0, unbiased=False).clamp_min(1e-6)
    usable = reference_validity & target_validity.unsqueeze(0)
    difference = torch.where(
        usable,
        (reference_conditions - target.unsqueeze(0)) / scale.unsqueeze(0),
        torch.zeros_like(reference_conditions),
    )
    score = difference[:, :3].square().sum(-1)
    score = score + 1000.0 * (
        (reference_conditions[:, 3] >= 0.5) != (target[3] >= 0.5)
    ).float()
    matched_count = min(reference_csi.shape[0], artifact["generated"]["csi"].shape[0])
    matched_indices = score.argsort()[:matched_count]
    tensor_keys = (
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
    reference = {
        key: torch.stack([records[int(index)][key] for index in matched_indices])
        for key in tensor_keys
        if key in records[0]
    }
    report = evaluate_generated_payload(
        artifact["model_kind"], artifact["generated"], reference
    )
    report["reference_split"] = args.split
    report["reference_candidates"] = count
    report["reference_samples"] = matched_count
    report["condition_matching"] = {
        "method": "nearest standardized physical conditions with exact LoS preference",
        "mean_standardized_distance": float(score[matched_indices].sqrt().mean()),
        "target": target.tolist(),
    }
    atomic_json_dump(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

