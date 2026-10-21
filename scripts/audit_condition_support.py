#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.data.conditional_experiments import CONDITION_INDEX
from physics_csi.data.dataset import PreparedShardDataset
from physics_csi.utils.io import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit continuous-condition support before pre-registering ID/sparse/OOD regions."
    )
    parser.add_argument("--prepared-root", default="artifacts/prepared_d2los")
    parser.add_argument(
        "--field",
        choices=["total_power_db", "delay_spread_ns", "k_factor_db"],
        default="delay_spread_ns",
    )
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--los-status", type=int, choices=[0, 1], default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.bins < 2:
        parser.error("--bins must be at least 2")

    field_index = CONDITION_INDEX[args.field]
    payload: dict[str, dict[str, torch.Tensor]] = {}
    all_values: list[torch.Tensor] = []
    for split in ("train", "validation", "test"):
        dataset = PreparedShardDataset(args.prepared_root, split)
        values: list[float] = []
        counts: list[int] = []
        los_values: list[int] = []
        valid_values: list[bool] = []
        for index in range(len(dataset)):
            record = dataset[index]
            los = int(record["conditions"][3] >= 0.5)
            if args.los_status is not None and los != args.los_status:
                continue
            values.append(float(record["conditions"][field_index]))
            counts.append(int(record["path_count"]))
            los_values.append(los)
            valid_values.append(bool(record["condition_validity"][field_index]))
        split_values = torch.tensor(values)
        split_validity = torch.tensor(valid_values, dtype=torch.bool)
        payload[split] = {
            "values": split_values,
            "path_count": torch.tensor(counts),
            "los": torch.tensor(los_values),
            "validity": split_validity,
        }
        all_values.append(split_values[split_validity])
    combined = torch.cat(all_values)
    quantiles = torch.linspace(0, 1, args.bins + 1)
    edges = torch.unique(torch.quantile(combined, quantiles))
    if edges.numel() < 3:
        raise ValueError("condition has insufficient distinct values for interval experiments")

    bins: list[dict[str, object]] = []
    for index in range(edges.numel() - 1):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        row: dict[str, object] = {"lower": lower, "upper": upper}
        for split, values in payload.items():
            inside = values["validity"] & (values["values"] >= lower)
            inside &= values["values"] <= upper if index + 2 == edges.numel() else values["values"] < upper
            path_histogram = Counter(values["path_count"][inside].tolist())
            los_histogram = Counter(values["los"][inside].tolist())
            row[split] = {
                "count": int(inside.sum()),
                "path_count_histogram": {
                    str(key): value for key, value in sorted(path_histogram.items())
                },
                "los_histogram": {
                    str(key): value for key, value in sorted(los_histogram.items())
                },
            }
        bins.append(row)
    report = {
        "field": args.field,
        "los_filter": args.los_status,
        "binning": "pooled quantile edges; inspect support only and pre-register before training",
        "bins": bins,
    }
    atomic_json_dump(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
