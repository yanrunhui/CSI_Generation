#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize dual-mode AoD control diagnostic.")
    parser.add_argument("report")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    physical: dict[str, dict[str, float | None]] = {}
    for mode in ("mean", "strongest"):
        records = [item for item in report["runs"] if item["mode"] == mode]
        physical[mode] = {}
        for metric in (
            "total_power_db_mae",
            "delay_spread_ns_mae",
            "k_factor_db_mae",
            "los_consistency",
        ):
            values = [
                item["condition_accuracy"][metric]
                for item in records
                if math.isfinite(item["condition_accuracy"][metric])
            ]
            physical[mode][f"maximum_{metric}"] = max(values) if values else None
    summary = {
        "experiment": report["experiment"],
        "map_id": report["map_id"],
        "generation_seeds": report["generation_seeds"],
        "target_angles_deg": report["target_angles_deg"],
        "samples_per_condition": report["samples_per_condition"],
        "aggregate": report["aggregate"],
        "physical_condition_audit": physical,
        "condition_support": report["condition_support"],
        "success_rule": report["success_rule"],
        "caution": report["caution"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
