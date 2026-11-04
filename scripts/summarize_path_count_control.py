#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


KEY_METRICS = (
    "path_delay_power_swd",
    "link_delay_power_swd",
    "path_power_aod_swd",
    "link_power_aod_swd",
    "pdp_swd",
    "tx_spatial_correlation_relative_frobenius",
    "phase_invariant_full_csi_projected_swd",
    "csi_diversity_ratio_to_real",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a path-count control report.")
    parser.add_argument("report")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    summary: dict[str, Any] = {
        "experiment": report["experiment"],
        "artifact_samples": report["artifact_samples"],
        "max_paths": report["max_paths"],
        "per_l": {},
    }
    train = report["support_audit"].get("train", {}).get("per_l", {})
    test = report["support_audit"].get("test", {}).get("per_l", {})
    for path_count, section in report["per_l"].items():
        item: dict[str, Any] = {
            "status": section["status"],
            "train_support": train.get(path_count, {}).get("samples"),
            "test_support": test.get(path_count, {}).get("samples"),
            "evaluated_samples": section["evaluated_samples"],
        }
        if section["status"] == "evaluated":
            item["control"] = {
                model: {
                    "path_count_accuracy": values["path_count_accuracy"],
                    "finite_csi_rate": values["finite_csi_rate"],
                    **values["condition_accuracy"],
                }
                for model, values in section["control"].items()
            }
            item["fidelity_to_real"] = {
                model: {metric: values[metric] for metric in KEY_METRICS}
                for model, values in section["fidelity_to_real"].items()
            }
            improvement = section["current_relative_improvement_over_b3"]
            item["current_relative_improvement_over_b3"] = {
                metric: improvement.get(metric)
                for metric in KEY_METRICS
                if metric != "csi_diversity_ratio_to_real"
            }
            floor = section.get("real_vs_real_finite_sample_floor")
            item["real_vs_real_finite_sample_floor"] = (
                {metric: floor[metric] for metric in KEY_METRICS if metric in floor}
                if floor
                else None
            )
        summary["per_l"][path_count] = item
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
