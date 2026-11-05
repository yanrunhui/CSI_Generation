#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = (
    "path_power_aod_swd",
    "link_power_aod_swd",
    "tx_spatial_correlation_relative_frobenius",
    "phase_invariant_full_csi_projected_swd",
    "beamforming_gain_swd",
    "csi_diversity_ratio_to_real",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AoD tolerance tradeoff.")
    parser.add_argument("report")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item in report["rows"]:
        row: dict[str, Any] = {
            key: item[key]
            for key in (
                "map_id",
                "L",
                "los_status",
                "mode",
                "target_az_deg",
                "tolerance_deg",
                "stratum_samples",
                "real_support",
                "natural_accepted_samples",
                "natural_acceptance_rate",
                "status",
            )
        }
        for multiplier, values in item["candidate_pool_statistics"].items():
            row[f"multiplier_{multiplier}_status"] = values["status"]
            row[f"multiplier_{multiplier}_fill_probability"] = values.get(
                "probability_of_filling_request"
            )
            row[f"multiplier_{multiplier}_mean_accepted"] = values.get(
                "mean_accepted_samples"
            )
        if item["status"] == "fidelity_evaluated":
            row["matched_samples"] = item["matched_samples"]
            row["selected_error_mean_deg"] = item["selected_natural_error_deg"][
                "mean"
            ]
            for metric in METRICS:
                row[f"realism_first_{metric}"] = item["realism_first_vs_real"].get(
                    metric
                )
                row[f"exact_control_{metric}"] = item["exact_control_vs_real"].get(
                    metric
                )
        rows.append(row)
    print(
        json.dumps(
            {
                "experiment": report["experiment"],
                "requested_samples_per_pool": report["requested_samples_per_pool"],
                "pool_trials": report["pool_trials"],
                "candidate_multipliers": report["candidate_multipliers"],
                "tolerances_deg": report["tolerances_deg"],
                "rows": rows,
                "interpretation": report["interpretation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
