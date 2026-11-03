#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AoD residual mechanism diagnostics.")
    parser.add_argument("report")
    args = parser.parse_args()
    report = json.load(Path(args.report).open("r", encoding="utf-8"))

    def compact(section: dict[str, Any]) -> dict[str, Any]:
        return {
            metric: {
                "relative_mean_improvement": value.get("relative_mean_improvement"),
                "bootstrap_95_percent_ci": value.get("bootstrap_95_percent_ci"),
                "win_rate_excluding_ties": value.get("win_rate_excluding_ties"),
                "two_sided_sign_test_p": value.get("two_sided_sign_test_p"),
                "evidence": value.get("evidence"),
            }
            for metric, value in section.items()
        }

    structure = report["residual_structure"]
    result = {
        "samples": report["samples"],
        "maps_evaluated": report["maps_evaluated"],
        "minimum_samples_per_map": report["minimum_samples_per_map"],
        "azimuth_structure": {
            key: structure["azimuth"][key]
            for key in (
                "effective_limit_deg",
                "signed",
                "absolute",
                "positive_fraction",
                "negative_fraction",
                "fraction_at_least_95_percent_of_limit",
                "map_constant_offset_r2",
                "link_constant_offset_r2",
                "within_link_path_std_deg",
                "global_correlations",
                "map_centered_correlations",
            )
        },
        "elevation_structure": {
            key: structure["elevation"][key]
            for key in (
                "effective_limit_deg",
                "signed",
                "absolute",
                "positive_fraction",
                "negative_fraction",
                "fraction_at_least_95_percent_of_limit",
                "map_constant_offset_r2",
                "link_constant_offset_r2",
                "within_link_path_std_deg",
                "global_correlations",
                "map_centered_correlations",
            )
        },
        "full_vs_constant": compact(
            report["summaries"]["full_aod_residual_vs_constant_offset"]
        ),
        "constant_vs_dp_only": compact(
            report["summaries"]["constant_offset_vs_dp_only"]
        ),
        "full_vs_dp_only_independent": compact(
            report["summaries"]["full_aod_residual_vs_dp_only"]
        ),
        "constant_vs_shuffled_constant": compact(
            report["summaries"]["constant_offset_vs_shuffled_constant_offset"]
        ),
        "constant_offset_artifact": report["constant_offset_artifact"],
        "decision_rule": report["decision_rule"],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
