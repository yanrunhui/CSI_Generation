#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Print AoD residual decision metrics.")
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

    result = {
        "samples": report["samples"],
        "maps_evaluated": report["maps_evaluated"],
        "correct_map_vs_dp_only": compact(report["summaries"]["correct_map_vs_dp_only"]),
        "correct_map_vs_shuffled_aod_map": compact(
            report["summaries"]["correct_map_vs_shuffled_map"]
        ),
        "correct_map_vs_b3": compact(
            report["summaries"]["correct_map_vs_physics_only_baseline"]
        ),
        "residual_audit": report["residual_audit"],
        "dp_preservation_maximum_absolute_error": report[
            "dp_preservation_maximum_absolute_error"
        ],
        "success_rule": report["success_rule"],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
