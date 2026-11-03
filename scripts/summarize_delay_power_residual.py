#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the DP residual decision metrics.")
    parser.add_argument("report")
    args = parser.parse_args()
    with Path(args.report).open("r", encoding="utf-8") as stream:
        report = json.load(stream)

    def compact(section: dict[str, Any]) -> dict[str, Any]:
        return {
            metric: {
                "relative_mean_improvement": values.get("relative_mean_improvement"),
                "bootstrap_95_percent_ci": values.get("bootstrap_95_percent_ci"),
                "win_rate_excluding_ties": values.get("win_rate_excluding_ties"),
                "two_sided_sign_test_p": values.get("two_sided_sign_test_p"),
                "evidence": values.get("evidence"),
            }
            for metric, values in section.items()
        }

    result = {
        "samples": report["samples"],
        "maps_evaluated": report["maps_evaluated"],
        "correct_map_vs_b3": compact(
            report["summaries"]["correct_map_vs_physics_only_baseline"]
        ),
        "correct_map_vs_shuffled_map": compact(
            report["summaries"]["correct_map_vs_shuffled_map"]
        ),
        "condition_audit": report["condition_audit"],
        "residual_audit": report["residual_audit"],
        "success_rule": report["success_rule"],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
