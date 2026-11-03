#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


METRICS = (
    "path_level_power_aod_sliced_wasserstein",
    "link_level_power_aod_sliced_wasserstein",
    "power_weighted_aod_sliced_wasserstein",
)
SECTIONS = (
    "unconditional",
    "local_condition_weighted",
    "common_path_count_controlled",
)


def _compact(summary: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for section in SECTIONS:
        output[section] = {}
        for metric in METRICS:
            values = summary[section][metric]
            if "status" in values:
                output[section][metric] = values
                continue
            output[section][metric] = {
                key: values[key]
                for key in (
                    "maps",
                    "correct_map_mean_distance",
                    "comparison_mean_distance",
                    "mean_improvement_comparison_minus_correct",
                    "relative_mean_improvement",
                    "bootstrap_95_percent_ci",
                    "correct_map_wins",
                    "correct_map_losses",
                    "win_rate_excluding_ties",
                    "two_sided_sign_test_p",
                    "evidence",
                )
                if key in values
            }
    return output


def _map_view(per_map: Mapping[str, Any], map_id: int) -> dict[str, Any]:
    values = per_map.get(f"map_{map_id}")
    if values is None:
        return {"status": "map_not_present"}
    if values.get("status") != "ok":
        return dict(values)
    result: dict[str, Any] = {
        "samples": values["samples"],
        "support": values["support"],
        "models": {},
    }
    for label, model in values["models"].items():
        result["models"][label] = {
            "unconditional": model["unconditional"],
            "local_condition_weighted": model["local_condition_weighted"],
            "common_path_count_controlled": model["common_path_count_controlled"],
            "path_count": {
                key: model["path_count"][key]
                for key in (
                    "exact_match_rate",
                    "mean_absolute_count_error",
                    "distribution_total_variation",
                )
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--focus-map", type=int, default=10)
    args = parser.parse_args()
    with Path(args.report).open(encoding="utf-8") as stream:
        report = json.load(stream)
    output = {
        "samples": report["samples"],
        "maps_total": report["maps_total"],
        "maps_evaluated": report["maps_evaluated"],
        "condition_input_audit": report["condition_input_audit"],
        "support_summary": report["support_summary"],
        "path_count_summary": report["path_count_summary"],
        "b3_scene_id_vs_b3": _compact(report["summaries"]["b3_scene_id_vs_b3"]),
        "b3_scene_id_vs_shuffled_map": _compact(
            report["summaries"]["b3_scene_id_vs_shuffled_map"]
        ),
        f"map_{args.focus_map}": _map_view(report["per_map"], args.focus_map),
        "decision_rule": {
            "positive_improvement": (
                "B3 or shuffled-Map distance minus B3-S-ID distance; positive favors correct Map ID."
            ),
            "result_A": (
                "B3-S-ID has positive Bootstrap intervals against both B3 and shuffled Map "
                "across path-, link- and power-weighted AoD metrics."
            ),
            "result_B": (
                "Intervals cross zero or favor B3/shuffled Map, indicating that correct Map "
                "conditioning has not learned stable per-Map Power-AoD fidelity."
            ),
        },
    }
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
