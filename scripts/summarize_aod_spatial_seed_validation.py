#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TX_METRIC = "tx_spatial_correlation_relative_frobenius"


def _compact_target(values: dict[str, Any]) -> dict[str, Any]:
    delta = values["tx_spatial_absolute_error_change"]
    support = values["rayverse_support_rate"]
    rotation = values["rotation_abs_mean_deg"]
    return {
        "maps": values["maps"],
        "map_fraction_degraded": values["map_fraction_degraded"],
        "absolute_error_change_map_cluster_mean": delta.get("map_cluster_mean"),
        "absolute_error_change_bootstrap_95_ci": delta.get(
            "bootstrap_95_percentile_ci"
        ),
        "rayverse_support_rate_map_cluster_mean": support.get("map_cluster_mean"),
        "rotation_abs_mean_deg_map_cluster_mean": rotation.get("map_cluster_mean"),
    }


def _compact_map(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: values[key]
        for key in (
            "map_id",
            "generation_seeds",
            "angles_evaluated",
            "mean_absolute_error_change",
            "maximum_absolute_error_increase",
            "fraction_rows_degraded",
            "mean_rayverse_support_rate",
            "per_target_angle_absolute_error_change",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Map-angle localization and generation-seed stability."
    )
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--localization", required=True)
    args = parser.parse_args()
    aggregate = json.loads(Path(args.aggregate).read_text(encoding="utf-8"))
    localization = json.loads(Path(args.localization).read_text(encoding="utf-8"))

    overall: dict[str, Any] = {}
    for name, scope in aggregate["scope_summaries"].items():
        if not name.startswith("overall|"):
            continue
        metric = scope["metrics"][TX_METRIC]
        overall[name] = {
            "maps_seen": scope["maps_seen"],
            "maps_with_fidelity": scope["maps_with_fidelity"],
            "fidelity_map_coverage": scope["fidelity_map_coverage"],
            "exact_minus_realism_relative": metric[
                "exact_minus_realism_relative"
            ],
            "exact_control_win_rate": metric["exact_control_win_rate"],
            "nondegradation_gate": metric["nondegradation_gate"],
        }
    target_summaries = {
        name: _compact_target(values)
        for name, values in localization["target_angle_summaries"].items()
    }
    seed_consistency = localization["generation_seed_direction_consistency"]
    print(
        json.dumps(
            {
                "experiment": localization["experiment"],
                "unique_b3_cache_generation_seeds": localization[
                    "unique_b3_cache_generation_seeds"
                ],
                "overall_tx_spatial_result": overall,
                "per_target_angle": target_summaries,
                "los_maps_ranked_by_mean_degradation": [
                    _compact_map(values)
                    for values in localization["los_map_summaries"]
                ],
                "generation_seed_direction_consistency": {
                    "minimum_distinct_generation_seeds": seed_consistency[
                        "minimum_distinct_generation_seeds"
                    ],
                    "comparable_groups": seed_consistency["comparable_groups"],
                    "nlos": seed_consistency["nlos"],
                    "los": seed_consistency["los"],
                },
                "worst_cases_are_in": "spatial_failure_localization.json:worst_map_angle_cases",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
