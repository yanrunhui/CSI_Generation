#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


KEY_METRICS = (
    "path_power_aod_swd",
    "link_power_aod_swd",
    "tx_spatial_correlation_relative_frobenius",
    "phase_invariant_full_csi_projected_swd",
    "beamforming_gain_swd",
)


def _compact_scope(scope: dict[str, Any]) -> dict[str, Any]:
    candidate_pool = {}
    for multiplier, values in scope["candidate_pool"].items():
        fill = values["fill_probability"]
        candidate_pool[multiplier] = {
            "maps": fill.get("maps", 0),
            "map_cluster_mean_fill_probability": fill.get("map_cluster_mean"),
            "bootstrap_95_percentile_ci": fill.get(
                "bootstrap_95_percentile_ci"
            ),
            "map_fraction_with_fill_probability_at_least_0_95": values[
                "map_fraction_with_fill_probability_at_least_0_95"
            ],
        }
    metrics = {}
    for metric in KEY_METRICS:
        values = scope["metrics"][metric]
        delta = values["exact_minus_realism_relative"]
        metrics[metric] = {
            "maps": delta.get("maps", 0),
            "exact_minus_realism_map_cluster_mean": delta.get(
                "map_cluster_mean"
            ),
            "bootstrap_95_percentile_ci": delta.get(
                "bootstrap_95_percentile_ci"
            ),
            "exact_control_win_rate": values["exact_control_win_rate"],
            "nondegradation_gate": values["nondegradation_gate"],
            "realism_first_excess_over_real_real": values[
                "realism_first_excess_over_real_real"
            ].get("map_cluster_mean"),
            "exact_control_excess_over_real_real": values[
                "exact_control_excess_over_real_real"
            ].get("map_cluster_mean"),
        }
    acceptance = scope["natural_acceptance_rate"]
    return {
        "maps_seen": scope["maps_seen"],
        "maps_with_fidelity": scope["maps_with_fidelity"],
        "fidelity_map_coverage": scope["fidelity_map_coverage"],
        "natural_acceptance_rate": {
            "map_cluster_mean": acceptance.get("map_cluster_mean"),
            "bootstrap_95_percentile_ci": acceptance.get(
                "bootstrap_95_percentile_ci"
            ),
        },
        "candidate_pool": candidate_pool,
        "exact_control_vs_realism_first": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compact summary for cross-Map, cross-angle AoD validation."
    )
    parser.add_argument("report")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    scopes = {
        name: _compact_scope(values)
        for name, values in report["scope_summaries"].items()
        if name.startswith("overall|")
    }
    print(
        json.dumps(
            {
                "experiment": report["experiment"],
                "report_count": report["report_count"],
                "unique_b3_cache_generation_seeds": report[
                    "unique_b3_cache_generation_seeds"
                ],
                "overall_scopes": scopes,
                "generation_seed_stability": report[
                    "generation_seed_stability"
                ],
                "decision_rule": report["decision_rule"],
                "angle_specific_results_are_in": "scope_summaries keys beginning with angle=",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
