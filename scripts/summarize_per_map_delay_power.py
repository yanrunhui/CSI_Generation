#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


METRICS = (
    "path_level_delay_power_sliced_wasserstein",
    "link_level_delay_power_sliced_wasserstein",
    "power_weighted_delay_wasserstein",
)
SECTIONS = (
    "unconditional",
    "local_condition_weighted",
    "common_path_count_controlled",
)


def _compact_fidelity(summary: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for section in SECTIONS:
        result[section] = {}
        for metric in METRICS:
            values = summary[section][metric]
            if "status" in values:
                result[section][metric] = values
                continue
            result[section][metric] = {
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
    return result


def _map_view(per_map: Mapping[str, Any], map_id: int) -> dict[str, Any]:
    values = per_map.get(f"map_{map_id}")
    if values is None:
        return {"status": "map_not_present"}
    if values.get("status") != "ok":
        return dict(values)
    return {
        "samples": values["samples"],
        "support": values["support"],
        "models": {
            label: {
                "unconditional": model["unconditional"],
                "local_condition_weighted": model["local_condition_weighted"],
                "common_path_count_controlled": model[
                    "common_path_count_controlled"
                ],
                "path_count": model["path_count"],
            }
            for label, model in values["models"].items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--focus-map", type=int, default=10)
    args = parser.parse_args()
    with Path(args.report).open(encoding="utf-8") as stream:
        report = json.load(stream)
    real = report["real_to_real"]
    fidelity = report["model_fidelity"]
    output = {
        "normalization": report["normalization"],
        "real_to_real": {
            "samples": real["samples"],
            "maps": real["maps"],
            "requested_balanced_map_pairs": real["requested_balanced_map_pairs"],
            "reported_map_pairs": real["reported_map_pairs"],
            "excluded_map_pairs": len(real["excluded_map_pairs"]),
            "requested_pair_degree_minimum": real["requested_pair_degree_minimum"],
            "requested_pair_degree_maximum": real["requested_pair_degree_maximum"],
            "reported_pair_degree_minimum": real["reported_pair_degree_minimum"],
            "reported_pair_degree_maximum": real["reported_pair_degree_maximum"],
            "matching": real["matching"],
            "equivalence_margin": real["equivalence_margin"],
            "metrics": real["summary"],
        },
        "model_fidelity": {
            "samples": fidelity["samples"],
            "maps_total": fidelity["maps_total"],
            "maps_evaluated": fidelity["maps_evaluated"],
            "condition_input_audit": fidelity["condition_input_audit"],
            "support_summary": fidelity["support_summary"],
            "path_count_summary": fidelity["path_count_summary"],
            "b3_scene_id_vs_b3": _compact_fidelity(
                fidelity["summaries"]["b3_scene_id_vs_b3"]
            ),
            "b3_scene_id_vs_shuffled_map": _compact_fidelity(
                fidelity["summaries"]["b3_scene_id_vs_shuffled_map"]
            ),
            f"map_{args.focus_map}": _map_view(
                fidelity["per_map"], args.focus_map
            ),
        },
        "decision_rule": {
            "generic_delay_power_supported": (
                "All primary real-to-real relative-effect confidence intervals lie "
                "inside the equivalence margin, and B3 is not worse than B3-S-ID."
            ),
            "map_dependent_delay_power": (
                "A real-to-real relative-effect interval lies above the equivalence "
                "margin; retaining Map dependence in p(tau,P|c,m) is then necessary."
            ),
            "underpowered": (
                "An interval crossing zero but extending outside the equivalence margin "
                "is inconclusive, not evidence of equivalence."
            ),
        },
    }
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
