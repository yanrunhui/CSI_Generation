#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


CORE = (
    "path_distribution.aod_direction_sliced_wasserstein",
    "path_distribution.power_aod_joint_sliced_wasserstein",
    "path_distribution.power_aod_link_path_count_conditioned_sliced_wasserstein",
    "path_distribution.delay_power_joint_sliced_wasserstein",
    "path_distribution.delay_power_link_path_count_conditioned_sliced_wasserstein",
    "path_distribution.delay_power_aod_joint_sliced_wasserstein",
    "path_distribution.delay_power_aod_link_path_count_conditioned_sliced_wasserstein",
    "path_distribution.full_multipath_path_count_conditioned_mmd_rbf",
    "path_distribution.full_multipath_path_count_conditioned_sliced_wasserstein",
    "csi_distribution.full_csi_projected_mmd_rbf",
    "csi_distribution.full_csi_projected_sliced_wasserstein",
    "csi_distribution.csi_summary_mmd_rbf",
    "csi_distribution.csi_summary_sliced_wasserstein",
    "csi_distribution.pdp_sliced_wasserstein",
)


def _short(path: str) -> str:
    return path.split(".")[-1]


def _comparison_rows(comparison: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "metric": _short(path),
            "b3_scene_id": comparison["metrics"][path]["b3_scene_id"],
            "b3": comparison["metrics"][path]["b3"],
            "b3_minus_scene": comparison["metrics"][path][
                "improvement_b3_minus_b3_scene_id"
            ],
            "winner": comparison["metrics"][path]["winner"],
        }
        for path in CORE
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the compact A/B/C result from a scene joint-decomposition report."
    )
    parser.add_argument("report")
    args = parser.parse_args()
    with Path(args.report).open("r", encoding="utf-8") as stream:
        report = json.load(stream)

    density: dict[str, Any] = {}
    for name, group in report["B_condition_density"]["groups"].items():
        if group.get("status") != "ok":
            density[name] = {"status": group.get("status"), "support": group["support"]}
            continue
        density[name] = {
            "support": group["support"],
            "pooled_comparison": _comparison_rows(group["pooled_comparison"]),
            "paired_maps": len(group["paired_map_analysis"]["maps_used"]),
        }

    repairs: dict[str, Any] = {}
    for name, repair in report["C_counterfactual_repairs"]["repairs"].items():
        repairs[name] = {
            "metric_sanity": repair["metric_sanity"],
            "improvements": {
                _short(path): repair["improvement_control_minus_repaired"][path]
                for path in CORE
            },
            "projection_effect": repair["constraint_projection_change"],
            "total_treatment_change": repair[
                "total_treatment_change_from_projection_only_control"
            ],
            "delay_order_audit": {
                "after_replacement_before_canonicalization": repair[
                    "integrity_after_replacement_before_canonicalization"
                ]["delay_order_violation_samples"],
                "before_projection_after_canonicalization": repair[
                    "integrity_before_projection"
                ]["delay_order_violation_samples"],
                "after_projection": repair["integrity_after_projection"][
                    "delay_order_violation_samples"
                ],
            },
        }

    summary = {
        "samples": report["samples"],
        "A_overall_b3_vs_b3_scene_id": _comparison_rows(
            report["A_distribution_decomposition"]["comparison"]
        ),
        "A_metric_sanity": report["A_distribution_decomposition"]["metric_sanity"],
        "B_condition_density": density,
        "C_matching": report["C_counterfactual_repairs"]["matching"],
        "C_projection_only_control": {
            key: report["C_counterfactual_repairs"]["projection_only_control"][key]
            for key in (
                "metric_sanity",
                "difference_unrepaired_minus_control",
                "constraint_feasible_rate",
                "condition_consistency",
                "constraint_projection_change",
            )
        },
        "C_positive_repair_improvement_is_better": repairs,
        "interpretation_guardrail": report["C_counterfactual_repairs"]["interpretation"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
