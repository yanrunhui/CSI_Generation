#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


METRICS = (
    "path_power_aod_swd",
    "link_power_aod_swd",
    "tx_spatial_correlation_relative_frobenius",
    "path_delay_power_swd",
    "link_delay_power_swd",
    "pdp_swd",
    "phase_invariant_full_csi_projected_swd",
    "csi_diversity_ratio_to_real",
)


def _aggregate(section: dict) -> dict:
    return {
        alpha: {
            metric: {
                "mean": values[metric]["mean"],
                "std_across_generation_seeds": values[metric][
                    "std_across_generation_seeds"
                ],
            }
            for metric in METRICS
        }
        for alpha, values in section["aggregate"].items()
    }


def _macro_primary(section: dict) -> dict:
    return {
        alpha: {
            metric: {
                "mean": values[metric]["mean"],
                "std_across_generation_seeds": values[metric][
                    "std_across_generation_seeds"
                ],
            }
            for metric in (
                "path_power_aod_swd",
                "link_power_aod_swd",
                "tx_spatial_correlation_relative_frobenius",
            )
        }
        for alpha, values in section["macro_primary_aggregate"].items()
    }


def _control(section: dict) -> dict:
    result = {}
    alpha_keys = section["per_generation_seed"][0][
        "control_audit_on_pooled_subset"
    ]
    for alpha in alpha_keys:
        audits = [
            run["control_audit_on_pooled_subset"][alpha]
            for run in section["per_generation_seed"]
        ]
        accuracy = [item["condition_accuracy"] for item in audits]
        result[alpha] = {
            "minimum_path_count_accuracy": min(
                item["path_count_accuracy"] for item in audits
            ),
            "minimum_finite_csi_rate": min(item["finite_csi_rate"] for item in audits),
            "minimum_los_consistency": min(item["los_consistency"] for item in accuracy),
            "maximum_total_power_db_mae": max(
                item["total_power_db_mae"] for item in accuracy
            ),
            "maximum_delay_spread_ns_mae": max(
                item["delay_spread_ns_mae"] for item in accuracy
            ),
            "maximum_k_factor_db_mae": max(
                (item["k_factor_db_mae"] for item in accuracy if math.isfinite(item["k_factor_db_mae"])),
                default=None,
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a path-count gate report.")
    parser.add_argument("report")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    selected = str(report["validation"]["selection"]["selected_alpha5"])
    summary = {
        "experiment": report["experiment"],
        "generation_seeds": report["generation_seeds"],
        "selected_alpha5_on_validation": report["validation"]["selection"][
            "selected_alpha5"
        ],
        "selection_rule": report["validation"]["selection"]["rule"],
        "validation_candidates": report["validation"]["selection"]["candidates"],
        "validation_pooled_auxiliary_metrics": _aggregate(report["validation"]),
        "validation_macro_primary_metrics": _macro_primary(report["validation"]),
        "validation_all_candidates_vs_alpha1": report["validation"][
            "all_candidates_vs_alpha1_hierarchical_bootstrap"
        ],
        "validation_control_audit": _control(report["validation"]),
        "test_pooled_auxiliary_metrics": _aggregate(report["test"]),
        "test_macro_primary_metrics": _macro_primary(report["test"]),
        "test_control_audit": _control(report["test"]),
        "validation_selected_vs_alpha1": report["validation"][
            "selected_vs_alpha1_hierarchical_bootstrap"
        ],
        "test_selected_vs_alpha1": report["test"][
            "selected_vs_alpha1_hierarchical_bootstrap"
        ],
        "test_interpretation": (
            "Use the frozen selected alpha and the hierarchical seed-by-Map confidence intervals. "
            "Positive improvement favors the selected gate over alpha5=1."
        ),
        "non_target_l_audit": report["non_target_l_audit"],
    }
    if selected not in report["test"]["aggregate"]:
        # JSON renders integral floats as "0.0" while the evaluator uses compact keys.
        compact = f"{float(selected):.8g}"
        if compact not in report["test"]["aggregate"]:
            raise KeyError("selected alpha is missing from the test aggregate")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
