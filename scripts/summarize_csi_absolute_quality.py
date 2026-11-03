#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SELECTED = (
    "phase_invariant_full_csi_projected_mmd_rbf",
    "phase_invariant_full_csi_projected_sliced_wasserstein",
    "raw_full_csi_projected_mmd_rbf",
    "raw_full_csi_projected_sliced_wasserstein",
    "path_level_delay_power_sliced_wasserstein",
    "link_level_delay_power_sliced_wasserstein",
    "path_level_power_aod_sliced_wasserstein",
    "link_level_power_aod_sliced_wasserstein",
    "power_weighted_aod_sliced_wasserstein",
    "pdp_sliced_wasserstein",
    "tx_spatial_correlation_relative_frobenius",
    "tx_spatial_correlation_magnitude_mae",
    "tx_spatial_correlation_phase_mae_rad",
    "frequency_correlation_relative_l2",
    "frequency_correlation_magnitude_mae",
    "frequency_correlation_phase_mae_rad",
    "phase_invariant_csi_precision",
    "phase_invariant_csi_recall",
    "phase_invariant_csi_coverage",
    "phase_invariant_csi_rare_reference_coverage",
    "multipath_precision",
    "multipath_recall",
    "multipath_coverage",
    "multipath_rare_reference_coverage",
)


def _select(values: dict[str, Any]) -> dict[str, Any]:
    return {metric: values[metric] for metric in SELECTED}


def _compact_floor(values: dict[str, Any]) -> dict[str, Any]:
    return {
        metric: {
            "model_mean": values[metric]["model_mean"],
            "real_real_mean": values[metric]["real_real_mean"],
            "model_to_real_real_ratio": values[metric]["model_to_real_real_ratio"],
            "oriented_gap_model_worse_if_positive": values[metric]["oriented_gap_model_worse_if_positive"],
            "bootstrap_95_percent_ci": values[metric]["bootstrap_95_percent_ci"],
            "status": values[metric]["status"],
        }
        for metric in SELECTED
    }


def _compact_comparison(values: dict[str, Any]) -> dict[str, Any]:
    return {
        metric: {
            "relative_mean_improvement": values[metric].get("relative_mean_improvement"),
            "bootstrap_95_percent_ci": values[metric]["bootstrap_95_percent_ci"],
            "win_rate_excluding_ties": values[metric].get("win_rate_excluding_ties"),
            "two_sided_sign_test_p": values[metric].get("two_sided_sign_test_p"),
            "evidence": values[metric]["evidence"],
        }
        for metric in SELECTED
    }


def _scope(values: dict[str, Any]) -> dict[str, Any]:
    if not values:
        return {"status": "unavailable_no_local_condition_groups"}
    return {
        "b3_vs_real_real_floor": _compact_floor(values["b3_vs_real_real_floor"]),
        "current_vs_real_real_floor": _compact_floor(values["current_vs_real_real_floor"]),
        "current_vs_b3": _compact_comparison(values["current_vs_b3"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize absolute CSI generation quality.")
    parser.add_argument("report")
    args = parser.parse_args()
    report = json.load(Path(args.report).open("r", encoding="utf-8"))
    output = {
        "model": report["model"],
        "samples_per_side": report["samples_per_side"],
        "maps_evaluated": report["maps_evaluated"],
        "pairing_audit": report["pairing_audit"],
        "overall": {label: _select(values) for label, values in report["overall"].items()},
        "per_map": _scope(report["per_map_summary"]),
        "local_condition": _scope(report["local_condition_summary"]),
        "interpretation": report["interpretation"],
        "cautions": report["cautions"],
    }
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
