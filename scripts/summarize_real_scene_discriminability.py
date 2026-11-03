#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


CORE = (
    "raw_full_csi_mmd_rbf",
    "raw_full_csi_sliced_wasserstein",
    "phase_invariant_full_csi_mmd_rbf",
    "phase_invariant_full_csi_sliced_wasserstein",
    "power_aod_path_sliced_wasserstein",
    "power_aod_link_sliced_wasserstein",
    "tx_spatial_correlation_relative_frobenius",
    "tx_spatial_correlation_magnitude_mae",
    "tx_spatial_correlation_phase_mae_rad",
)


def _compact(scope: Mapping[str, Any]) -> dict[str, Any]:
    if "status" in scope:
        return dict(scope)
    return {
        metric: {
            key: values[key]
            for key in (
                "same_map_mean",
                "cross_map_mean",
                "delta_scene_cross_minus_same",
                "cross_to_same_ratio",
                "map_pair_win_rate",
                "repeat_win_rate",
                "delta_scene_cluster_bootstrap_95_percent_ci",
                "delta_scene_bootstrap_95_percent_ci",
                "cluster_bootstrap_two_sided_p",
                "evidence",
            )
            if key in values
        }
        for metric, values in scope.items()
        if metric in CORE
    }


def _counts(scope: Mapping[str, Any]) -> dict[str, int]:
    values = [value for key, value in scope.items() if key in CORE]
    return {
        "positive_scene_discriminability": sum(
            value.get("evidence") == "positive_scene_discriminability"
            for value in values
        ),
        "inconclusive": sum(value.get("evidence") == "inconclusive" for value in values),
        "negative": sum(value.get("evidence") == "negative" for value in values),
        "metrics": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    args = parser.parse_args()
    with Path(args.report).open(encoding="utf-8") as stream:
        report = json.load(stream)
    summary = report["summary"]
    output = {
        "samples_in_split": report["samples_in_split"],
        "maps": report["maps"],
        "reported_map_pairs": report["reported_map_pairs"],
        "excluded_map_pairs": len(report["excluded_map_pairs"]),
        "matching_audit": report["matching"]["observed"],
        "phase_invariance_validation": report["phase_invariance_validation"],
        "map_10_vs_map_25": _compact(summary["specific_pair"]),
        "map_10_vs_other_maps": _compact(summary["anchor_map_vs_others"]),
        "all_reported_map_pairs": _compact(summary["all_reported_map_pairs"]),
        "evidence_counts": {
            "map_10_vs_map_25": _counts(summary["specific_pair"]),
            "map_10_vs_other_maps": _counts(summary["anchor_map_vs_others"]),
            "all_reported_map_pairs": _counts(summary["all_reported_map_pairs"]),
        },
        "decision_rule": {
            "result_A": (
                "Positive confidence intervals for phase-invariant Full CSI, Power-AoD or "
                "complete Tx correlation show detectable real scene structure."
            ),
            "result_B": (
                "Intervals crossing zero across these metrics mean current conditions and "
                "metrics do not reliably distinguish real Maps."
            ),
            "sign": "delta_scene = D_cross - D_same; positive means cross-Map is farther.",
        },
    }
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
