#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _compact_floor(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_mean_error": value["model_mean_error"],
        "real_real_mean_error": value["real_real_mean_error"],
        "model_to_real_real_ratio": value["model_to_real_real_ratio"],
        "bootstrap_95_percent_ci": value["bootstrap_95_percent_ci"],
        "status": value["status"],
    }


def _compact_comparison(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "relative_mean_improvement": value.get("relative_mean_improvement"),
        "bootstrap_95_percent_ci": value["bootstrap_95_percent_ci"],
        "win_rate_excluding_ties": value.get("win_rate_excluding_ties"),
        "two_sided_sign_test_p": value.get("two_sided_sign_test_p"),
        "evidence": value["evidence"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize remaining CSI bottleneck diagnostics.")
    parser.add_argument("report")
    args = parser.parse_args()
    report = json.load(Path(args.report).open("r", encoding="utf-8"))
    decomposition = report["frequency_and_tx_decomposition"]
    frequency = {
        lag: {
            "current_vs_real_real_floor": _compact_floor(value["current_vs_real_real_floor"]),
            "current_vs_b3": _compact_comparison(value["current_vs_b3"]),
            "systematic_bias_fraction": {
                label: decomposition["frequency_systematic_bias"][label][lag]["systematic_bias_fraction"]
                for label in ("real_vs_real", "b3_vs_real", "current_vs_real")
            },
        }
        for lag, value in decomposition["frequency_per_lag_map_summary"].items()
    }
    spacing = {
        value["distance_wavelengths"]: {
            "antenna_pairs": value["antenna_pairs"],
            "current_vs_real_real_floor": _compact_floor(value["current_vs_real_real_floor"]),
            "current_vs_b3": _compact_comparison(value["current_vs_b3"]),
        }
        for value in decomposition["tx_per_spacing_map_summary"].values()
    }
    counterfactual: dict[str, Any] = {}
    for model, suite in report["counterfactual"]["models"].items():
        counterfactual[model] = {
            "artifact_vs_resynthesized_control_relative_mse": suite["artifact_vs_resynthesized_control_relative_mse"],
            "path_alignment_cost_mean": suite["path_alignment_cost_mean"],
            "variants": {
                name: {
                    "frequency_relative_l2": value["metrics"]["frequency_vector"]["relative_l2"],
                    "tx_relative_l2": value["metrics"]["tx_matrix"]["relative_l2"],
                    "gain_control_minus_variant": value["gain_control_minus_variant"],
                    **({"projection_audit": value["projection_audit"]} if "projection_audit" in value else {}),
                    **({"relative_angle_change_vs_control": value["relative_angle_change_vs_control"]} if "relative_angle_change_vs_control" in value else {}),
                }
                for name, value in suite["variants"].items()
            },
        }
    output = {
        "samples": report["samples"],
        "maps": report["maps"],
        "pairing_audit": report["pairing_audit"],
        "pooled_vs_per_map_frequency": decomposition["pooled_vs_per_map_frequency"],
        "frequency_by_lag": frequency,
        "tx_by_spacing_wavelengths": spacing,
        "counterfactual_selection_audit": report["counterfactual_selection_audit"],
        "counterfactual_real_vs_real_floor": {
            "frequency_relative_l2": report["counterfactual"]["real_vs_real_floor"]["frequency_vector"]["relative_l2"],
            "tx_relative_l2": report["counterfactual"]["real_vs_real_floor"]["tx_matrix"]["relative_l2"],
        },
        "counterfactual": counterfactual,
        "interpretation": report["interpretation"],
        "cautions": report["cautions"],
    }
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
