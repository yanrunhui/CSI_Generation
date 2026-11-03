#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


KEY_METRICS = {
    "power_aod_link": "path_distribution.power_aod_link_path_count_conditioned_sliced_wasserstein",
    "delay_power_link": "path_distribution.delay_power_link_path_count_conditioned_sliced_wasserstein",
    "delay_power_aod_link": "path_distribution.delay_power_aod_link_path_count_conditioned_sliced_wasserstein",
    "full_multipath_swd": "path_distribution.full_multipath_path_count_conditioned_sliced_wasserstein",
    "full_csi_mmd": "csi_distribution.full_csi_projected_mmd_rbf",
    "full_csi_swd": "csi_distribution.full_csi_projected_sliced_wasserstein",
    "csi_summary_mmd": "csi_distribution.csi_summary_mmd_rbf",
    "csi_summary_swd": "csi_distribution.csi_summary_sliced_wasserstein",
    "pdp_swd": "csi_distribution.pdp_sliced_wasserstein",
}


def _projection_summary(grid: Mapping[str, Any]) -> dict[str, Any]:
    return {
        dimension: {
            "mmd_mean": values["mmd_mean"],
            "mmd_range": [values["mmd_min"], values["mmd_max"]],
            "swd_mean": values["swd_mean"],
            "swd_range": [values["swd_min"], values["swd_max"]],
        }
        for dimension, values in grid["dimensions"].items()
    }


def _coherence_summary(values: Mapping[str, Any]) -> dict[str, Any]:
    joint = values["joint_path_associations"]
    return {
        "phase_delay_swd": joint["phase_delay_sliced_wasserstein"],
        "phase_aod_swd": joint["phase_aod_sliced_wasserstein"],
        "phase_delay_aod_swd": joint["phase_delay_aod_sliced_wasserstein"],
        "aoa_aod_swd": joint["aoa_aod_sliced_wasserstein"],
        "frequency_complex_swd": values["frequency_complex_correlation"][
            "sliced_wasserstein"
        ],
        "frequency_complex_mmd": values["frequency_complex_correlation"][
            "mmd_rbf"
        ],
        "tx_complex_swd": values["tx_complex_correlation"]["sliced_wasserstein"],
        "tx_complex_mmd": values["tx_complex_correlation"]["mmd_rbf"],
        "standard_csi": values["standard_csi_metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("symmetric_report")
    parser.add_argument("coherence_report")
    args = parser.parse_args()
    with Path(args.symmetric_report).open(encoding="utf-8") as stream:
        symmetric = json.load(stream)
    with Path(args.coherence_report).open(encoding="utf-8") as stream:
        coherence = json.load(stream)

    differential = {}
    for repair, repair_values in symmetric["differential_recovery"].items():
        differential[repair] = {
            "scene_attribution_guard": repair_values["scene_attribution_guard"],
            "b3_attribution_guard": repair_values["b3_attribution_guard"],
            "metrics": {
                label: repair_values["metrics"][path]
                for label, path in KEY_METRICS.items()
            },
        }

    models = {
        name: {
            "coherence": _coherence_summary(values["coherence"]),
            "projection_grid": _projection_summary(
                values["full_csi_projection_grid"]
            ),
        }
        for name, values in coherence["models_vs_real"].items()
    }
    tests = {
        name: {
            "relative_csi_error": values[
                "relative_csi_error_vs_resynthesized_real"
            ],
            "coherence": _coherence_summary(values["coherence"]),
            "projection_grid": _projection_summary(
                values["full_csi_projection_grid"]
            ),
        }
        for name, values in coherence["controlled_real_parameter_tests"].items()
    }
    output = {
        "experiment_1_symmetric_counterfactual": {
            "common_samples": symmetric["common_counterfactual_samples"],
            "common_sample_rate": symmetric["common_sample_rate"],
            "selection": symmetric["selection"],
            "differential_recovery": differential,
            "sign_convention": (
                "Positive differential_gain_scene_minus_b3 means B3-S-ID recovered more."
            ),
        },
        "experiment_2_complex_coherence": {
            "samples": coherence["samples"],
            "reference_synthesis_validation": coherence[
                "reference_synthesis_validation"
            ],
            "models_vs_real": models,
            "controlled_tests": tests,
            "projection_grid": coherence["projection_grid"],
        },
    }
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
