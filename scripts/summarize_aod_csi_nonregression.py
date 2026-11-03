#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SELECTED_METRICS = (
    "pdp_mmd_rbf",
    "pdp_sliced_wasserstein",
    "raw_full_csi_projected_mmd_rbf",
    "raw_full_csi_projected_sliced_wasserstein",
    "phase_invariant_full_csi_projected_mmd_rbf",
    "phase_invariant_full_csi_projected_sliced_wasserstein",
    "tx_spatial_correlation_relative_frobenius",
    "tx_spatial_correlation_magnitude_mae",
    "tx_spatial_correlation_phase_mae_rad",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AoD CSI non-regression results.")
    parser.add_argument("report")
    args = parser.parse_args()
    report = json.load(Path(args.report).open("r", encoding="utf-8"))

    def compact(section: dict[str, Any]) -> dict[str, Any]:
        return {
            metric: {
                "relative_mean_improvement": section[metric].get(
                    "relative_mean_improvement"
                ),
                "bootstrap_95_percent_ci": section[metric][
                    "bootstrap_95_percent_ci"
                ],
                "win_rate_excluding_ties": section[metric].get(
                    "win_rate_excluding_ties"
                ),
                "two_sided_sign_test_p": section[metric].get(
                    "two_sided_sign_test_p"
                ),
                "evidence": section[metric]["evidence"],
                "noninferiority_status": section[metric]["noninferiority_status"],
                "passes_noninferiority_gate": section[metric][
                    "passes_noninferiority_gate"
                ],
            }
            for metric in SELECTED_METRICS
        }

    output = {
        "candidate_label": report["candidate_label"],
        "samples": report["samples"],
        "maps_evaluated": report["maps_evaluated"],
        "relative_noninferiority_margin": report[
            "relative_noninferiority_margin"
        ],
        "normalization_audit": report["normalization_audit"],
        "nonregression_gates": report["nonregression_gates"],
        "comparisons": {
            name: compact(values) for name, values in report["summaries"].items()
        },
        "interpretation": report["interpretation"],
    }
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
