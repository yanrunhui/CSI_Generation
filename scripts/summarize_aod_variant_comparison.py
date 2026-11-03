#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize an AoD residual ablation.")
    parser.add_argument("report")
    args = parser.parse_args()
    report = json.load(Path(args.report).open("r", encoding="utf-8"))

    def compact(section: dict[str, Any]) -> dict[str, Any]:
        return {
            metric: {
                "relative_mean_improvement": value.get("relative_mean_improvement"),
                "bootstrap_95_percent_ci": value.get("bootstrap_95_percent_ci"),
                "win_rate_excluding_ties": value.get("win_rate_excluding_ties"),
                "two_sided_sign_test_p": value.get("two_sided_sign_test_p"),
                "evidence": value.get("evidence"),
            }
            for metric, value in section.items()
        }

    def audit(section: dict[str, Any]) -> dict[str, Any]:
        return {
            "model_kind": section["model_kind"],
            "aod_architecture": section["aod_architecture"],
            "architectural_maximum_deg": section["architectural_maximum_deg"],
            "learned_gate": section["learned_gate"],
            "loss_residual_normalization_deg": section[
                "loss_residual_normalization_deg"
            ],
            "azimuth": section["azimuth"],
            "elevation": section["elevation"],
        }

    result = {
        "experiment": report["experiment"],
        "candidate_label": report["candidate_label"],
        "baseline_label": report["baseline_label"],
        "samples": report["samples"],
        "maps_evaluated": report["maps_evaluated"],
        "candidate_residual_audit": audit(report["candidate_residual_audit"]),
        "baseline_residual_audit": audit(report["baseline_residual_audit"]),
        "candidate_vs_baseline": compact(
            report["summaries"]["candidate_vs_baseline"]
        ),
        "candidate_vs_dp_only": compact(
            report["summaries"]["candidate_vs_dp_only"]
        ),
        "baseline_vs_dp_only": compact(
            report["summaries"]["baseline_vs_dp_only"]
        ),
        "candidate_correct_vs_shuffled": compact(
            report["summaries"]["candidate_correct_vs_shuffled"]
        ),
        "baseline_correct_vs_shuffled": compact(
            report["summaries"]["baseline_correct_vs_shuffled"]
        ),
        "decision_rule": report["decision_rule"],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
