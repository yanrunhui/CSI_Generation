#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _overall(values: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dimension in values["b3"]:
        b3 = values["b3"][dimension]
        scene = values["b3_scene_id"][dimension]
        result[dimension] = {
            "mmd": {
                "b3": b3["mmd_mean"],
                "b3_scene_id": scene["mmd_mean"],
                "b3_minus_scene": b3["mmd_mean"] - scene["mmd_mean"],
                "winner": "B3-S-ID" if scene["mmd_mean"] < b3["mmd_mean"] else "B3",
            },
            "swd": {
                "b3": b3["swd_mean"],
                "b3_scene_id": scene["swd_mean"],
                "b3_minus_scene": b3["swd_mean"] - scene["swd_mean"],
                "winner": "B3-S-ID" if scene["swd_mean"] < b3["swd_mean"] else "B3",
            },
        }
    return result


def _mode(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "overall": _overall(values["overall"]),
        "per_map_paired_summary": values["per_map_paired_summary"],
        "map_condition_paired_summary": values["map_condition_paired_summary"],
        "maps_evaluated": len(values["per_map"]),
        "map_condition_groups_evaluated": len(values["map_condition_groups"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    args = parser.parse_args()
    with Path(args.report).open(encoding="utf-8") as stream:
        report = json.load(stream)
    invariant_status = report["phase_invariance_validation"]["status"]
    invariant_summary = _mode(report["C_global_phase_invariant"])
    invariant_summary["result_validity"] = (
        "valid"
        if invariant_status == "ok"
        else "invalid: canonicalization failed the random-rotation invariance check"
    )
    output = {
        "samples": report["samples"],
        "maps": report["maps"],
        "map_condition_groups": report["map_condition_groups"],
        "phase_invariance_validation": report["phase_invariance_validation"],
        "A_raw_full_csi": _mode(report["A_raw_full_csi"]),
        "B_independent_random_global_phase": _mode(
            report["B_independent_random_global_phase"]
        ),
        "C_global_phase_invariant": invariant_summary,
        "sign_convention": (
            "Overall and paired-summary positive B3-minus-B3-S-ID values favor B3-S-ID."
        ),
        "interpretation_cases": {
            "A": "If phase-invariant evaluation makes B3-S-ID clearly better, global phase explained much of the raw gap.",
            "B": "If B3 remains better after phase removal, global phase is not the main explanation.",
            "C": "If both distances change strongly but ranking stays fixed, the metric is phase-sensitive but phase does not explain the relative gap.",
        },
    }
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
