#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.utils.io import atomic_json_dump


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate matched ID/sparse/OOD reports into a sample-support curve."
    )
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for path_string in args.reports:
        path = Path(path_string)
        report = json.loads(path.read_text(encoding="utf-8"))
        experiment = report.get("experiment") or {}
        row = {
            "report": str(path.resolve()),
            "experiment": experiment.get("name"),
            "mode": experiment.get("mode", report.get("baseline", "unknown")),
            "retained_region_samples": experiment.get("sparse_retain_count"),
            "reference_samples": report.get("reference_samples", report.get("samples")),
            "csi_mmd": _nested(report, "distribution", "csi_summary_mmd_rbf"),
            "csi_sliced_wasserstein": _nested(
                report, "distribution", "csi_summary_sliced_wasserstein"
            ),
            "full_joint_mmd": _nested(
                report, "multipath_distribution", "full_joint_mmd_rbf"
            ),
            "full_joint_sliced_wasserstein": _nested(
                report, "multipath_distribution", "full_joint_sliced_wasserstein"
            ),
            "active_path_correlation_mae": _nested(
                report, "multipath_distribution", "active_path_correlation_mae"
            ),
            "mode_coverage": _nested(
                report, "multipath_diversity", "reference_mode_coverage"
            ),
            "path_count_balanced_accuracy": _nested(
                report, "path_count", "balanced_accuracy"
            ),
            "constraint_delay_change_ns": _nested(
                report, "constraint_changes", "delay_change_ns_mean"
            ),
        }
        rows.append(row)
    mode_order = {"ood": 0, "sparse": 1, "id": 2}
    rows.sort(
        key=lambda row: (
            mode_order.get(str(row["mode"]), 3),
            int(row["retained_region_samples"] or 0),
        )
    )
    payload = {
        "interpretation": (
            "Compare rows only when region, seed, protocol and reference sample budget match. "
            "Lower distribution distances are better; higher mode coverage and balanced accuracy are better."
        ),
        "rows": rows,
    }
    atomic_json_dump(payload, args.output)
    if args.csv:
        output = Path(args.csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
