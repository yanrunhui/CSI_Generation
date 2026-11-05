#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


KEY_METRICS = (
    "path_power_aod_swd",
    "link_power_aod_swd",
    "tx_spatial_correlation_relative_frobenius",
    "phase_invariant_full_csi_projected_swd",
    "pdp_swd",
    "beamforming_gain_swd",
    "csi_diversity_ratio_to_real",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize dual-mode AoD fidelity.")
    parser.add_argument("report")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for group in report["groups"]:
        if group.get("status") != "ok":
            continue
        projection = group["projection_vs_angle_matched_real"]
        paired_unprojected = group.get(
            "paired_unprojected_vs_angle_matched_real"
        )
        natural = group.get("natural_filtered_vs_angle_matched_real")
        resampled_only = group.get("resampled_only_vs_angle_matched_real")
        resampling_projected = group.get(
            "resampling_small_projection_vs_angle_matched_real"
        )
        resampling_rotation = group.get(
            "resampling_small_projection_rotation_absolute_deg"
        )
        row: dict[str, Any] = {
            "map_id": group["map_id"],
            "L": group["path_count"],
            "los": group["los_status"],
            "mode": group["mode"],
            "target_az_deg": group["target_az_deg"],
            "real_support": group["real_angle_support"],
            "natural_support": group["natural_angle_support"],
            "projection_samples": group["projection_samples"],
            "rotation_abs_mean_deg": group["projection_rotation_absolute_deg"]["mean"],
            "rotation_abs_p95_deg": group["projection_rotation_absolute_deg"]["p95"],
            "resampling_selected_samples": (
                group.get("resampling_selection_audit", {}).get(
                    "condition_matched_samples"
                )
            ),
            "resampling_rotation_abs_mean_deg": (
                resampling_rotation.get("mean") if resampling_rotation else None
            ),
            "resampling_rotation_abs_p95_deg": (
                resampling_rotation.get("p95") if resampling_rotation else None
            ),
            "resampling_fraction_within_small_threshold": (
                resampling_rotation.get("fraction_within_threshold")
                if resampling_rotation
                else None
            ),
        }
        for metric in KEY_METRICS:
            row[f"projection_{metric}"] = projection.get(metric)
            row[f"paired_unprojected_{metric}"] = (
                paired_unprojected.get(metric) if paired_unprojected else None
            )
            row[f"natural_{metric}"] = natural.get(metric) if natural else None
            row[f"resampled_only_{metric}"] = (
                resampled_only.get(metric) if resampled_only else None
            )
            row[f"resampling_projected_{metric}"] = (
                resampling_projected.get(metric) if resampling_projected else None
            )
        rows.append(row)
    print(
        json.dumps(
            {
                "experiment": report["experiment"],
                "artifact": report["artifact"],
                "groups_completed": report["groups_completed"],
                "rows": rows,
                "interpretation": report["interpretation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
