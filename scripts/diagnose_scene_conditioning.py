#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.evaluation.scene_diagnostics import (
    paired_map_summary,
    scene_structure_distances,
)
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


METRIC_GROUPS = {
    "aod_direction_distribution": ["aod_direction_sliced_wasserstein"],
    "multipath_angular_clustering": ["angular_clustering_sliced_wasserstein"],
    "path_power_angle_coupling": ["power_angle_correlation_mae"],
    "full_spatial_correlation_matrix": [
        "spatial_correlation_relative_frobenius",
        "spatial_correlation_magnitude_mae",
        "spatial_correlation_phase_mae_rad",
    ],
    "full_conditioned_csi_distribution": [
        "full_csi_sliced_wasserstein",
        "full_csi_mmd_rbf",
    ],
}


def _subset_to_device(
    payload: dict[str, Any], keep: torch.Tensor, device: torch.device
) -> dict[str, torch.Tensor]:
    count = int(keep.numel())
    result: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor) and value.shape[:1] == (count,):
            result[key] = value[keep].to(device)
    return result


def _validate_payload(payload: dict[str, Any]) -> None:
    required_top_level = ("reference", "correct_map", "shuffled_map")
    missing = [key for key in required_top_level if key not in payload]
    if missing:
        raise ValueError(f"generated payload is missing: {missing}")
    required_tensor_keys = (
        "csi",
        "path_mask",
        "path_powers_linear",
        "aoa_az_deg",
        "aoa_el_deg",
        "aod_az_deg",
        "aod_el_deg",
    )
    for group in required_top_level:
        missing = [key for key in required_tensor_keys if key not in payload[group]]
        if missing:
            raise ValueError(f"{group} is missing tensors: {missing}")
    if "map_id" not in payload["reference"]:
        raise ValueError("reference payload has no map_id")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose whether correct Map IDs improve scene-specific CSI structure."
    )
    parser.add_argument("--generated", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--minimum-per-map", type=int, default=8)
    parser.add_argument("--projections", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.minimum_per_map < 2:
        raise ValueError("--minimum-per-map must be at least 2")
    if args.projections < 1 or args.bootstrap_samples < 1:
        raise ValueError("projections and bootstrap samples must be positive")

    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    payload = torch.load(args.generated, map_location="cpu", weights_only=False)
    _validate_payload(payload)
    reference = payload["reference"]
    map_ids = reference["map_id"].long()
    known_maps = sorted(int(value) for value in map_ids.unique())
    comparisons = ["shuffled_map"]
    if isinstance(payload.get("physics_only_baseline"), dict):
        comparisons.append("physics_only_baseline")

    print(
        json.dumps(
            {
                "event": "diagnostic_started",
                "device": str(device),
                "samples": int(map_ids.numel()),
                "maps": len(known_maps),
                "comparisons": comparisons,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    per_map: dict[str, Any] = {}
    accumulated: dict[str, dict[str, dict[str, list[float]]]] = {
        comparison: {} for comparison in comparisons
    }
    evaluated_maps = 0
    for map_index, map_id in enumerate(known_maps, start=1):
        keep = map_ids == map_id
        sample_count = int(keep.sum())
        if sample_count < args.minimum_per_map:
            per_map[f"map_{map_id}"] = {
                "samples": sample_count,
                "status": "insufficient_samples",
            }
            continue
        reference_map = _subset_to_device(reference, keep, device)
        correct_map = _subset_to_device(payload["correct_map"], keep, device)
        correct_metrics = scene_structure_distances(
            correct_map,
            reference_map,
            projections=args.projections,
        )
        item: dict[str, Any] = {
            "samples": sample_count,
            "correct_map": correct_metrics,
        }
        for comparison in comparisons:
            comparison_map = _subset_to_device(payload[comparison], keep, device)
            comparison_metrics = scene_structure_distances(
                comparison_map,
                reference_map,
                projections=args.projections,
            )
            item[comparison] = comparison_metrics
            for metric, correct_value in correct_metrics.items():
                comparison_value = comparison_metrics.get(metric)
                if correct_value is None or comparison_value is None:
                    continue
                values = accumulated[comparison].setdefault(
                    metric, {"correct": [], "comparison": []}
                )
                values["correct"].append(float(correct_value))
                values["comparison"].append(float(comparison_value))
        per_map[f"map_{map_id}"] = item
        evaluated_maps += 1
        print(
            json.dumps(
                {
                    "event": "map_complete",
                    "map": map_id,
                    "map_index": map_index,
                    "maps_total": len(known_maps),
                    "samples": sample_count,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summaries: dict[str, Any] = {}
    for comparison, metrics in accumulated.items():
        summaries[comparison] = {
            metric: paired_map_summary(
                values["correct"],
                values["comparison"],
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
            for metric, values in metrics.items()
        }

    hypotheses: dict[str, Any] = {}
    for group, metric_names in METRIC_GROUPS.items():
        hypotheses[group] = {}
        for comparison in comparisons:
            available = {
                metric: summaries[comparison][metric]
                for metric in metric_names
                if metric in summaries[comparison]
            }
            hypotheses[group][comparison] = available

    report = {
        "format_version": 1,
        "purpose": (
            "Paired per-Map diagnosis of whether correct Map conditioning improves "
            "held-out scene-specific propagation structure."
        ),
        "source": str(Path(args.generated).resolve()),
        "scene_plan_fingerprint": payload.get("scene_plan_fingerprint"),
        "device": str(device),
        "samples": int(map_ids.numel()),
        "maps_total": len(known_maps),
        "maps_evaluated": evaluated_maps,
        "minimum_per_map": args.minimum_per_map,
        "projections": args.projections,
        "bootstrap_samples": args.bootstrap_samples,
        "metric_direction": "lower_is_better",
        "hypotheses": hypotheses,
        "summaries": summaries,
        "per_map": per_map,
        "cautions": [
            "This is an exploratory multi-metric analysis; account for multiple comparisons in paper claims.",
            "A positive bootstrap interval supports correct-Map fidelity; output change alone is insufficient.",
            "Repeat generation with multiple sampling seeds before making a definitive scene-conditioning claim.",
        ],
    }
    atomic_json_dump(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
