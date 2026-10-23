#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.data.scene_id import (
    load_seen_map_spatial_plan,
    make_scene_id_datasets,
)
from physics_csi.evaluation.diffusion_comparison import compare_csi_distributions
from physics_csi.evaluation.distribution import distribution_metrics
from physics_csi.evaluation.scene_diagnostics import scene_structure_distances
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


PAYLOAD_KEYS = (
    "csi",
    "path_mask",
    "path_delays_ns",
    "path_powers_linear",
    "path_phases_rad",
    "aoa_az_deg",
    "aoa_el_deg",
    "aod_az_deg",
    "aod_el_deg",
    "conditions",
    "condition_validity",
)


def _stack(records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([record[key] for record in records]) for key in PAYLOAD_KEYS}


def _to_device(
    payload: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in payload.items()}


def _condition_matrix(payload: dict[str, torch.Tensor], std: torch.Tensor) -> torch.Tensor:
    values = payload["conditions"].float()
    validity = payload["condition_validity"].bool()
    # LoS is matched exactly below; K is zeroed for NLoS instead of inventing a value.
    normalized = values[:, :3] / std[:3].clamp_min(1e-6)
    normalized[:, 2] = torch.where(validity[:, 2], normalized[:, 2], 0.0)
    return normalized


def _matched_indices(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
    std: torch.Tensor,
    maximum_distance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    first_values = _condition_matrix(first, std)
    second_values = _condition_matrix(second, std)
    distances = torch.cdist(first_values, second_values)
    first_los = first["conditions"][:, 3] >= 0.5
    second_los = second["conditions"][:, 3] >= 0.5
    distances = distances.masked_fill(first_los[:, None] != second_los[None, :], torch.inf)
    nearest_distance, nearest_index = distances.min(dim=1)
    keep = torch.nonzero(nearest_distance <= maximum_distance, as_tuple=False).flatten()
    return keep, nearest_index[keep], nearest_distance[keep]


def _aggregate_pair_section(
    pair_reports: list[dict[str, Any]], section: str
) -> dict[str, dict[str, float]]:
    if not pair_reports:
        return {}
    metric_names = sorted(
        {
            key
            for report in pair_reports
            for key, value in report.get(section, {}).items()
            if isinstance(value, (int, float))
        }
    )
    summary: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        selected = [
            report
            for report in pair_reports
            if isinstance(report.get(section, {}).get(metric), (int, float))
        ]
        values = torch.tensor(
            [report[section][metric] for report in selected], dtype=torch.float64
        )
        weights = torch.tensor(
            [report["matched_samples"] for report in selected], dtype=torch.float64
        )
        summary[metric] = {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "matched_sample_weighted_mean": float(
                (values * weights).sum() / weights.sum().clamp_min(1.0)
            ),
        }
    return summary


def _between_within_contrast(
    between: dict[str, dict[str, float]],
    within: dict[str, dict[str, float]],
) -> dict[str, dict[str, float | None]]:
    contrast: dict[str, dict[str, float | None]] = {}
    for metric in sorted(set(between) & set(within)):
        between_mean = between[metric]["mean"]
        within_mean = within[metric]["mean"]
        contrast[metric] = {
            "between_map_mean": between_mean,
            "within_map_null_mean": within_mean,
            "between_minus_within": between_mean - within_mean,
            "between_to_within_ratio": (
                between_mean / within_mean if abs(within_mean) > 1e-20 else None
            ),
        }
    return contrast


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether real maps retain different channel distributions after "
            "nearest-neighbor matching of the four physical conditions."
        )
    )
    parser.add_argument("--scene-plan", required=True)
    parser.add_argument("--prepared-root", default="artifacts/prepared_d2los")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--condition-std-json", default=None)
    parser.add_argument("--maximum-condition-distance", type=float, default=1.0)
    parser.add_argument("--minimum-matches", type=int, default=16)
    parser.add_argument("--maximum-map-pairs", type=int, default=100)
    parser.add_argument("--maximum-samples-per-map", type=int, default=256)
    parser.add_argument(
        "--null-comparison-samples",
        type=int,
        default=64,
        help=(
            "Use exactly this many condition-matched samples for every cross-Map and "
            "within-Map scene-structure distance, removing sample-size bias."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device for map matching and distribution metrics: auto, cpu or cuda.",
    )
    parser.add_argument(
        "--log-every-map-pairs",
        type=int,
        default=100,
        help="Print map-matching progress after this many candidate map pairs.",
    )
    parser.add_argument(
        "--diagnostic-projections",
        type=int,
        default=64,
        help="Random projections for scene-structure sliced Wasserstein metrics.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.diagnostic_projections < 1:
        raise ValueError("--diagnostic-projections must be positive")
    if args.null_comparison_samples < args.minimum_matches:
        raise ValueError("--null-comparison-samples must be at least --minimum-matches")
    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    torch.set_grad_enabled(False)

    plan = load_seen_map_spatial_plan(args.scene_plan)
    dataset = make_scene_id_datasets(args.prepared_root, plan)[args.split]
    records_by_map: dict[int, list[dict[str, Any]]] = {}
    for index in range(len(dataset)):
        record = dataset[index]
        records_by_map.setdefault(int(record["map_id"]), []).append(record)
        if (index + 1) % 1000 == 0 or index + 1 == len(dataset):
            print(
                json.dumps(
                    {
                        "event": "dataset_loading",
                        "samples_complete": index + 1,
                        "samples_total": len(dataset),
                        "maps_seen": len(records_by_map),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    generator = torch.Generator().manual_seed(args.seed)
    for map_id, records in records_by_map.items():
        if len(records) > args.maximum_samples_per_map:
            selection = torch.randperm(len(records), generator=generator)[
                : args.maximum_samples_per_map
            ]
            records_by_map[map_id] = [records[int(index)] for index in selection]
    # Keep the CSI payloads on CPU.  Only compact condition tensors remain on the
    # accelerator during matching; selected fixed-size Map pairs are transferred
    # just before their metrics are computed.  This keeps peak CUDA memory bounded.
    payloads = {map_id: _stack(records) for map_id, records in records_by_map.items()}
    del records_by_map
    if args.condition_std_json:
        with Path(args.condition_std_json).open("r", encoding="utf-8") as stream:
            std = torch.tensor(json.load(stream)["condition_std"], dtype=torch.float32)
    else:
        all_conditions = torch.cat(
            [payload["conditions"].float() for payload in payloads.values()]
        )
        std = all_conditions.std(0, unbiased=False).clamp_min(1e-6)
    matching_payloads = {
        map_id: {
            "conditions": payload["conditions"].to(device),
            "condition_validity": payload["condition_validity"].to(device),
        }
        for map_id, payload in payloads.items()
    }
    matching_std = std.to(device)

    map_pairs = list(itertools.combinations(sorted(payloads), 2))
    print(
        json.dumps(
            {
                "event": "audit_started",
                "device": str(device),
                "maps": len(payloads),
                "map_pairs": len(map_pairs),
                "maximum_samples_per_map": args.maximum_samples_per_map,
                "null_comparison_samples": args.null_comparison_samples,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    candidate_pairs: list[tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for pair_index, (first_map, second_map) in enumerate(map_pairs, start=1):
        first_index, second_index, distances = _matched_indices(
            matching_payloads[first_map],
            matching_payloads[second_map],
            matching_std,
            args.maximum_condition_distance,
        )
        first_index = first_index.cpu()
        second_index = second_index.cpu()
        distances = distances.cpu()
        if first_index.numel() >= args.minimum_matches:
            candidate_pairs.append(
                (
                    int(first_index.numel()),
                    first_map,
                    second_map,
                    first_index,
                    second_index,
                    distances,
                )
            )
        if (
            pair_index % max(args.log_every_map_pairs, 1) == 0
            or pair_index == len(map_pairs)
        ):
            print(
                json.dumps(
                    {
                        "event": "condition_matching",
                        "pairs_complete": pair_index,
                        "pairs_total": len(map_pairs),
                        "candidate_pairs": len(candidate_pairs),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    candidate_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    diagnostic_candidate_pairs = [
        item for item in candidate_pairs if item[0] >= args.null_comparison_samples
    ]
    selected_pairs = diagnostic_candidate_pairs[: args.maximum_map_pairs]
    pair_reports: list[dict[str, Any]] = []
    for report_index, (
        matched_count,
        first_map,
        second_map,
        first_index,
        second_index,
        distances,
    ) in enumerate(selected_pairs, start=1):
        diagnostic_selection = torch.randperm(
            matched_count, generator=generator
        )[: args.null_comparison_samples]
        selected_first_index = first_index[diagnostic_selection]
        selected_second_index = second_index[diagnostic_selection]
        selected_distances = distances[diagnostic_selection]
        print(
            json.dumps(
                {
                    "event": "distribution_metrics",
                    "pair": report_index,
                    "pairs_total": len(selected_pairs),
                    "first_map": first_map,
                    "second_map": second_map,
                    "available_matches": matched_count,
                    "diagnostic_samples": args.null_comparison_samples,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        first = _to_device(
            {
                key: value[selected_first_index]
                for key, value in payloads[first_map].items()
            },
            device,
        )
        second = _to_device(
            {
                key: value[selected_second_index]
                for key, value in payloads[second_map].items()
            },
            device,
        )
        pair_reports.append(
            {
                "first_map": first_map,
                "second_map": second_map,
                "available_matches": matched_count,
                "matched_samples": args.null_comparison_samples,
                "condition_distance_mean": float(selected_distances.mean()),
                "condition_distance_maximum": float(selected_distances.max()),
                "csi_distribution": compare_csi_distributions(first["csi"], second["csi"]),
                "multipath_distribution": distribution_metrics(
                    first, second, [0.5, 1.0, 2.0, 4.0]
                ),
                "scene_structure": scene_structure_distances(
                    first,
                    second,
                    projections=args.diagnostic_projections,
                ),
            }
        )
        del first, second
    within_map_reports: list[dict[str, Any]] = []
    for map_index, map_id in enumerate(sorted(payloads), start=1):
        payload = payloads[map_id]
        sample_count = int(payload["conditions"].shape[0])
        if sample_count < 2 * args.null_comparison_samples:
            continue
        permutation = torch.randperm(sample_count, generator=generator)
        midpoint = sample_count // 2
        first_half = {
            key: value[permutation[:midpoint]] for key, value in payload.items()
        }
        second_half = {
            key: value[permutation[midpoint:]] for key, value in payload.items()
        }
        first_index, second_index, distances = _matched_indices(
            {
                "conditions": first_half["conditions"].to(device),
                "condition_validity": first_half["condition_validity"].to(device),
            },
            {
                "conditions": second_half["conditions"].to(device),
                "condition_validity": second_half["condition_validity"].to(device),
            },
            matching_std,
            args.maximum_condition_distance,
        )
        first_index = first_index.cpu()
        second_index = second_index.cpu()
        distances = distances.cpu()
        available_matches = int(first_index.numel())
        if available_matches < args.null_comparison_samples:
            continue
        diagnostic_selection = torch.randperm(
            available_matches, generator=generator
        )[: args.null_comparison_samples]
        selected_first_index = first_index[diagnostic_selection]
        selected_second_index = second_index[diagnostic_selection]
        selected_distances = distances[diagnostic_selection]
        first = _to_device(
            {
                key: value[selected_first_index]
                for key, value in first_half.items()
            },
            device,
        )
        second = _to_device(
            {
                key: value[selected_second_index]
                for key, value in second_half.items()
            },
            device,
        )
        within_map_reports.append(
            {
                "map": map_id,
                "available_matches": available_matches,
                "matched_samples": args.null_comparison_samples,
                "condition_distance_mean": float(selected_distances.mean()),
                "condition_distance_maximum": float(selected_distances.max()),
                "scene_structure": scene_structure_distances(
                    first,
                    second,
                    projections=args.diagnostic_projections,
                ),
            }
        )
        print(
            json.dumps(
                {
                    "event": "within_map_null",
                    "map": map_id,
                    "map_index": map_index,
                    "maps_total": len(payloads),
                    "available_matches": available_matches,
                    "diagnostic_samples": args.null_comparison_samples,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del first, second, first_half, second_half
    between_summary = _aggregate_pair_section(pair_reports, "scene_structure")
    within_summary = _aggregate_pair_section(within_map_reports, "scene_structure")
    report = {
        "format_version": 1,
        "purpose": (
            "Real-map separation after physical-condition matching; nonzero distances "
            "measure scene information not explained by the four control conditions."
        ),
        "scene_plan_fingerprint": plan["fingerprint"],
        "split": args.split,
        "device": str(device),
        "maps": len(payloads),
        "candidate_map_pairs": len(candidate_pairs),
        "diagnostic_candidate_map_pairs": len(diagnostic_candidate_pairs),
        "reported_map_pairs": len(pair_reports),
        "matching": {
            "maximum_standardized_condition_distance": args.maximum_condition_distance,
            "minimum_matches": args.minimum_matches,
            "maximum_samples_per_map": args.maximum_samples_per_map,
            "null_comparison_samples": args.null_comparison_samples,
            "sample_size_note": (
                "Every reported cross-Map and within-Map scene-structure distance "
                "uses the same number of matched samples."
            ),
            "los_matched_exactly": True,
        },
        "scene_structure_summary": between_summary,
        "within_map_scene_structure_null": within_summary,
        "between_vs_within_scene_structure": _between_within_contrast(
            between_summary, within_summary
        ),
        "within_map_nulls": within_map_reports,
        "map_pairs": pair_reports,
    }
    atomic_json_dump(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
