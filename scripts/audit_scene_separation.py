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
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = load_seen_map_spatial_plan(args.scene_plan)
    dataset = make_scene_id_datasets(args.prepared_root, plan)[args.split]
    records_by_map: dict[int, list[dict[str, Any]]] = {}
    for index in range(len(dataset)):
        record = dataset[index]
        records_by_map.setdefault(int(record["map_id"]), []).append(record)
    generator = torch.Generator().manual_seed(args.seed)
    for map_id, records in records_by_map.items():
        if len(records) > args.maximum_samples_per_map:
            selection = torch.randperm(len(records), generator=generator)[
                : args.maximum_samples_per_map
            ]
            records_by_map[map_id] = [records[int(index)] for index in selection]
    payloads = {map_id: _stack(records) for map_id, records in records_by_map.items()}
    if args.condition_std_json:
        with Path(args.condition_std_json).open("r", encoding="utf-8") as stream:
            std = torch.tensor(json.load(stream)["condition_std"], dtype=torch.float32)
    else:
        all_conditions = torch.cat(
            [payload["conditions"].float() for payload in payloads.values()]
        )
        std = all_conditions.std(0, unbiased=False).clamp_min(1e-6)

    candidate_pairs: list[tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for first_map, second_map in itertools.combinations(sorted(payloads), 2):
        first_index, second_index, distances = _matched_indices(
            payloads[first_map],
            payloads[second_map],
            std,
            args.maximum_condition_distance,
        )
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
    candidate_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected_pairs = candidate_pairs[: args.maximum_map_pairs]
    pair_reports: list[dict[str, Any]] = []
    for matched_count, first_map, second_map, first_index, second_index, distances in selected_pairs:
        first = {
            key: value[first_index] for key, value in payloads[first_map].items()
        }
        second = {
            key: value[second_index] for key, value in payloads[second_map].items()
        }
        pair_reports.append(
            {
                "first_map": first_map,
                "second_map": second_map,
                "matched_samples": matched_count,
                "condition_distance_mean": float(distances.mean()),
                "condition_distance_maximum": float(distances.max()),
                "csi_distribution": compare_csi_distributions(first["csi"], second["csi"]),
                "multipath_distribution": distribution_metrics(
                    first, second, [0.5, 1.0, 2.0, 4.0]
                ),
            }
        )
    report = {
        "format_version": 1,
        "purpose": (
            "Real-map separation after physical-condition matching; nonzero distances "
            "measure scene information not explained by the four control conditions."
        ),
        "scene_plan_fingerprint": plan["fingerprint"],
        "split": args.split,
        "maps": len(payloads),
        "candidate_map_pairs": len(candidate_pairs),
        "reported_map_pairs": len(pair_reports),
        "matching": {
            "maximum_standardized_condition_distance": args.maximum_condition_distance,
            "minimum_matches": args.minimum_matches,
            "maximum_samples_per_map": args.maximum_samples_per_map,
            "los_matched_exactly": True,
        },
        "map_pairs": pair_reports,
    }
    atomic_json_dump(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
