#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.data.dataset import PreparedShardDataset
from physics_csi.data.scene_id import load_seen_map_spatial_plan, make_scene_id_datasets
from physics_csi.evaluation.research_diagnostics import CONDITION_NAMES
from physics_csi.utils.io import atomic_json_dump


def _collect(dataset: Dataset, batch_size: int, num_workers: int) -> dict[str, torch.Tensor]:
    values: dict[str, list[torch.Tensor]] = defaultdict(list)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    for batch in loader:
        for key in ("map_id", "conditions", "condition_validity"):
            values[key].append(batch[key].cpu())
    if not values:
        raise ValueError("dataset split is empty")
    return {key: torch.cat(chunks) for key, chunks in values.items()}


def _quantile_edges(values: torch.Tensor, bins: int) -> torch.Tensor:
    if values.numel() == 0:
        return torch.tensor([-0.5, 0.5], dtype=torch.float32)
    edges = torch.unique(torch.quantile(values.float(), torch.linspace(0, 1, bins + 1)))
    if edges.numel() < 2:
        center = float(values[0])
        edges = torch.tensor([center - 0.5, center + 0.5])
    return edges


def _bin_index(values: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    return torch.bucketize(values.contiguous(), edges[1:-1], right=False)


def _condition_scale(train: dict[str, torch.Tensor]) -> torch.Tensor:
    scale = train["conditions"][:, :3].std(0, unbiased=False).clamp_min(1e-6)
    valid_k = train["condition_validity"][:, 2].bool()
    if bool(valid_k.any()):
        scale[2] = train["conditions"][valid_k, 2].std(unbiased=False).clamp_min(1e-6)
    return scale


def _standardized_distance(
    candidates: dict[str, torch.Tensor], query: torch.Tensor, query_valid: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    comparable = candidates["condition_validity"].bool() & query_valid.unsqueeze(0)
    comparable[:, 3] = False
    delta = torch.where(
        comparable,
        (candidates["conditions"] - query.unsqueeze(0)) / torch.cat([scale, scale.new_ones(1)]),
        torch.zeros_like(candidates["conditions"]),
    )
    return delta[:, :3].square().sum(-1).sqrt()


def _parse_query(value: str) -> dict[str, Any]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "query must be MAP_ID,TOTAL_POWER_DB,DELAY_SPREAD_NS,K_FACTOR_DB_OR_NAN,LOS"
        )
    map_id = int(parts[0].removeprefix("map_"))
    los = int(parts[4])
    if los not in {0, 1}:
        raise argparse.ArgumentTypeError("query LoS must be 0 or 1")
    k_factor = float(parts[3])
    return {
        "map_id": map_id,
        "conditions": torch.tensor(
            [float(parts[1]), float(parts[2]), 0.0 if math.isnan(k_factor) else k_factor, float(los)]
        ),
        "validity": torch.tensor([True, True, not math.isnan(k_factor), True]),
        "raw": value,
    }


def _neighborhood_summary(
    payload: dict[str, torch.Tensor],
    map_id: int,
    scale: torch.Tensor,
    radii: list[float],
    maximum_queries: int,
    seed: int,
) -> dict[str, Any]:
    selected = torch.nonzero(payload["map_id"].long() == map_id, as_tuple=False).flatten()
    if selected.numel() == 0:
        return {"status": "no_samples"}
    generator = torch.Generator().manual_seed(seed + map_id)
    if selected.numel() > maximum_queries:
        order = torch.randperm(selected.numel(), generator=generator)[:maximum_queries]
        query_indices = selected[order]
    else:
        query_indices = selected
    conditions = payload["conditions"].float()
    validity = payload["condition_validity"].bool()
    neighbor_counts = {radius: [] for radius in radii}
    nearest: list[float] = []
    for query_index in query_indices.tolist():
        query = conditions[query_index]
        query_valid = validity[query_index]
        same_map = payload["map_id"].long() == map_id
        same_los = (conditions[:, 3] >= 0.5) == (query[3] >= 0.5)
        same_k_validity = validity[:, 2] == query_valid[2]
        candidates_index = torch.nonzero(same_map & same_los & same_k_validity, as_tuple=False).flatten()
        candidates_index = candidates_index[candidates_index != query_index]
        if candidates_index.numel() == 0:
            nearest.append(float("inf"))
            for radius in radii:
                neighbor_counts[radius].append(0)
            continue
        candidate = {
            "conditions": conditions[candidates_index],
            "condition_validity": validity[candidates_index],
        }
        distance = _standardized_distance(candidate, query, query_valid, scale)
        nearest.append(float(distance.min()))
        for radius in radii:
            neighbor_counts[radius].append(int((distance <= radius).sum()))
    finite_nearest = torch.tensor([value for value in nearest if math.isfinite(value)])
    result: dict[str, Any] = {
        "samples_in_map": int(selected.numel()),
        "sampled_queries": int(query_indices.numel()),
        "nearest_neighbor_standardized_distance": (
            {
                "median": float(finite_nearest.median()),
                "p90": float(torch.quantile(finite_nearest, 0.9)),
                "maximum": float(finite_nearest.max()),
            }
            if finite_nearest.numel()
            else None
        ),
        "neighbors": {},
    }
    for radius, counts in neighbor_counts.items():
        tensor = torch.tensor(counts, dtype=torch.float32)
        result["neighbors"][str(radius)] = {
            "median": float(tensor.median()),
            "p10": float(torch.quantile(tensor, 0.1)),
            "fraction_below_8": float((tensor < 8).float().mean()),
            "fraction_below_32": float((tensor < 32).float().mean()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Map x physical-condition coverage and local effective sample counts."
    )
    parser.add_argument("--prepared-root", default="artifacts/prepared_d2los")
    parser.add_argument("--scene-plan", default=None)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--radii", default="0.5,1.0,2.0")
    parser.add_argument("--minimum-cell-samples", type=int, default=8)
    parser.add_argument("--maximum-neighborhood-queries-per-map", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        type=_parse_query,
        metavar="MAP,POWER,DELAY,K_OR_NAN,LOS",
        help="Repeat for explicit local-support questions, for example map_10,-90,100,5,1.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.bins < 2 or args.minimum_cell_samples < 1:
        parser.error("--bins must be >=2 and --minimum-cell-samples must be positive")
    radii = sorted({float(value) for value in args.radii.split(",")})
    if not radii or radii[0] <= 0:
        parser.error("--radii must contain positive comma-separated values")

    if args.scene_plan:
        plan = load_seen_map_spatial_plan(args.scene_plan)
        datasets = make_scene_id_datasets(args.prepared_root, plan)
        plan_fingerprint = plan["fingerprint"]
    else:
        datasets = {
            split: PreparedShardDataset(args.prepared_root, split)
            for split in ("train", "validation", "test")
        }
        plan_fingerprint = None
    payloads = {
        split: _collect(dataset, args.batch_size, args.num_workers)
        for split, dataset in datasets.items()
    }
    train = payloads["train"]
    scale = _condition_scale(train)
    edges = {
        0: _quantile_edges(train["conditions"][:, 0], args.bins),
        1: _quantile_edges(train["conditions"][:, 1], args.bins),
        2: _quantile_edges(
            train["conditions"][train["condition_validity"][:, 2].bool(), 2], args.bins
        ),
    }
    maps = sorted(
        {int(value) for payload in payloads.values() for value in payload["map_id"].unique()}
    )
    per_map: dict[str, Any] = {}
    for map_id in maps:
        item: dict[str, Any] = {"splits": {}}
        for split, payload in payloads.items():
            keep = payload["map_id"].long() == map_id
            condition = payload["conditions"][keep]
            validity = payload["condition_validity"][keep]
            cells: Counter[str] = Counter()
            if condition.numel():
                power_bin = _bin_index(condition[:, 0], edges[0])
                delay_bin = _bin_index(condition[:, 1], edges[1])
                k_bin = _bin_index(condition[:, 2], edges[2])
                for row in range(condition.shape[0]):
                    los = int(condition[row, 3] >= 0.5)
                    k_label = str(int(k_bin[row])) if bool(validity[row, 2]) else "NA"
                    cells[f"los={los}|power={int(power_bin[row])}|delay={int(delay_bin[row])}|k={k_label}"] += 1
            counts = torch.tensor(list(cells.values()), dtype=torch.float32)
            item["splits"][split] = {
                "samples": int(keep.sum()),
                "los_samples": int((condition[:, 3] >= 0.5).sum()) if condition.numel() else 0,
                "nlos_samples": int((condition[:, 3] < 0.5).sum()) if condition.numel() else 0,
                "occupied_joint_cells": len(cells),
                "cells_with_at_least_minimum": int((counts >= args.minimum_cell_samples).sum()),
                "fraction_of_samples_in_sparse_cells": (
                    float(
                        sum(count for count in cells.values() if count < args.minimum_cell_samples)
                        / max(int(keep.sum()), 1)
                    )
                ),
                "joint_cell_counts": dict(sorted(cells.items())),
            }
        item["training_local_neighborhoods"] = _neighborhood_summary(
            train,
            map_id,
            scale,
            radii,
            args.maximum_neighborhood_queries_per_map,
            args.seed,
        )
        per_map[f"map_{map_id}"] = item

    queries: list[dict[str, Any]] = []
    for query in args.query:
        map_id = query["map_id"]
        same_map = train["map_id"].long() == map_id
        same_los = (train["conditions"][:, 3] >= 0.5) == (query["conditions"][3] >= 0.5)
        same_k_validity = train["condition_validity"][:, 2] == query["validity"][2]
        keep = same_map & same_los & same_k_validity
        candidate = {
            "conditions": train["conditions"][keep],
            "condition_validity": train["condition_validity"][keep],
        }
        distance = (
            _standardized_distance(candidate, query["conditions"], query["validity"], scale)
            if bool(keep.any())
            else torch.empty(0)
        )
        queries.append(
            {
                "query": query["raw"],
                "map_id": map_id,
                "map_training_samples": int(same_map.sum()),
                "exact_los_and_k_validity_candidates": int(keep.sum()),
                "nearest_standardized_distance": float(distance.min()) if distance.numel() else None,
                "neighbors_by_radius": {
                    str(radius): int((distance <= radius).sum()) for radius in radii
                },
            }
        )

    report = {
        "format_version": 1,
        "diagnostic": "map_physical_condition_coverage",
        "prepared_root": str(Path(args.prepared_root).resolve()),
        "scene_plan_fingerprint": plan_fingerprint,
        "condition_order": list(CONDITION_NAMES),
        "standardization_scale_from_training": dict(zip(CONDITION_NAMES[:3], scale.tolist())),
        "joint_binning": {
            "method": "pooled training quantiles; LoS and K-validity are exact strata",
            "bins_requested": args.bins,
            "minimum_cell_samples": args.minimum_cell_samples,
            "edges": {
                CONDITION_NAMES[index]: values.tolist() for index, values in edges.items()
            },
        },
        "local_neighborhood_definition": (
            "Euclidean distance over valid standardized power/delay/K; exact LoS and exact "
            "K-validity; leave-one-out for empirical training summaries."
        ),
        "explicit_queries": queries,
        "per_map": per_map,
    }
    atomic_json_dump(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
