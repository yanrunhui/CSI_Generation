#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.utils.io import atomic_json_dump
from scripts.aggregate_aod_cross_map_validation import _map_cluster_summary


TX_METRIC = "tx_spatial_correlation_relative_frobenius"


def _report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, raw_path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("report label cannot be empty")
    return label, Path(raw_path)


def _mean(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def _detail(
    source: dict[str, Any], label: str, generation_seed: int | None
) -> dict[str, Any] | None:
    if source.get("status") != "fidelity_evaluated":
        return None
    first = source["realism_first_vs_real"].get(TX_METRIC)
    exact = source["exact_control_vs_real"].get(TX_METRIC)
    if first is None or exact is None or not math.isfinite(first) or not math.isfinite(exact):
        return None
    absolute_delta = float(exact - first)
    relative_delta = absolute_delta / abs(first) if abs(first) > 1e-20 else None
    stratum = int(source["stratum_samples"])
    real_support = int(source["real_support"])
    rotation = source.get("selected_natural_error_deg", {})
    return {
        "report_label": label,
        "b3_cache_generation_seed": generation_seed,
        "map_id": int(source["map_id"]),
        "L": int(source["L"]),
        "los_status": int(source["los_status"]),
        "k_factor_valid": bool(source["k_factor_valid"]),
        "mode": source["mode"],
        "target_az_deg": float(source["target_az_deg"]),
        "tolerance_deg": float(source["tolerance_deg"]),
        "stratum_samples": stratum,
        "rayverse_support": real_support,
        "rayverse_support_rate": float(
            source.get("rayverse_support_rate", real_support / max(stratum, 1))
        ),
        "natural_candidate_support": int(source["natural_accepted_samples"]),
        "natural_candidate_acceptance_rate": float(source["natural_acceptance_rate"]),
        "matched_samples": int(source["matched_samples"]),
        "rotation_abs_mean_deg": rotation.get("mean"),
        "rotation_abs_p95_deg": rotation.get("p95"),
        "rotation_abs_maximum_deg": rotation.get("maximum"),
        "realism_first_tx_spatial_error": float(first),
        "exact_control_tx_spatial_error": float(exact),
        "exact_minus_realism_absolute_error": absolute_delta,
        "exact_minus_realism_relative_error": relative_delta,
        "exact_control_degraded": absolute_delta > 0.0,
    }


def _map_values(rows: list[dict[str, Any]], key: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None and math.isfinite(value):
            grouped[int(row["map_id"])].append(float(value))
    return {map_id: sum(values) / len(values) for map_id, values in grouped.items()}


def _scope_summary(
    rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    map_delta = _map_values(rows, "exact_minus_realism_absolute_error")
    return {
        "evaluated_rows": len(rows),
        "maps": len(map_delta),
        "map_ids": sorted(map_delta),
        "rayverse_support_rate": _map_cluster_summary(
            rows,
            lambda row: row["rayverse_support_rate"],
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "rotation_abs_mean_deg": _map_cluster_summary(
            rows,
            lambda row: row["rotation_abs_mean_deg"],
            bootstrap_samples=bootstrap_samples,
            seed=seed + 1,
        ),
        "tx_spatial_absolute_error_change": _map_cluster_summary(
            rows,
            lambda row: row["exact_minus_realism_absolute_error"],
            bootstrap_samples=bootstrap_samples,
            seed=seed + 2,
        ),
        "map_fraction_degraded": (
            sum(value > 0.0 for value in map_delta.values()) / max(len(map_delta), 1)
        ),
    }


def _map_summaries(rows: list[dict[str, Any]], los_status: int) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["los_status"] == los_status:
            grouped[row["map_id"]].append(row)
    result: list[dict[str, Any]] = []
    for map_id, values in grouped.items():
        per_angle: dict[float, list[float]] = defaultdict(list)
        for row in values:
            per_angle[row["target_az_deg"]].append(
                row["exact_minus_realism_absolute_error"]
            )
        angle_delta = {
            str(angle): sum(items) / len(items) for angle, items in per_angle.items()
        }
        result.append(
            {
                "map_id": map_id,
                "generation_seeds": sorted(
                    {
                        row["b3_cache_generation_seed"]
                        for row in values
                        if row["b3_cache_generation_seed"] is not None
                    }
                ),
                "angles_evaluated": len(per_angle),
                "mean_absolute_error_change": _mean(
                    row["exact_minus_realism_absolute_error"] for row in values
                ),
                "maximum_absolute_error_increase": max(
                    row["exact_minus_realism_absolute_error"] for row in values
                ),
                "fraction_rows_degraded": sum(
                    row["exact_control_degraded"] for row in values
                )
                / len(values),
                "mean_rayverse_support_rate": _mean(
                    row["rayverse_support_rate"] for row in values
                ),
                "per_target_angle_absolute_error_change": angle_delta,
            }
        )
    return sorted(
        result,
        key=lambda item: (
            item["mean_absolute_error_change"]
            if item["mean_absolute_error_change"] is not None
            else -math.inf
        ),
        reverse=True,
    )


def _seed_consistency(rows: list[dict[str, Any]], minimum_seeds: int) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["map_id"],
                row["L"],
                row["los_status"],
                row["mode"],
                row["target_az_deg"],
                row["tolerance_deg"],
            )
        ].append(row)
    comparable = []
    for key, values in grouped.items():
        by_seed: dict[int, list[float]] = defaultdict(list)
        for row in values:
            generation_seed = row["b3_cache_generation_seed"]
            if generation_seed is not None:
                by_seed[int(generation_seed)].append(
                    row["exact_minus_realism_absolute_error"]
                )
        if len(by_seed) < minimum_seeds:
            continue
        seed_means = {
            seed: sum(items) / len(items) for seed, items in by_seed.items()
        }
        comparable.append(
            {
                "map_id": key[0],
                "L": key[1],
                "los_status": key[2],
                "mode": key[3],
                "target_az_deg": key[4],
                "tolerance_deg": key[5],
                "per_seed_absolute_error_change": seed_means,
                "all_seeds_degraded": all(value > 0 for value in seed_means.values()),
                "all_seeds_improved": all(value < 0 for value in seed_means.values()),
            }
        )
    los = [item for item in comparable if item["los_status"] == 1]
    nlos = [item for item in comparable if item["los_status"] == 0]

    def summary(values: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "groups": len(values),
            "all_seeds_degraded_fraction": sum(
                item["all_seeds_degraded"] for item in values
            )
            / max(len(values), 1),
            "all_seeds_improved_fraction": sum(
                item["all_seeds_improved"] for item in values
            )
            / max(len(values), 1),
        }

    return {
        "minimum_distinct_generation_seeds": minimum_seeds,
        "comparable_groups": len(comparable),
        "nlos": summary(nlos),
        "los": summary(los),
        "per_map_angle": comparable,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Locate Map-angle Tx-spatial failures after exact AoD projection."
    )
    parser.add_argument("--report", action="append", type=_report, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--minimum-seeds", type=int, default=3)
    parser.add_argument("--worst-cases", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("bootstrap-samples must be at least 100")
    if args.minimum_seeds < 2:
        parser.error("minimum-seeds must be at least two")

    details: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for label, path in args.report:
        report = json.loads(path.read_text(encoding="utf-8"))
        generation_seed = report.get("b3_cache_generation_seed")
        sources.append(
            {
                "label": label,
                "path": str(path.resolve()),
                "artifact": report.get("artifact"),
                "b3_cache_generation_seed": generation_seed,
            }
        )
        for source in report["rows"]:
            current = _detail(source, label, generation_seed)
            if current is not None:
                details.append(current)

    scopes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        name = (
            f"los={row['los_status']}|mode={row['mode']}|"
            f"target={row['target_az_deg']:g}|tol={row['tolerance_deg']:g}"
        )
        scopes[name].append(row)
    target_summaries = {
        name: _scope_summary(
            values,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + number * 10,
        )
        for number, (name, values) in enumerate(sorted(scopes.items()))
    }
    worst = sorted(
        details,
        key=lambda row: row["exact_minus_realism_absolute_error"],
        reverse=True,
    )[: args.worst_cases]
    output = {
        "format_version": 1,
        "experiment": "aod_exact_control_spatial_failure_localization",
        "sources": sources,
        "unique_b3_cache_generation_seeds": sorted(
            {
                item["b3_cache_generation_seed"]
                for item in sources
                if item["b3_cache_generation_seed"] is not None
            }
        ),
        "evaluated_rows": len(details),
        "target_angle_summaries": target_summaries,
        "los_map_summaries": _map_summaries(details, 1),
        "nlos_map_summaries": _map_summaries(details, 0),
        "worst_map_angle_cases": worst,
        "generation_seed_direction_consistency": _seed_consistency(
            details, args.minimum_seeds
        ),
        "per_map_angle_rows": details,
        "interpretation": [
            "Positive exact_minus_realism_absolute_error means exact projection worsened Tx spatial correlation error.",
            "Statistics first average repeated rows within each Map and then bootstrap Maps, so shared Maps are not treated as independent evidence.",
            "RayVerse support is reported separately; unsupported groups are not silently pooled into fidelity conclusions.",
            "Generation-seed consistency reuses fixed trained weights and therefore tests sampling stability, not training stability.",
        ],
    }
    atomic_json_dump(output, args.output)
    print(
        json.dumps(
            {
                "event": "aod_spatial_failure_localization_complete",
                "output": str(Path(args.output).resolve()),
                "rows": len(details),
                "los_maps": len(output["los_map_summaries"]),
                "generation_seeds": output["unique_b3_cache_generation_seeds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
