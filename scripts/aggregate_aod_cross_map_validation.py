#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.utils.io import atomic_json_dump


METRICS = (
    "path_power_aod_swd",
    "link_power_aod_swd",
    "tx_spatial_correlation_relative_frobenius",
    "phase_invariant_full_csi_projected_swd",
    "beamforming_gain_swd",
    "csi_diversity_ratio_to_real",
)


def _report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, raw_path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("report label cannot be empty")
    return label, Path(raw_path)


def _finite(values: Iterable[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(value)]


def _audit(values: Iterable[float | None]) -> dict[str, Any]:
    tensor = torch.tensor(_finite(values), dtype=torch.float32)
    if not tensor.numel():
        return {"count": 0}
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean()),
        "median": float(torch.quantile(tensor, 0.50)),
        "p10": float(torch.quantile(tensor, 0.10)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "minimum": float(tensor.min()),
        "maximum": float(tensor.max()),
    }


def _map_cluster_summary(
    rows: list[Mapping[str, Any]],
    value_function: Callable[[Mapping[str, Any]], float | None],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    by_map: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = value_function(row)
        if value is not None and math.isfinite(value):
            by_map[int(row["map_id"])].append(float(value))
    map_ids = sorted(by_map)
    if not map_ids:
        return {"maps": 0, "observations": 0}
    map_means = torch.tensor(
        [sum(by_map[map_id]) / len(by_map[map_id]) for map_id in map_ids],
        dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        0,
        map_means.numel(),
        (bootstrap_samples, map_means.numel()),
        generator=generator,
    )
    estimates = map_means[indices].mean(-1)
    return {
        "maps": len(map_ids),
        "map_ids": map_ids,
        "observations": sum(len(values) for values in by_map.values()),
        "map_cluster_mean": float(map_means.mean()),
        "map_cluster_median": float(torch.quantile(map_means, 0.50)),
        "bootstrap_95_percentile_ci": [
            float(torch.quantile(estimates, 0.025)),
            float(torch.quantile(estimates, 0.975)),
        ],
        "per_map_mean_audit": _audit(map_means.tolist()),
    }


def _relative_delta(row: Mapping[str, Any], metric: str) -> float | None:
    if row.get("status") != "fidelity_evaluated":
        return None
    first = row["realism_first_vs_real"].get(metric)
    second = row["exact_control_vs_real"].get(metric)
    if first is None or second is None or not math.isfinite(first) or abs(first) <= 1e-20:
        return None
    if metric == "csi_diversity_ratio_to_real":
        return abs(second - 1.0) - abs(first - 1.0)
    return (second - first) / abs(first)


def _absolute_metric(row: Mapping[str, Any], metric: str, policy: str) -> float | None:
    if row.get("status") != "fidelity_evaluated":
        return None
    value = row[f"{policy}_vs_real"].get(metric)
    return float(value) if value is not None and math.isfinite(value) else None


def _real_real_metric(row: Mapping[str, Any], metric: str) -> float | None:
    values = row.get("real_vs_real")
    if not values:
        return None
    value = values.get(metric)
    return float(value) if value is not None and math.isfinite(value) else None


def _excess_over_real_real(
    row: Mapping[str, Any], metric: str, policy: str
) -> float | None:
    current = _absolute_metric(row, metric, policy)
    floor = _real_real_metric(row, metric)
    if current is None or floor is None or abs(floor) <= 1e-20:
        return None
    return (current - floor) / abs(floor)


def _scope_summary(
    rows: list[dict[str, Any]],
    *,
    multipliers: list[int],
    bootstrap_samples: int,
    seed: int,
    maximum_allowed_relative_degradation: float,
) -> dict[str, Any]:
    maps_seen = sorted({int(row["map_id"]) for row in rows})
    fidelity = [row for row in rows if row.get("status") == "fidelity_evaluated"]
    fidelity_maps = sorted({int(row["map_id"]) for row in fidelity})
    result: dict[str, Any] = {
        "rows": len(rows),
        "maps_seen": len(maps_seen),
        "map_ids_seen": maps_seen,
        "fidelity_rows": len(fidelity),
        "maps_with_fidelity": len(fidelity_maps),
        "fidelity_map_coverage": len(fidelity_maps) / max(len(maps_seen), 1),
        "natural_acceptance_rate": _map_cluster_summary(
            rows,
            lambda row: row.get("natural_acceptance_rate"),
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "candidate_pool": {},
        "metrics": {},
    }
    for number, multiplier in enumerate(multipliers):
        key = str(multiplier)

        def fill_probability(row: Mapping[str, Any], key: str = key) -> float | None:
            values = row.get("candidate_pool_statistics", {}).get(key, {})
            if values.get("status") != "ok":
                return None
            return values.get("probability_of_filling_request")

        valid_rows = [row for row in rows if fill_probability(row) is not None]
        fill_by_map: dict[int, list[float]] = defaultdict(list)
        for row in valid_rows:
            fill_by_map[int(row["map_id"])].append(float(fill_probability(row)))
        qualifying_maps = sum(
            sum(values) / len(values) >= 0.95 for values in fill_by_map.values()
        )
        result["candidate_pool"][key] = {
            "available_maps": len({int(row["map_id"]) for row in valid_rows}),
            "fill_probability": _map_cluster_summary(
                rows,
                fill_probability,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 100 + number,
            ),
            "map_fraction_with_fill_probability_at_least_0_95": (
                qualifying_maps / max(len(fill_by_map), 1)
            ),
        }
    for number, metric in enumerate(METRICS):
        delta = _map_cluster_summary(
            fidelity,
            lambda row, metric=metric: _relative_delta(row, metric),
            bootstrap_samples=bootstrap_samples,
            seed=seed + 1000 + number,
        )
        exact_wins = [
            _relative_delta(row, metric) < 0
            for row in fidelity
            if _relative_delta(row, metric) is not None
        ]
        ci = delta.get("bootstrap_95_percentile_ci")
        nondegradation = bool(
            ci and ci[1] <= maximum_allowed_relative_degradation
        )
        result["metrics"][metric] = {
            "real_vs_real": _map_cluster_summary(
                fidelity,
                lambda row, metric=metric: _real_real_metric(row, metric),
                bootstrap_samples=bootstrap_samples,
                seed=seed + 1500 + number,
            ),
            "realism_first": _map_cluster_summary(
                fidelity,
                lambda row, metric=metric: _absolute_metric(
                    row, metric, "realism_first"
                ),
                bootstrap_samples=bootstrap_samples,
                seed=seed + 2000 + number,
            ),
            "exact_control": _map_cluster_summary(
                fidelity,
                lambda row, metric=metric: _absolute_metric(
                    row, metric, "exact_control"
                ),
                bootstrap_samples=bootstrap_samples,
                seed=seed + 3000 + number,
            ),
            "realism_first_excess_over_real_real": _map_cluster_summary(
                fidelity,
                lambda row, metric=metric: _excess_over_real_real(
                    row, metric, "realism_first"
                ),
                bootstrap_samples=bootstrap_samples,
                seed=seed + 4000 + number,
            ),
            "exact_control_excess_over_real_real": _map_cluster_summary(
                fidelity,
                lambda row, metric=metric: _excess_over_real_real(
                    row, metric, "exact_control"
                ),
                bootstrap_samples=bootstrap_samples,
                seed=seed + 5000 + number,
            ),
            "exact_minus_realism_relative": delta,
            "exact_control_win_rate": sum(exact_wins) / max(len(exact_wins), 1),
            "nondegradation_gate": nondegradation,
        }
    return result


def _seed_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["map_id"],
            row["L"],
            row["los_status"],
            row["k_factor_valid"],
            row["mode"],
            row["target_az_deg"],
            row["tolerance_deg"],
        )
        grouped[key].append(row)
    repeated = [
        values
        for values in grouped.values()
        if len(
            {
                row["b3_cache_generation_seed"]
                for row in values
                if row.get("b3_cache_generation_seed") is not None
            }
        )
        >= 2
    ]
    result: dict[str, Any] = {
        "groups_with_multiple_generation_seeds": len(repeated),
        "acceptance_rate_std": _audit(
            torch.tensor([row["natural_acceptance_rate"] for row in values]).std(unbiased=False).item()
            for values in repeated
        ),
        "metric_delta_sign_consistency": {},
    }
    for metric in METRICS:
        consistencies: list[float] = []
        for values in repeated:
            deltas = _finite(_relative_delta(row, metric) for row in values)
            if len(deltas) >= 2:
                nonzero = [value for value in deltas if abs(value) > 1e-12]
                if nonzero:
                    positive = sum(value > 0 for value in nonzero)
                    consistencies.append(max(positive, len(nonzero) - positive) / len(nonzero))
        result["metric_delta_sign_consistency"][metric] = _audit(consistencies)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map-clustered aggregation of cross-Map, cross-angle AoD validation."
    )
    parser.add_argument("--report", action="append", type=_report, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--maximum-relative-degradation", type=float, default=0.05)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("bootstrap-samples must be at least 100")

    reports: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    multipliers: set[int] = set()
    for label, path in args.report:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("experiment") != "aod_tolerance_candidate_multiplier_tradeoff":
            raise ValueError(f"{path} is not an AoD tolerance tradeoff report")
        reports.append(
            {
                "label": label,
                "path": str(path.resolve()),
                "artifact": report["artifact"],
                "artifact_evaluation_seed": report.get("artifact_evaluation_seed"),
                "b3_cache_generation_seed": report.get("b3_cache_generation_seed"),
            }
        )
        multipliers.update(int(value) for value in report["candidate_multipliers"])
        for source in report["rows"]:
            row = dict(source)
            row["report_label"] = label
            row["b3_cache_generation_seed"] = report.get(
                "b3_cache_generation_seed"
            )
            rows.append(row)
    multiplier_list = sorted(multipliers)
    scopes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        base = f"mode={row['mode']}|los={row['los_status']}|tol={row['tolerance_deg']:g}"
        scopes[f"overall|{base}"].append(row)
        scopes[f"angle={row['target_az_deg']:g}|{base}"].append(row)
    summaries = {
        name: _scope_summary(
            values,
            multipliers=multiplier_list,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + number * 10000,
            maximum_allowed_relative_degradation=args.maximum_relative_degradation,
        )
        for number, (name, values) in enumerate(sorted(scopes.items()))
    }
    output = {
        "format_version": 1,
        "experiment": "aod_cross_map_cross_angle_validation",
        "reports": reports,
        "report_count": len(reports),
        "unique_b3_cache_generation_seeds": sorted(
            {
                item["b3_cache_generation_seed"]
                for item in reports
                if item["b3_cache_generation_seed"] is not None
            }
        ),
        "rows": len(rows),
        "candidate_multipliers": multiplier_list,
        "maximum_allowed_relative_degradation": args.maximum_relative_degradation,
        "scope_summaries": summaries,
        "generation_seed_stability": _seed_stability(rows),
        "decision_rule": {
            "candidate_supply": "Prefer configurations whose per-Map fill probability is >=0.95.",
            "conditional_fidelity": "Use Map-cluster bootstrap; exact-control passes non-degradation only when the upper 95% CI is <= the configured relative degradation tolerance.",
            "support": "Unsupported Map-angle groups remain outside the denominator of fidelity metrics but are reported through fidelity_map_coverage.",
            "seed_stability": "A single artifact cannot establish generation-seed stability.",
        },
    }
    atomic_json_dump(output, args.output)
    print(
        json.dumps(
            {
                "event": "aod_cross_map_validation_complete",
                "output": str(Path(args.output).resolve()),
                "reports": len(reports),
                "rows": len(rows),
                "scopes": len(summaries),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
