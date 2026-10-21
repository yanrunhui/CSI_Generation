#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.data.conditional_experiments import create_condition_experiment_plan
from physics_csi.utils.io import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create matched ID, sparse-condition or continuous-region OOD splits."
    )
    parser.add_argument("--prepared-root", default="artifacts/prepared_d2los")
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", choices=["id", "sparse", "ood"], required=True)
    parser.add_argument(
        "--field",
        choices=["total_power_db", "delay_spread_ns", "k_factor_db"],
        default="delay_spread_ns",
    )
    parser.add_argument("--lower", type=float, required=True)
    parser.add_argument("--upper", type=float, required=True)
    parser.add_argument("--los-status", type=int, choices=[0, 1], default=None)
    parser.add_argument("--path-counts", type=int, nargs="*", default=None)
    parser.add_argument("--sparse-retain-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.upper <= args.lower:
        parser.error("--upper must be greater than --lower")
    if args.mode == "sparse" and args.sparse_retain_count is None:
        parser.error("sparse mode requires --sparse-retain-count")
    region = {
        "field": args.field,
        "lower": args.lower,
        "upper": args.upper,
        "los_status": args.los_status,
        "path_counts": args.path_counts,
    }
    plan = create_condition_experiment_plan(
        args.prepared_root,
        name=args.name,
        mode=args.mode,
        region=region,
        seed=args.seed,
        sparse_retain_count=args.sparse_retain_count,
    )
    atomic_json_dump(plan, args.output)
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
