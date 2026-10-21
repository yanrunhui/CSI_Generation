#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.data.scene_id import create_seen_map_spatial_plan
from physics_csi.utils.io import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a seen-map, receiver-spatial holdout for B3 and B3-S-ID."
    )
    parser.add_argument("--prepared-root", default="artifacts/prepared_d2los")
    parser.add_argument("--name", default="scene_id_seen_map_v1")
    parser.add_argument(
        "--output", default="artifacts/splits/scene_id_seen_map_v1.json"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tile-size-m", type=float, default=10.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--guard-band-tiles", type=int, default=0)
    args = parser.parse_args()
    plan = create_seen_map_spatial_plan(
        args.prepared_root,
        name=args.name,
        seed=args.seed,
        tile_size_m=args.tile_size_m,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        guard_band_tiles=args.guard_band_tiles,
    )
    atomic_json_dump(plan, args.output)
    summary = {key: value for key, value in plan.items() if key != "indices"}
    summary["split_samples"] = {
        split: len(indices) for split, indices in plan["indices"].items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
