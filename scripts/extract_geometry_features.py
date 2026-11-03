#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.data.geometry import build_geometry_feature_artifact
from physics_csi.data.rayverse import discover_propbin_files, ids_from_path
from physics_csi.utils.io import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract map-global building statistics for the B3-S-Geo-lite probe."
    )
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--geometry-root", default=None)
    parser.add_argument("--map-size", type=float, default=256.0)
    parser.add_argument("--orientation-bins", type=int, default=12)
    parser.add_argument("--layout-grid-size", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raw_root = Path(args.raw_root)
    geometry_root = Path(args.geometry_root) if args.geometry_root else raw_root / "buildings_complete"
    files = discover_propbin_files(raw_root)
    map_ids = sorted({ids_from_path(path)[0] for path in files})
    if not map_ids:
        raise ValueError("no non-empty RayVerse maps were discovered")
    artifact = build_geometry_feature_artifact(
        geometry_root,
        map_ids,
        map_size=args.map_size,
        orientation_bins=args.orientation_bins,
        layout_grid_size=args.layout_grid_size,
    )
    atomic_json_dump(artifact, args.output)
    summary = {
        key: value for key, value in artifact.items() if key not in {"maps"}
    }
    summary["maps"] = len(artifact["maps"])
    summary["feature_dimension"] = len(artifact["feature_names"])
    summary["output"] = str(Path(args.output).resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
