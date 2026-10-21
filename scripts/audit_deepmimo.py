#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.config import load_project_config
from physics_csi.data.rayverse import discover_propbin_files, ids_from_path, read_propbin
from physics_csi.utils.io import atomic_json_dump


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit RayVerse/D2Los PROP v2 metadata.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument(
        "--max-files",
        type=int,
        default=100,
        help="Number of source files to inspect; use 0 for the complete dataset.",
    )
    parser.add_argument("--output", default="artifacts/data_audit.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config_dir)["data"]
    files = discover_propbin_files(config["raw_root"])
    selected = files if args.max_files == 0 else files[: args.max_files]
    path_counts: Counter[int] = Counter()
    versions: Counter[int] = Counter()
    carriers: Counter[float] = Counter()
    receiver_count = 0
    usable_receiver_count = 0
    path_record_count = 0
    errors: list[dict[str, str]] = []
    for path in selected:
        try:
            propbin = read_propbin(path, fallback_carrier_hz=config["system"]["carrier_hz"])
        except (OSError, ValueError, EOFError) as error:
            errors.append({"file": str(path), "error": str(error)})
            continue
        versions[propbin.version] += 1
        carriers[propbin.carrier_hz] += 1
        receiver_count += len(propbin.receivers)
        path_record_count += len(propbin.paths)
        counts = propbin.receivers["path_count"].astype(np.int64)
        usable_receiver_count += int((counts > 0).sum())
        path_counts.update(int(value) for value in counts[counts > 0])
    map_file_counts = Counter(ids_from_path(path)[0] for path in files)
    report = {
        "raw_root": str(Path(config["raw_root"]).resolve()),
        "total_files": len(files),
        "audited_files": len(selected),
        "nonempty_maps": len(map_file_counts),
        "map_file_counts": {str(key): value for key, value in sorted(map_file_counts.items())},
        "versions": {str(key): value for key, value in sorted(versions.items())},
        "carriers_hz": {str(key): value for key, value in sorted(carriers.items())},
        "receiver_records": receiver_count,
        "usable_receiver_records": usable_receiver_count,
        "path_records": path_record_count,
        "path_count_histogram": {
            str(key): value for key, value in sorted(path_counts.items())
        },
        "max_observed_paths": max(path_counts, default=0),
        "configured_max_paths": int(config["max_paths"]),
        "receivers_exceeding_max_paths": sum(
            count for paths, count in path_counts.items() if paths > int(config["max_paths"])
        ),
        "max_path_class_is_censored": any(
            paths > int(config["max_paths"]) for paths in path_counts
        ),
        "missing_native_fields": [
            "complex_csi",
            "independent_path_phase",
            "ofdm_bandwidth",
            "array_geometry",
            "transmit_power",
        ],
        "errors": errors,
    }
    atomic_json_dump(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

