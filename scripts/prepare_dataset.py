#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.config import load_project_config
from physics_csi.data.prepare import prepare_dataset
from physics_csi.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare sharded D2Los multipath/CSI data.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_project_config(args.config_dir)
    seed_everything(int(config["data"]["seed"]))
    manifest = prepare_dataset(config, overwrite=args.overwrite)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

