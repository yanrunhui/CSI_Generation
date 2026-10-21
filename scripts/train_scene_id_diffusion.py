#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.config import deep_merge, load_yaml
from physics_csi.data.diffusion_stats import load_or_compute_diffusion_statistics
from physics_csi.data.scene_id import (
    load_seen_map_spatial_plan,
    make_scene_id_dataloaders,
    make_scene_id_datasets,
)
from physics_csi.models.diffusion.multipath_diffusion import MultipathDiffusionModel
from physics_csi.physics.synthesizer import SystemConfig
from physics_csi.training.diffusion_trainer import DiffusionTrainer
from physics_csi.utils.seed import seed_everything


def _load_compatible_b3_weights(
    model: MultipathDiffusionModel, checkpoint_path: str
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_kind") != "multipath_diffusion":
        raise ValueError("--initialize-from-b3 must point to a physics-only B3 checkpoint")
    source = checkpoint.get("ema_state_dict", checkpoint["model_state_dict"])
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    result = model.load_state_dict(compatible, strict=False)
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "loaded_tensors": len(compatible),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "note": "optimizer, scheduler and B3 history are intentionally not restored",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train B3-S-ID or its matched physics-only seen-map baseline."
    )
    parser.add_argument(
        "--config", default="configs/multipath_diffusion_scene_id.yaml"
    )
    parser.add_argument(
        "--scene-plan",
        default=None,
        help="Plan produced by create_scene_id_split.py; required for a new run.",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument(
        "--initialize-from-b3",
        default=None,
        help="Load compatible EMA modules from B3 but start a new optimizer and history.",
    )
    args = parser.parse_args()
    if args.resume and args.initialize_from_b3:
        raise ValueError("--resume and --initialize-from-b3 are mutually exclusive")
    resume_checkpoint = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume
        else None
    )
    config = copy.deepcopy(
        resume_checkpoint["config"]
        if resume_checkpoint is not None
        else deep_merge(load_yaml(ROOT / "configs/data.yaml"), load_yaml(args.config))
    )
    if resume_checkpoint is None:
        if not args.scene_plan:
            raise ValueError("a new run requires --scene-plan")
        plan = load_seen_map_spatial_plan(args.scene_plan)
        config["scene_id_experiment"] = plan
        scene_config = config["multipath_diffusion"].setdefault(
            "scene_conditioning", {}
        )
        if bool(scene_config.get("enabled", False)):
            scene_config["map_id_to_index"] = dict(plan["map_id_to_index"])
        config["statistics_path"] = str(
            Path("artifacts/statistics")
            / f"diffusion_seen_map_{plan['fingerprint'][:12]}.json"
        )
    else:
        if args.scene_plan:
            external = load_seen_map_spatial_plan(args.scene_plan)
            embedded = config.get("scene_id_experiment")
            if embedded is None or external["fingerprint"] != embedded["fingerprint"]:
                raise ValueError("external scene plan does not match resumed checkpoint")
        plan = config.get("scene_id_experiment")
        if plan is None:
            raise ValueError("checkpoint has no embedded scene-ID experiment plan")
    if args.checkpoint_dir:
        config["train"]["checkpoint_dir"] = args.checkpoint_dir
    train = config["train"]
    seed_everything(int(train["seed"]))
    prepared_root = config["data"]["prepared_root"]
    datasets = make_scene_id_datasets(prepared_root, plan)
    system = SystemConfig.from_mapping(config["data"]["system"])
    statistics = (
        resume_checkpoint["statistics"]
        if resume_checkpoint is not None
        else load_or_compute_diffusion_statistics(
            config["statistics_path"],
            prepared_root,
            system,
            batch_size=int(config.get("statistics_batch_size", 512)),
            num_workers=int(train["num_workers"]),
            dataset=datasets["train"],
            source_identity=f"seen_map_spatial:{plan['fingerprint']}",
        )
    )
    loaders = make_scene_id_dataloaders(
        datasets,
        batch_size=int(train["batch_size"]),
        num_workers=int(train["num_workers"]),
    )
    model = MultipathDiffusionModel(
        config["multipath_diffusion"], system, statistics
    )
    if args.initialize_from_b3:
        config["initialization"] = _load_compatible_b3_weights(
            model, args.initialize_from_b3
        )
        print(json.dumps(config["initialization"], indent=2), flush=True)
    model_kind = (
        "multipath_diffusion_scene_id"
        if model.scene_conditioning_enabled
        else "multipath_diffusion_seen_map_baseline"
    )
    trainer = DiffusionTrainer(model, train, config, statistics, model_kind)
    if args.resume:
        trainer.resume(args.resume)
    trainer.fit(loaders)


if __name__ == "__main__":
    main()
