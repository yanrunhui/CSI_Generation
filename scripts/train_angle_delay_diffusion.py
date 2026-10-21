#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.config import deep_merge, load_yaml
from physics_csi.data.dataset import make_dataloaders
from physics_csi.data.diffusion_stats import load_or_compute_diffusion_statistics
from physics_csi.models.diffusion.angle_delay_diffusion import AngleDelayDiffusionModel
from physics_csi.physics.synthesizer import SystemConfig
from physics_csi.training.diffusion_trainer import DiffusionTrainer
from physics_csi.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train conditional angle-delay diffusion.")
    parser.add_argument("--config", default="configs/angle_delay_diffusion.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    resume_checkpoint = (
        torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    )
    config = (
        resume_checkpoint["config"]
        if resume_checkpoint is not None
        else deep_merge(load_yaml(ROOT / "configs/data.yaml"), load_yaml(args.config))
    )
    train = config["train"]
    seed_everything(int(train["seed"]))
    system = SystemConfig.from_mapping(config["data"]["system"])
    statistics = (
        resume_checkpoint["statistics"]
        if resume_checkpoint is not None
        else load_or_compute_diffusion_statistics(
            config["statistics_path"],
            config["data"]["prepared_root"],
            system,
            batch_size=int(config.get("statistics_batch_size", 512)),
            num_workers=int(train["num_workers"]),
        )
    )
    loaders = make_dataloaders(
        config["data"]["prepared_root"],
        batch_size=int(train["batch_size"]),
        num_workers=int(train["num_workers"]),
    )
    model = AngleDelayDiffusionModel(config["angle_delay_diffusion"], system, statistics)
    trainer = DiffusionTrainer(model, train, config, statistics, "angle_delay_diffusion")
    if args.resume:
        trainer.resume(args.resume)
    trainer.fit(loaders)


if __name__ == "__main__":
    main()
