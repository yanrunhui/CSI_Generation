#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.config import load_project_config
from physics_csi.data.dataset import make_dataloaders
from physics_csi.models.conditional_vae import PhysicsConditionedVAE
from physics_csi.physics.synthesizer import SystemConfig
from physics_csi.training.losses import CVAELoss
from physics_csi.training.trainer import Trainer
from physics_csi.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the physics-conditioned multipath VAE.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    config = load_project_config(args.config_dir)
    seed_everything(int(config["train"]["seed"]))
    loaders = make_dataloaders(
        config["data"]["prepared_root"],
        batch_size=int(config["train"]["batch_size"]),
        num_workers=int(config["train"]["num_workers"]),
    )
    system = SystemConfig.from_mapping(config["data"]["system"])
    model = PhysicsConditionedVAE(config["model"], system)
    loss = CVAELoss(
        config["train"]["loss_weights"],
        delay_scale_ns=float(config["model"]["path_scales"]["delay_ns"]),
    )
    trainer = Trainer(model, loss, config["train"], config)
    if args.resume:
        trainer.resume(args.resume)
    trainer.fit(loaders)


if __name__ == "__main__":
    main()

