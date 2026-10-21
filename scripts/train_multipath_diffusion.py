#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.config import deep_merge, load_yaml
from physics_csi.data.dataset import make_dataloaders
from physics_csi.data.conditional_experiments import (
    load_condition_experiment_plan,
    make_experiment_dataloaders,
    make_experiment_datasets,
)
from physics_csi.data.diffusion_stats import load_or_compute_diffusion_statistics
from physics_csi.models.diffusion.multipath_diffusion import MultipathDiffusionModel
from physics_csi.physics.synthesizer import SystemConfig
from physics_csi.training.diffusion_trainer import DiffusionTrainer
from physics_csi.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train conditional multipath diffusion.")
    parser.add_argument("--config", default="configs/multipath_diffusion.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--experiment-plan",
        default=None,
        help="JSON plan produced by create_condition_experiment.py.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Override checkpoint directory; strongly recommended for every ablation.",
    )
    args = parser.parse_args()
    resume_checkpoint = (
        torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    )
    config = copy.deepcopy(
        resume_checkpoint["config"]
        if resume_checkpoint is not None
        else deep_merge(load_yaml(ROOT / "configs/data.yaml"), load_yaml(args.config))
    )
    if resume_checkpoint is None and args.experiment_plan:
        plan = load_condition_experiment_plan(args.experiment_plan)
        config["condition_experiment"] = plan
        config["statistics_path"] = str(
            Path("artifacts/statistics") / f"diffusion_{plan['name']}.json"
        )
        config["train"]["checkpoint_dir"] = str(
            Path("artifacts/checkpoints") / f"multipath_diffusion_{plan['name']}"
        )
    if resume_checkpoint is not None and args.experiment_plan:
        raise ValueError("resume uses the experiment plan embedded in the checkpoint")
    if args.checkpoint_dir:
        config["train"]["checkpoint_dir"] = args.checkpoint_dir
    train = config["train"]
    seed_everything(int(train["seed"]))
    system = SystemConfig.from_mapping(config["data"]["system"])
    plan = config.get("condition_experiment")
    experiment_datasets = (
        make_experiment_datasets(config["data"]["prepared_root"], plan)
        if plan is not None
        else None
    )
    statistics = (
        resume_checkpoint["statistics"]
        if resume_checkpoint is not None
        else load_or_compute_diffusion_statistics(
            config["statistics_path"],
            config["data"]["prepared_root"],
            system,
            batch_size=int(config.get("statistics_batch_size", 512)),
            num_workers=int(train["num_workers"]),
            dataset=experiment_datasets["train"] if experiment_datasets else None,
            source_identity=(
                f"condition_experiment:{plan['fingerprint']}" if plan is not None else "train"
            ),
        )
    )
    loaders = (
        make_experiment_dataloaders(
            experiment_datasets,
            batch_size=int(train["batch_size"]),
            num_workers=int(train["num_workers"]),
        )
        if experiment_datasets is not None
        else make_dataloaders(
            config["data"]["prepared_root"],
            batch_size=int(train["batch_size"]),
            num_workers=int(train["num_workers"]),
        )
    )
    model = MultipathDiffusionModel(config["multipath_diffusion"], system, statistics)
    trainer = DiffusionTrainer(model, train, config, statistics, "multipath_diffusion")
    if args.resume:
        trainer.resume(args.resume)
    trainer.fit(loaders)


if __name__ == "__main__":
    main()

