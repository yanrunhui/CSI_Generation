#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.data.conditional_experiments import make_experiment_datasets
from physics_csi.data.dataset import PreparedShardDataset
from physics_csi.data.scene_id import make_scene_id_datasets
from physics_csi.models.diffusion.multipath_diffusion import MultipathDiffusionModel
from physics_csi.physics.synthesizer import SystemConfig
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


def _ece(probabilities: torch.Tensor, target: torch.Tensor, bins: int = 15) -> float:
    confidence, prediction = probabilities.max(-1)
    correct = prediction == target
    result = torch.tensor(0.0)
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        keep = (confidence >= lower) & (confidence < upper if index + 1 < bins else confidence <= upper)
        if keep.any():
            result += keep.float().mean() * (
                correct[keep].float().mean() - confidence[keep].mean()
            ).abs()
    return float(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate Model B path-count probabilities on the original-distribution validation view."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prepared-root", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-checkpoint", default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    root = args.prepared_root or config["data"]["prepared_root"]
    plan = config.get("condition_experiment")
    scene_plan = config.get("scene_id_experiment")
    if scene_plan is not None:
        dataset = make_scene_id_datasets(root, scene_plan)["validation"]
    elif plan is not None:
        dataset = make_experiment_datasets(root, plan)["validation"]
    else:
        dataset = PreparedShardDataset(root, "validation")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    device = resolve_device(args.device)
    model = MultipathDiffusionModel(
        config["multipath_diffusion"],
        SystemConfig.from_mapping(config["data"]["system"]),
        checkpoint["statistics"],
    ).to(device)
    model.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))
    model.eval()
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            conditions = batch["conditions"].to(device)
            validity = batch["condition_validity"].to(device)
            embedding = model.condition_embedding(
                conditions,
                validity,
                map_id=(batch.get("map_id").to(device) if "map_id" in batch else None),
                map_index=(
                    batch.get("map_index").to(device) if "map_index" in batch else None
                ),
            )
            value = model.path_count_model(embedding)
            bias = model._path_count_logit_bias()
            if bias is not None:
                value = value + bias.to(value)
            logits.append(value.cpu())
            targets.append(batch["path_count"].long().cpu() - 1)
    values = torch.cat(logits)
    target = torch.cat(targets)
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(values / temperature, target)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0))
    before = torch.softmax(values, dim=-1)
    after = torch.softmax(values / temperature, dim=-1)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "validation_samples": int(target.numel()),
        "temperature": temperature,
        "nll_before": float(F.cross_entropy(values, target)),
        "nll_after": float(F.cross_entropy(values / temperature, target)),
        "ece_before": _ece(before, target),
        "ece_after": _ece(after, target),
        "accuracy": float((after.argmax(-1) == target).float().mean()),
    }
    atomic_json_dump(report, args.output)
    if args.output_checkpoint:
        calibrated = copy.deepcopy(checkpoint)
        calibrated["config"]["sampling"]["path_count_temperature"] = temperature
        output = Path(args.output_checkpoint)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(calibrated, output)
        report["output_checkpoint"] = str(output.resolve())
        atomic_json_dump(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
