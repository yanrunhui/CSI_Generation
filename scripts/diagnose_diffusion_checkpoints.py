#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.data.dataset import PreparedShardDataset
from physics_csi.models.diffusion.angle_delay_diffusion import AngleDelayDiffusionModel
from physics_csi.models.diffusion.multipath_diffusion import MultipathDiffusionModel
from physics_csi.physics.synthesizer import SystemConfig
from physics_csi.training.trainer import move_to_device, resolve_device
from physics_csi.utils.io import atomic_json_dump


def _build_model(checkpoint: dict[str, Any], device: torch.device) -> torch.nn.Module:
    config = checkpoint["config"]
    system = SystemConfig.from_mapping(config["data"]["system"])
    kind = checkpoint.get("model_kind")
    if kind == "multipath_diffusion":
        model = MultipathDiffusionModel(
            config["multipath_diffusion"], system, checkpoint["statistics"]
        )
    elif kind == "angle_delay_diffusion":
        model = AngleDelayDiffusionModel(
            config["angle_delay_diffusion"], system, checkpoint["statistics"]
        )
    else:
        raise ValueError(f"unsupported checkpoint model_kind: {kind!r}")
    return model.to(device)


def _state_health(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    nonfinite: list[str] = []
    floating_values = 0
    for name, value in state.items():
        if not (value.is_floating_point() or value.is_complex()):
            continue
        floating_values += value.numel()
        finite = torch.isfinite(value.real).all()
        if value.is_complex():
            finite = finite & torch.isfinite(value.imag).all()
        if not bool(finite):
            nonfinite.append(name)
    return {
        "all_finite": not nonfinite,
        "nonfinite_tensors": nonfinite,
        "floating_values_checked": floating_values,
    }


def _history_summary(checkpoint: dict[str, Any]) -> dict[str, Any]:
    history = checkpoint.get("history", [])
    if not history:
        return {"records": 0, "last": None, "minimum_validation": None}
    validation = [
        (float(record["validation"]["loss"]), int(record["epoch"]))
        for record in history
        if "validation" in record and "loss" in record["validation"]
    ]
    minimum = min(validation) if validation else None
    return {
        "records": len(history),
        "last": history[-1],
        "minimum_validation": (
            {"loss": minimum[0], "epoch": minimum[1]} if minimum is not None else None
        ),
    }


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    samples = 0
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_to_device(batch, device)
        batch_size = int(batch["conditions"].shape[0])
        metrics = model(batch)
        for name, value in metrics.items():
            scalar = float(value.detach())
            if not math.isfinite(scalar):
                raise FloatingPointError(
                    f"non-finite validation metric {name} at batch {batch_index}"
                )
            totals[name] += scalar * batch_size
        samples += batch_size
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    if samples == 0:
        raise ValueError("no validation samples were evaluated")
    return {
        "samples": samples,
        "seconds": seconds,
        "metrics": {name: value / samples for name, value in totals.items()},
    }


def _relative_state_distance(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
) -> float:
    difference_square = 0.0
    reference_square = 0.0
    for name, first_value in first.items():
        if name not in second or not (first_value.is_floating_point() or first_value.is_complex()):
            continue
        second_value = second[name]
        difference_square += float((first_value.cpu() - second_value.cpu()).abs().square().sum())
        reference_square += float(first_value.cpu().abs().square().sum())
    return math.sqrt(difference_square / max(reference_square, 1e-30))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare diffusion checkpoint integrity and held-out loss.")
    parser.add_argument("checkpoints", nargs="+", help="best.pt and/or last.pt")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--prepared-root", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--weights", choices=["ema", "online"], default="ema")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="artifacts/diagnostics/diffusion_checkpoints.json")
    args = parser.parse_args()
    device = resolve_device(args.device)
    reports: list[dict[str, Any]] = []
    evaluated_states: list[dict[str, torch.Tensor]] = []

    for checkpoint_path in args.checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        weight_key = "ema_state_dict" if args.weights == "ema" else "model_state_dict"
        state = checkpoint[weight_key]
        health = _state_health(state)
        if not health["all_finite"]:
            raise FloatingPointError(
                f"checkpoint contains non-finite tensors: {health['nonfinite_tensors']}"
            )
        model = _build_model(checkpoint, device)
        model.load_state_dict(state)
        prepared_root = args.prepared_root or checkpoint["config"]["data"]["prepared_root"]
        loader = DataLoader(
            PreparedShardDataset(prepared_root, args.split),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        evaluation = _evaluate(model, loader, device, args.max_batches, args.seed)
        reports.append(
            {
                "path": str(Path(checkpoint_path).resolve()),
                "model_kind": checkpoint.get("model_kind"),
                "epoch": int(checkpoint["epoch"]),
                "global_step": int(checkpoint.get("global_step", 0)),
                "recorded_best_validation": float(checkpoint["best_validation"]),
                "weights": args.weights,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "state_health": health,
                "history": _history_summary(checkpoint),
                "held_out": evaluation,
            }
        )
        evaluated_states.append(state)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload: dict[str, Any] = {
        "split": args.split,
        "device": str(device),
        "seed": args.seed,
        "checkpoints": reports,
    }
    if len(evaluated_states) == 2:
        payload["relative_weight_l2_distance"] = _relative_state_distance(
            evaluated_states[0], evaluated_states[1]
        )
        first_loss = reports[0]["held_out"]["metrics"]["loss"]
        second_loss = reports[1]["held_out"]["metrics"]["loss"]
        payload["lower_held_out_loss"] = reports[0]["path"] if first_loss <= second_loss else reports[1]["path"]
    atomic_json_dump(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
