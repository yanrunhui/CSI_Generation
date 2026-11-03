#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.config import load_yaml
from physics_csi.data.delay_power_residual import (
    ConditionNeighborhoodBatchSampler,
    DelayPowerResidualCacheDataset,
    cache_collate,
)
from physics_csi.models.scene_delay_power_residual import SceneConditionedDelayPowerResidual
from physics_csi.training.delay_power_residual import (
    DelayPowerDistributionLoss,
    DelayPowerLossWeights,
)
from physics_csi.training.trainer import move_to_device, resolve_device
from physics_csi.utils.io import atomic_json_dump
from physics_csi.utils.seed import seed_everything


class _EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        online = model.state_dict()
        for name, value in self.model.state_dict().items():
            source = online[name].detach()
            if value.is_floating_point() or value.is_complex():
                value.lerp_(source, 1.0 - self.decay)
            else:
                value.copy_(source)


def _make_model(cache: Mapping[str, Any], config: Mapping[str, Any]) -> SceneConditionedDelayPowerResidual:
    statistics = cache["b3_statistics"]
    multipath = cache["b3_multipath_config"]
    values = dict(config["delay_power_residual"])
    return SceneConditionedDelayPowerResidual(
        max_paths=int(multipath["max_paths"]),
        map_id_to_index=cache["map_id_to_index"],
        condition_mean=statistics["condition_mean"],
        condition_std=statistics["condition_std"],
        constraints=multipath["constraints"],
        **values,
    )


def _make_loss(cache: Mapping[str, Any], config: Mapping[str, Any]) -> DelayPowerDistributionLoss:
    values = dict(config["distribution_loss"])
    weights = DelayPowerLossWeights(**values.pop("weights"))
    multipath_statistics = cache["b3_statistics"]["multipath"]
    return DelayPowerDistributionLoss(
        max_paths=int(cache["b3_multipath_config"]["max_paths"]),
        power_db_mean=float(multipath_statistics["log_power_fraction_db_mean"]),
        power_db_std=float(multipath_statistics["log_power_fraction_db_std"]),
        weights=weights,
        seed=int(config["train"]["seed"]),
        **values,
    )


def _different_map_indices(
    correct: torch.Tensor, map_count: int, generator: torch.Generator | None
) -> torch.Tensor:
    if map_count < 2:
        raise ValueError("shuffled-Map training requires at least two Maps")
    # A neighborhood batch contains one target Map.  Use one wrong Map for
    # the whole batch; mixing many wrong Maps would make the margin task
    # artificially easy by creating a distribution that no single scene
    # branch is expected to represent.
    offset = torch.randint(
        1,
        map_count,
        (),
        generator=generator,
        device=correct.device,
    )
    return (correct + offset) % map_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Train scene-conditioned Delay--Power transport.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config", default="configs/delay_power_residual.yaml")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    train = config["train"]
    if args.checkpoint_dir:
        train["checkpoint_dir"] = args.checkpoint_dir
    seed_everything(int(train["seed"]))
    device = resolve_device(str(train.get("device", "auto")))
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    train_dataset = DelayPowerResidualCacheDataset(cache, "train")
    validation_dataset = DelayPowerResidualCacheDataset(cache, "validation")
    condition_std = cache["b3_statistics"]["condition_std"]
    train_sampler = ConditionNeighborhoodBatchSampler(
        train_dataset,
        batch_size=int(train["batch_size"]),
        batches_per_epoch=int(train["train_batches_per_epoch"]),
        minimum_batch_size=int(train["minimum_batch_size"]),
        condition_std=condition_std,
        seed=int(train["seed"]),
        shuffle=True,
    )
    validation_sampler = ConditionNeighborhoodBatchSampler(
        validation_dataset,
        batch_size=int(train["batch_size"]),
        batches_per_epoch=int(train["validation_batches_per_epoch"]),
        minimum_batch_size=int(train["minimum_batch_size"]),
        condition_std=condition_std,
        seed=int(train["seed"]) + 100000,
        shuffle=False,
    )
    expected_training_maps = {int(value) for value in torch.unique(train_dataset.map_ids)}
    covered_training_maps = {int(key[0]) for key in train_sampler.strata}
    missing_training_maps = sorted(expected_training_maps - covered_training_maps)
    if missing_training_maps:
        names = ", ".join(f"map_{value}" for value in missing_training_maps)
        raise ValueError(
            "No eligible distribution batch exists for "
            f"{names}; lower train.minimum_batch_size instead of leaving Map embeddings untrained."
        )
    print(
        json.dumps(
            {
                "event": "dp_residual_neighborhood_audit",
                "train_samples": len(train_dataset),
                "validation_samples": len(validation_dataset),
                "train_maps": len(expected_training_maps),
                "train_strata": len(train_sampler.strata),
                "validation_strata": len(validation_sampler.strata),
                "train_eligible_samples": int(train_sampler.eligible.numel()),
                "validation_eligible_samples": int(validation_sampler.eligible.numel()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    loaders = {
        "train": DataLoader(
            train_dataset, batch_sampler=train_sampler, collate_fn=cache_collate, num_workers=0
        ),
        "validation": DataLoader(
            validation_dataset,
            batch_sampler=validation_sampler,
            collate_fn=cache_collate,
            num_workers=0,
        ),
    }
    model = _make_model(cache, config).to(device)
    loss_function = _make_loss(cache, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train["learning_rate"]),
        weight_decay=float(train.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(train.get("lr_reduce_factor", 0.5)),
        patience=int(train.get("lr_patience", 5)),
    )
    ema = _EMA(model, float(train.get("ema_decay", 0.995)))
    use_amp = bool(train.get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 0
    global_step = 0
    best_validation = float("inf")
    history: list[dict[str, Any]] = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        ema.model.load_state_dict(checkpoint["ema_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_validation = float(checkpoint["best_validation"])
        history = list(checkpoint.get("history", []))

    map_count = len(model.map_id_to_index)

    def run_epoch(epoch: int, *, training: bool) -> dict[str, float]:
        nonlocal global_step
        sampler = train_sampler if training else validation_sampler
        sampler.set_epoch(epoch)
        active = model if training else ema.model
        active.train(training)
        totals: dict[str, float] = defaultdict(float)
        samples = 0
        context = torch.enable_grad if training else torch.no_grad
        wrong_generator = torch.Generator(device=device).manual_seed(
            int(train["seed"]) + epoch * 1009 + (0 if training else 500000)
        )
        with context():
            for step, batch in enumerate(loaders["train" if training else "validation"]):
                batch = move_to_device(batch, device)
                reference = batch["reference"]
                base = batch["base"]
                conditions = reference["conditions"]
                validity = reference["condition_validity"]
                map_id = reference["map_id"]
                correct_index = active.map_indices(map_id).to(device)
                wrong_index = _different_map_indices(correct_index, map_count, wrong_generator)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_amp
                ):
                    correct_output = active(
                        base,
                        conditions,
                        validity,
                        map_index=correct_index,
                    )
                    shuffled_output = active(
                        base,
                        conditions,
                        validity,
                        map_index=wrong_index,
                    )
                    correct_payload = active.apply_to_payload(base, correct_output)
                    shuffled_payload = active.apply_to_payload(base, shuffled_output)
                    metrics = loss_function(
                        correct_payload=correct_payload,
                        shuffled_payload=shuffled_payload,
                        base_payload=base,
                        reference_payload=reference,
                        conditions=conditions,
                        correct_output=correct_output,
                    )
                if training:
                    scaler.scale(metrics["loss"]).backward()
                    scaler.unscale_(optimizer)
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(train.get("gradient_clip_norm", 1.0))
                    )
                    if not torch.isfinite(gradient_norm):
                        raise FloatingPointError("DP residual produced a non-finite gradient")
                    scaler.step(optimizer)
                    scaler.update()
                    ema.update(model)
                    global_step += 1
                    metrics = dict(metrics)
                    metrics["gradient_norm"] = gradient_norm.detach()
                batch_size = int(conditions.shape[0])
                for key, value in metrics.items():
                    totals[key] += float(value.detach()) * batch_size
                samples += batch_size
                if training and step % int(train.get("log_every_steps", 25)) == 0:
                    print(
                        json.dumps(
                            {
                                "event": "dp_residual_step",
                                "epoch": epoch,
                                "step": step,
                                "global_step": global_step,
                                **{key: float(value.detach()) for key, value in metrics.items()},
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if args.dry_run:
                    break
        if samples == 0:
            raise RuntimeError("neighborhood sampler produced no data")
        return {key: value / samples for key, value in totals.items()}

    checkpoint_dir = Path(train["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(path: Path, epoch: int) -> None:
        payload = {
            "format_version": 1,
            "model_kind": "scene_delay_power_residual",
            "epoch": epoch,
            "global_step": global_step,
            "best_validation": best_validation,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
            "model_init": {
                "max_paths": int(cache["b3_multipath_config"]["max_paths"]),
                "map_id_to_index": cache["map_id_to_index"],
                "condition_mean": cache["b3_statistics"]["condition_mean"],
                "condition_std": cache["b3_statistics"]["condition_std"],
                "constraints": cache["b3_multipath_config"]["constraints"],
                **config["delay_power_residual"],
            },
            "loss_init": {
                "max_paths": int(cache["b3_multipath_config"]["max_paths"]),
                "power_db_mean": float(
                    cache["b3_statistics"]["multipath"]["log_power_fraction_db_mean"]
                ),
                "power_db_std": float(
                    cache["b3_statistics"]["multipath"]["log_power_fraction_db_std"]
                ),
                "seed": int(train["seed"]),
                **{key: value for key, value in config["distribution_loss"].items() if key != "weights"},
                "weights": config["distribution_loss"]["weights"],
            },
            "cache_contract": {
                "path": str(Path(args.cache).resolve()),
                "scene_plan_fingerprint": cache["scene_plan_fingerprint"],
                "b3_checkpoint_sha256": cache["b3_checkpoint_sha256"],
                "forced_to_reference_path_count": True,
            },
            "system": cache["system"],
            "history": history,
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    patience = 0
    for epoch in range(start_epoch, int(train["epochs"])):
        started = time.monotonic()
        train_metrics = run_epoch(epoch, training=True)
        validation_metrics = run_epoch(epoch, training=False)
        scheduler.step(validation_metrics["loss"])
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "seconds": time.monotonic() - started,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        improved = validation_metrics["loss"] < best_validation
        if improved:
            best_validation = validation_metrics["loss"]
            patience = 0
            save(checkpoint_dir / "best.pt", epoch)
        else:
            patience += 1
        save(checkpoint_dir / "last.pt", epoch)
        atomic_json_dump(history, checkpoint_dir / "history.json")
        if args.dry_run or patience >= int(train.get("patience", 15)):
            break


if __name__ == "__main__":
    main()
