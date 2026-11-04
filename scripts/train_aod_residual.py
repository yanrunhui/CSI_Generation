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
from physics_csi.models.scene_aod_residual import SceneConditionedAoDResidual
from physics_csi.models.scene_delay_power_residual import SceneConditionedDelayPowerResidual
from physics_csi.models.scene_link_global_aod_residual import (
    SceneConditionedLinkGlobalAoDResidual,
)
from physics_csi.models.scene_relative_aod_residual import (
    SceneConditionedRelativeAoDResidual,
)
from physics_csi.training.aod_residual import AoDDistributionLoss, AoDLossWeights
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


def _load_frozen_dp(
    checkpoint: Mapping[str, Any], device: torch.device
) -> SceneConditionedDelayPowerResidual:
    if checkpoint.get("model_kind") != "scene_delay_power_residual":
        raise ValueError("--dp-checkpoint is not a trained Delay--Power residual")
    model = SceneConditionedDelayPowerResidual(**checkpoint["model_init"]).to(device)
    model.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))
    model.requires_grad_(False).eval()
    return model


def _architecture(config: Mapping[str, Any]) -> tuple[str, str, type[torch.nn.Module]]:
    name = str(config.get("aod_architecture", "per_path"))
    if name == "per_path":
        return "scene_aod_residual", "aod_residual", SceneConditionedAoDResidual
    if name == "link_global":
        return (
            "scene_link_global_aod_residual",
            "link_global_aod_residual",
            SceneConditionedLinkGlobalAoDResidual,
        )
    if name == "relative_cluster":
        return (
            "scene_relative_aod_residual",
            "relative_aod_residual",
            SceneConditionedRelativeAoDResidual,
        )
    raise ValueError(f"unsupported AoD residual architecture: {name!r}")


def _residual_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    _, section, _ = _architecture(config)
    return config[section]


def _make_model(cache: Mapping[str, Any], config: Mapping[str, Any]) -> torch.nn.Module:
    _, _, model_class = _architecture(config)
    return model_class(
        max_paths=int(cache["b3_multipath_config"]["max_paths"]),
        map_id_to_index=cache["map_id_to_index"],
        condition_mean=cache["b3_statistics"]["condition_mean"],
        condition_std=cache["b3_statistics"]["condition_std"],
        **_residual_config(config),
    )


def _make_loss(cache: Mapping[str, Any], config: Mapping[str, Any]) -> AoDDistributionLoss:
    values = dict(config["distribution_loss"])
    weights = AoDLossWeights(**values.pop("weights"))
    stats = cache["b3_statistics"]["multipath"]
    residual = _residual_config(config)
    system = cache["system"]
    azimuth_normalization = float(
        values.pop(
            "residual_normalization_azimuth_deg",
            residual["maximum_azimuth_residual_deg"],
        )
    )
    elevation_normalization = float(
        values.pop(
            "residual_normalization_elevation_deg",
            residual["maximum_elevation_residual_deg"],
        )
    )
    return AoDDistributionLoss(
        max_paths=int(cache["b3_multipath_config"]["max_paths"]),
        power_db_mean=float(stats["log_power_fraction_db_mean"]),
        power_db_std=float(stats["log_power_fraction_db_std"]),
        maximum_azimuth_residual_deg=azimuth_normalization,
        maximum_elevation_residual_deg=elevation_normalization,
        tx_array_shape=tuple(system["tx_array_shape"]),
        tx_spacing_wavelengths=float(system["tx_spacing_wavelengths"]),
        tx_array_plane=str(system.get("tx_array_plane", "xy")),
        weights=weights,
        seed=int(config["train"]["seed"]),
        **values,
    )


def _different_map_indices(
    correct: torch.Tensor, map_count: int, generator: torch.Generator
) -> torch.Tensor:
    if map_count < 2:
        raise ValueError("shuffled-Map AoD training requires at least two Maps")
    offset = torch.randint(1, map_count, (), generator=generator, device=correct.device)
    return (correct + offset) % map_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Train conditional per-path AoD residual.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dp-checkpoint", required=True)
    parser.add_argument(
        "--link-global-checkpoint",
        default=None,
        help="Frozen Link-global parent; required for relative_cluster architecture.",
    )
    parser.add_argument("--config", default="configs/aod_residual.yaml")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    model_kind, _, _ = _architecture(config)
    residual_config = _residual_config(config)
    train = config["train"]
    if args.checkpoint_dir:
        train["checkpoint_dir"] = args.checkpoint_dir
    seed_everything(int(train["seed"]))
    device = resolve_device(str(train.get("device", "auto")))
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    dp_checkpoint = torch.load(args.dp_checkpoint, map_location="cpu", weights_only=False)
    if cache["scene_plan_fingerprint"] != dp_checkpoint["cache_contract"]["scene_plan_fingerprint"]:
        raise ValueError("cache and DP checkpoint use different scene plans")
    if cache["b3_checkpoint_sha256"] != dp_checkpoint["cache_contract"]["b3_checkpoint_sha256"]:
        raise ValueError("cache and DP checkpoint use different frozen B3 backbones")
    dp_model = _load_frozen_dp(dp_checkpoint, device)
    link_model: SceneConditionedLinkGlobalAoDResidual | None = None
    link_checkpoint: Mapping[str, Any] | None = None
    if str(config.get("aod_architecture", "per_path")) == "relative_cluster":
        if not args.link_global_checkpoint:
            raise ValueError(
                "--link-global-checkpoint is required for relative_cluster training"
            )
        link_checkpoint = torch.load(
            args.link_global_checkpoint, map_location="cpu", weights_only=False
        )
        if link_checkpoint.get("model_kind") != "scene_link_global_aod_residual":
            raise ValueError("--link-global-checkpoint is not a Link-global AoD model")
        for key, expected in (
            ("scene_plan_fingerprint", cache["scene_plan_fingerprint"]),
            ("b3_checkpoint_sha256", cache["b3_checkpoint_sha256"]),
        ):
            if link_checkpoint["cache_contract"][key] != expected:
                raise ValueError(f"Link-global checkpoint has a different {key}")
        link_model = SceneConditionedLinkGlobalAoDResidual(
            **link_checkpoint["model_init"]
        ).to(device)
        link_model.load_state_dict(
            link_checkpoint.get("ema_state_dict", link_checkpoint["model_state_dict"])
        )
        link_model.requires_grad_(False).eval()

    train_dataset = DelayPowerResidualCacheDataset(cache, "train")
    validation_dataset = DelayPowerResidualCacheDataset(cache, "validation")
    sampler_arguments = {
        "batch_size": int(train["batch_size"]),
        "minimum_batch_size": int(train["minimum_batch_size"]),
        "condition_std": cache["b3_statistics"]["condition_std"],
    }
    train_sampler = ConditionNeighborhoodBatchSampler(
        train_dataset,
        batches_per_epoch=int(train["train_batches_per_epoch"]),
        seed=int(train["seed"]),
        shuffle=True,
        **sampler_arguments,
    )
    validation_sampler = ConditionNeighborhoodBatchSampler(
        validation_dataset,
        batches_per_epoch=int(train["validation_batches_per_epoch"]),
        seed=int(train["seed"]) + 100000,
        shuffle=False,
        **sampler_arguments,
    )
    expected_maps = {int(value) for value in torch.unique(train_dataset.map_ids)}
    covered_maps = {int(key[0]) for key in train_sampler.strata}
    missing_maps = sorted(expected_maps - covered_maps)
    if missing_maps:
        raise ValueError(
            "AoD training has no eligible neighborhood for "
            + ", ".join(f"map_{value}" for value in missing_maps)
        )
    print(
        json.dumps(
            {
                "event": "aod_residual_neighborhood_audit",
                "train_samples": len(train_dataset),
                "validation_samples": len(validation_dataset),
                "train_maps": len(expected_maps),
                "train_strata": len(train_sampler.strata),
                "validation_strata": len(validation_sampler.strata),
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
    if model.map_id_to_index != dp_model.map_id_to_index:
        raise ValueError("AoD and DP residuals use different Map index mappings")
    if link_model is not None and model.map_id_to_index != link_model.map_id_to_index:
        raise ValueError("Relative and Link-global AoD models use different Map mappings")
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
        dp_model.eval()
        if link_model is not None:
            link_model.eval()
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
                with torch.no_grad():
                    dp_output = dp_model(
                        base, conditions, validity, map_index=correct_index
                    )
                    dp_payload = dp_model.apply_to_payload(base, dp_output)
                    if link_model is not None:
                        link_output = link_model(
                            dp_payload,
                            conditions,
                            validity,
                            map_index=correct_index,
                        )
                        aod_base_payload = link_model.apply_to_payload(
                            dp_payload, link_output
                        )
                    else:
                        aod_base_payload = dp_payload
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_amp
                ):
                    correct_output = active(
                        aod_base_payload, conditions, validity, map_index=correct_index
                    )
                    shuffled_output = active(
                        aod_base_payload, conditions, validity, map_index=wrong_index
                    )
                    correct_payload = active.apply_to_payload(
                        aod_base_payload, correct_output
                    )
                    shuffled_payload = active.apply_to_payload(
                        aod_base_payload, shuffled_output
                    )
                    metrics = loss_function(
                        correct_payload=correct_payload,
                        shuffled_payload=shuffled_payload,
                        dp_payload=aod_base_payload,
                        reference_payload=reference,
                        correct_output=correct_output,
                    )
                if training:
                    if not torch.isfinite(metrics["loss"]):
                        print(
                            json.dumps(
                                {
                                    "event": "aod_residual_nonfinite_loss",
                                    "epoch": epoch,
                                    "step": step,
                                    "metrics": {
                                        key: float(value.detach())
                                        for key, value in metrics.items()
                                    },
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        raise FloatingPointError(
                            "AoD residual produced a non-finite loss"
                        )
                    scaler.scale(metrics["loss"]).backward()
                    scaler.unscale_(optimizer)
                    nonfinite_parameters = [
                        name
                        for name, parameter in model.named_parameters()
                        if parameter.grad is not None
                        and not bool(torch.isfinite(parameter.grad).all())
                    ]
                    if nonfinite_parameters:
                        print(
                            json.dumps(
                                {
                                    "event": "aod_residual_nonfinite_gradient",
                                    "epoch": epoch,
                                    "step": step,
                                    "global_step": global_step,
                                    "parameters": nonfinite_parameters,
                                    "metrics": {
                                        key: float(value.detach())
                                        for key, value in metrics.items()
                                    },
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        raise FloatingPointError(
                            "AoD residual produced a non-finite gradient in: "
                            + ", ".join(nonfinite_parameters)
                        )
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(train.get("gradient_clip_norm", 1.0))
                    )
                    if not torch.isfinite(gradient_norm):
                        raise FloatingPointError("AoD residual produced a non-finite gradient")
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
                                "event": "aod_residual_step",
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
            raise RuntimeError("AoD neighborhood sampler produced no data")
        return {key: value / samples for key, value in totals.items()}

    checkpoint_dir = Path(train["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(path: Path, epoch: int) -> None:
        payload = {
            "format_version": 1,
            "model_kind": model_kind,
            "aod_architecture": str(config.get("aod_architecture", "per_path")),
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
                **residual_config,
            },
            "loss_init": {
                "max_paths": int(cache["b3_multipath_config"]["max_paths"]),
                "power_db_mean": float(
                    cache["b3_statistics"]["multipath"]["log_power_fraction_db_mean"]
                ),
                "power_db_std": float(
                    cache["b3_statistics"]["multipath"]["log_power_fraction_db_std"]
                ),
                "maximum_azimuth_residual_deg": float(
                    config["distribution_loss"].get(
                        "residual_normalization_azimuth_deg",
                        residual_config["maximum_azimuth_residual_deg"],
                    )
                ),
                "maximum_elevation_residual_deg": float(
                    config["distribution_loss"].get(
                        "residual_normalization_elevation_deg",
                        residual_config["maximum_elevation_residual_deg"],
                    )
                ),
                "tx_array_shape": tuple(cache["system"]["tx_array_shape"]),
                "tx_spacing_wavelengths": float(
                    cache["system"]["tx_spacing_wavelengths"]
                ),
                "tx_array_plane": str(cache["system"].get("tx_array_plane", "xy")),
                "seed": int(train["seed"]),
                **{
                    key: value
                    for key, value in config["distribution_loss"].items()
                    if key
                    not in {
                        "weights",
                        "residual_normalization_azimuth_deg",
                        "residual_normalization_elevation_deg",
                    }
                },
                "weights": config["distribution_loss"]["weights"],
            },
            "cache_contract": {
                "path": str(Path(args.cache).resolve()),
                "scene_plan_fingerprint": cache["scene_plan_fingerprint"],
                "b3_checkpoint_sha256": cache["b3_checkpoint_sha256"],
            },
            "dp_checkpoint": str(Path(args.dp_checkpoint).resolve()),
            "dp_checkpoint_epoch": int(dp_checkpoint["epoch"]),
            "parent_link_global_checkpoint": (
                str(Path(args.link_global_checkpoint).resolve())
                if args.link_global_checkpoint
                else None
            ),
            "parent_link_global_checkpoint_epoch": (
                int(link_checkpoint["epoch"]) if link_checkpoint is not None else None
            ),
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
