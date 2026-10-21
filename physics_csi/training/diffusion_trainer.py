from __future__ import annotations

import copy
import json
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from physics_csi.training.trainer import move_to_device, resolve_device
from physics_csi.utils.io import atomic_json_dump


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, online: nn.Module) -> None:
        online_state = online.state_dict()
        for name, value in self.model.state_dict().items():
            source = online_state[name].detach()
            if value.is_floating_point() or value.is_complex():
                value.lerp_(source, 1.0 - self.decay)
            else:
                value.copy_(source)


class DiffusionTrainer:
    """Shared trainer for the two independent diffusion branches."""

    def __init__(
        self,
        model: nn.Module,
        train_config: Mapping[str, Any],
        full_config: Mapping[str, Any],
        statistics: Mapping[str, Any],
        model_kind: str,
    ) -> None:
        self.config = dict(train_config)
        self.full_config = dict(full_config)
        self.statistics = dict(statistics)
        self.model_kind = model_kind
        self.device = resolve_device(str(self.config.get("device", "auto")))
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config.get("weight_decay", 0.0)),
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=float(self.config.get("lr_reduce_factor", 0.5)),
            patience=int(self.config.get("lr_patience", 10)),
        )
        self.ema = ExponentialMovingAverage(self.model, float(self.config.get("ema_decay", 0.999)))
        self.use_amp = bool(self.config.get("mixed_precision", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
            init_scale=float(self.config.get("amp_initial_scale", 4096.0)),
            growth_interval=int(self.config.get("amp_growth_interval", 2000)),
        )
        self.start_epoch = 0
        self.global_step = 0
        self.best_validation = float("inf")
        self.history: list[dict[str, Any]] = []

    def fit(self, loaders: Mapping[str, Any]) -> list[dict[str, Any]]:
        checkpoint_dir = Path(self.config["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        patience_counter = 0
        for epoch in range(self.start_epoch, int(self.config["epochs"])):
            started = time.monotonic()
            train_metrics = self._run_epoch(loaders["train"], train=True)
            validation_metrics = self._run_epoch(loaders["validation"], train=False)
            self.scheduler.step(validation_metrics["loss"])
            record = {
                "epoch": epoch,
                "seconds": time.monotonic() - started,
                "global_step": self.global_step,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "validation": validation_metrics,
            }
            self.history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            improved = validation_metrics["loss"] < self.best_validation
            if improved:
                self.best_validation = validation_metrics["loss"]
                patience_counter = 0
                self.save_checkpoint(checkpoint_dir / "best.pt", epoch)
            else:
                patience_counter += 1
            self.save_checkpoint(checkpoint_dir / "last.pt", epoch)
            atomic_json_dump(self.history, checkpoint_dir / "history.json")
            if patience_counter >= int(self.config.get("patience", 50)):
                break
        return self.history

    def _run_epoch(self, loader: Any, *, train: bool) -> dict[str, float]:
        active_model = self.model if train else self.ema.model
        active_model.train(train)
        totals: dict[str, float] = defaultdict(float)
        samples = 0
        skipped_updates = 0
        consecutive_skipped_updates = 0
        context = torch.enable_grad if train else torch.no_grad
        with context():
            for step, batch in enumerate(loader):
                batch = move_to_device(batch, self.device)
                batch_size = int(batch["conditions"].shape[0])
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    metrics = active_model(batch)
                if train:
                    self.scaler.scale(metrics["loss"]).backward()
                    self.scaler.unscale_(self.optimizer)
                    scene_gradient_norm = getattr(
                        self.model, "scene_gradient_norm", lambda: None
                    )()
                    if scene_gradient_norm is not None:
                        metrics = dict(metrics)
                        metrics["scene_gradient_norm"] = scene_gradient_norm
                        metrics["scene_gradient_nonzero"] = (
                            scene_gradient_norm > 0.0
                        ).float()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), float(self.config.get("gradient_clip_norm", 1.0))
                    )
                    if not torch.isfinite(gradient_norm) and not self.use_amp:
                        raise FloatingPointError("non-finite diffusion gradient norm")
                    scale_before = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    step_was_skipped = self.use_amp and self.scaler.get_scale() < scale_before
                    if step_was_skipped:
                        skipped_updates += 1
                        consecutive_skipped_updates += 1
                        print(
                            json.dumps(
                                {
                                    "event": "amp_nonfinite_gradient_skipped",
                                    "epoch_step": step,
                                    "global_step": self.global_step,
                                    "scale_before": scale_before,
                                    "scale_after": self.scaler.get_scale(),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        if consecutive_skipped_updates >= int(
                            self.config.get("max_consecutive_skipped_updates", 20)
                        ):
                            raise FloatingPointError(
                                "AMP skipped too many consecutive optimizer updates"
                            )
                    else:
                        consecutive_skipped_updates = 0
                        self.ema.update(self.model)
                        self.global_step += 1
                for key, value in metrics.items():
                    totals[key] += float(value.detach()) * batch_size
                samples += batch_size
                if train and step % int(self.config.get("log_every_steps", 20)) == 0:
                    message = {"epoch_step": step, "global_step": self.global_step}
                    message.update({key: float(value.detach()) for key, value in metrics.items()})
                    print(json.dumps(message, sort_keys=True), flush=True)
        if samples == 0:
            raise ValueError("data loader produced no samples")
        result = {key: value / samples for key, value in totals.items()}
        if train:
            result["skipped_nonfinite_updates"] = float(skipped_updates)
            if (
                bool(getattr(self.model, "scene_conditioning_enabled", False))
                and result.get("scene_gradient_nonzero", 0.0) == 0.0
            ):
                raise RuntimeError(
                    "scene-conditioning parameters received no gradients during the epoch"
                )
        return result

    def save_checkpoint(self, path: str | Path, epoch: int) -> None:
        payload = {
            "format_version": 2,
            "model_kind": self.model_kind,
            "epoch": epoch,
            "global_step": self.global_step,
            "best_validation": self.best_validation,
            "model_state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": self.full_config,
            "statistics": self.statistics,
            "history": self.history,
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def resume(self, path: str | Path) -> None:
        # RNG states must remain CPU ByteTensors.  Loading the complete
        # checkpoint directly onto CUDA also moves these tensors and makes
        # torch.cuda.set_rng_state reject them.  Module/optimizer loaders copy
        # their own state to the parameter device as needed.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_kind") != self.model_kind:
            raise ValueError("checkpoint belongs to a different diffusion model")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.ema.model.load_state_dict(checkpoint["ema_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_validation = float(checkpoint["best_validation"])
        self.history = list(checkpoint.get("history", []))
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if self.device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
            cuda_states = [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
            visible_devices = torch.cuda.device_count()
            for device_index, state in enumerate(cuda_states[:visible_devices]):
                torch.cuda.set_rng_state(state, device=device_index)

