from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from physics_csi.training.losses import CVAELoss
from physics_csi.utils.io import atomic_json_dump


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return value
    return value


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_function: CVAELoss,
        train_config: Mapping[str, Any],
        full_config: Mapping[str, Any],
    ) -> None:
        self.config = dict(train_config)
        self.full_config = dict(full_config)
        self.device = resolve_device(str(self.config["device"]))
        self.model = model.to(self.device)
        self.loss_function = loss_function.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )
        self.use_amp = bool(self.config["mixed_precision"]) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.start_epoch = 0
        self.best_validation = float("inf")
        self.history: list[dict[str, Any]] = []

    def fit(self, loaders: Mapping[str, Any]) -> list[dict[str, Any]]:
        checkpoint_dir = Path(self.config["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        patience_counter = 0
        for epoch in range(self.start_epoch, int(self.config["epochs"])):
            start = time.monotonic()
            beta = min(1.0, (epoch + 1) / max(1, int(self.config["kl_warmup_epochs"])))
            train_metrics = self._run_epoch(loaders["train"], train=True, beta=beta)
            validation_metrics = self._run_epoch(
                loaders["validation"], train=False, beta=beta
            )
            self.scheduler.step(validation_metrics["loss"])
            record = {
                "epoch": epoch,
                "seconds": time.monotonic() - start,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "beta": beta,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            self.history.append(record)
            print(json.dumps(record, sort_keys=True))
            improved = validation_metrics["loss"] < self.best_validation
            if improved:
                self.best_validation = validation_metrics["loss"]
                patience_counter = 0
                self.save_checkpoint(checkpoint_dir / "best.pt", epoch)
            else:
                patience_counter += 1
            self.save_checkpoint(checkpoint_dir / "last.pt", epoch)
            atomic_json_dump(self.history, checkpoint_dir / "history.json")
            if patience_counter >= int(self.config["patience"]):
                break
        return self.history

    @torch.no_grad()
    def evaluate(self, loader: Any, beta: float = 1.0) -> dict[str, float]:
        return self._run_epoch(loader, train=False, beta=beta)

    def _run_epoch(self, loader: Any, *, train: bool, beta: float) -> dict[str, float]:
        self.model.train(train)
        totals: dict[str, float] = defaultdict(float)
        samples = 0
        context = torch.enable_grad if train else torch.no_grad
        with context():
            for step, batch in enumerate(loader):
                batch = move_to_device(batch, self.device)
                batch_size = int(batch["path_mask"].shape[0])
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    output = self.model(batch)
                    metrics = self.loss_function(output, batch, beta=beta)
                if train:
                    self.scaler.scale(metrics["loss"]).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), float(self.config["gradient_clip_norm"])
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                for key, value in metrics.items():
                    totals[key] += float(value.detach()) * batch_size
                samples += batch_size
                if train and step % int(self.config["log_every_steps"]) == 0:
                    print(
                        json.dumps(
                            {
                                "step": step,
                                "loss": float(metrics["loss"].detach()),
                                "csi_nmse": float(metrics["csi_nmse"]),
                            }
                        )
                    )
        if samples == 0:
            raise ValueError("Data loader produced no samples")
        return {key: value / samples for key, value in totals.items()}

    def save_checkpoint(self, path: str | Path, epoch: int) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "epoch": epoch,
            "best_validation": self.best_validation,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": self.full_config,
            "history": self.history,
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        torch.save(payload, path)

    def resume(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_validation = float(checkpoint["best_validation"])
        self.history = list(checkpoint.get("history", []))
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if self.device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])

