#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.config import load_yaml
from physics_csi.data.scene_id import load_seen_map_spatial_plan, make_scene_id_datasets
from physics_csi.evaluation.research_diagnostics import (
    collect_learning_arrays,
    condition_inputs,
)
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


class StatisticalPredictor(nn.Module):
    """Small conditional-mean predictor; deliberately unrelated to diffusion."""

    def __init__(
        self,
        condition_dim: int,
        target_dim: int,
        map_count: int,
        *,
        hidden_dim: int,
        embedding_dim: int,
        map_aware: bool,
    ) -> None:
        super().__init__()
        self.map_aware = map_aware
        # Build the shared MLP first so matched seeds give both ablations the
        # same backbone initialization.  Only the scene input differs.
        self.network = nn.Sequential(
            nn.Linear(condition_dim + embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, target_dim),
        )
        self.map_embedding = nn.Embedding(map_count, embedding_dim) if map_aware else None
        self.scene_bias = (
            None if map_aware else nn.Parameter(torch.zeros(embedding_dim))
        )

    def forward(self, conditions: torch.Tensor, map_index: torch.Tensor) -> torch.Tensor:
        if self.map_embedding is not None:
            scene = self.map_embedding(map_index)
        else:
            assert self.scene_bias is not None
            scene = self.scene_bias.unsqueeze(0).expand(conditions.shape[0], -1)
        return self.network(torch.cat([conditions, scene], dim=-1))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _condition_statistics(train: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    values = train["conditions"].float()
    validity = train["condition_validity"].bool()
    mean = values.mean(0)
    std = values.std(0, unbiased=False).clamp_min(1e-6)
    if bool(validity[:, 2].any()):
        mean[2] = values[validity[:, 2], 2].mean()
        std[2] = values[validity[:, 2], 2].std(unbiased=False).clamp_min(1e-6)
    return mean, std


def _prepare_arrays(
    arrays: dict[str, Any],
    map_id_to_index: dict[int, int],
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "x": condition_inputs(
            arrays["conditions"], arrays["condition_validity"], condition_mean, condition_std
        ),
        "map_index": torch.tensor(
            [map_id_to_index[int(value)] for value in arrays["map_id"]], dtype=torch.long
        ),
        "map_id": arrays["map_id"].long(),
        "y": (arrays["target"] - target_mean) / target_std,
    }


def _loss_on_arrays(
    model: nn.Module,
    arrays: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    loader = DataLoader(
        TensorDataset(arrays["x"], arrays["map_index"], arrays["y"]),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.inference_mode():
        for x, map_index, target in loader:
            prediction = model(x.to(device), map_index.to(device))
            loss = (prediction - target.to(device)).square().sum()
            total += float(loss)
            count += target.numel()
    return total / max(count, 1)


def _fit(
    arrays: dict[str, dict[str, torch.Tensor]],
    *,
    map_count: int,
    map_aware: bool,
    hidden_dim: int,
    embedding_dim: int,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> tuple[StatisticalPredictor, dict[str, Any]]:
    _seed_everything(seed)
    model = StatisticalPredictor(
        arrays["train"]["x"].shape[1],
        arrays["train"]["y"].shape[1],
        map_count,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        map_aware=map_aware,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            arrays["train"]["x"],
            arrays["train"]["map_index"],
            arrays["train"]["y"],
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        model.train()
        train_sum = 0.0
        train_count = 0
        for x, map_index, target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x.to(device), map_index.to(device))
            loss = torch.nn.functional.mse_loss(prediction, target.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite predictor loss")
            loss.backward()
            optimizer.step()
            train_sum += float(loss.detach()) * target.numel()
            train_count += target.numel()
        validation_loss = _loss_on_arrays(model, arrays["validation"], device, batch_size)
        history.append(
            {
                "epoch": epoch,
                "train_standardized_mse": train_sum / max(train_count, 1),
                "validation_standardized_mse": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("predictor did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_validation_standardized_mse": best_loss,
        "epochs_completed": len(history),
        "history": history,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def _predict(
    model: nn.Module,
    arrays: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    map_override: torch.Tensor | None = None,
) -> torch.Tensor:
    model.eval()
    map_index = arrays["map_index"] if map_override is None else map_override
    loader = DataLoader(
        TensorDataset(arrays["x"], map_index), batch_size=batch_size, shuffle=False
    )
    result: list[torch.Tensor] = []
    with torch.inference_mode():
        for x, current_map in loader:
            result.append(model(x.to(device), current_map.to(device)).cpu())
    return torch.cat(result)


def _metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    feature_names: tuple[str, ...],
    map_ids: torch.Tensor,
) -> dict[str, Any]:
    error = (prediction - target).square()
    target_variance = target.var(0, unbiased=False).clamp_min(1e-12)
    feature_mse = error.mean(0)
    feature_r2 = 1.0 - feature_mse / target_variance

    def group_indices(prefix: str) -> list[int]:
        return [index for index, name in enumerate(feature_names) if name.startswith(prefix)]

    groups = {
        "aod": group_indices("aod_"),
        "spatial_correlation": group_indices("tx_corr_"),
    }
    group_report = {
        name: {
            "standardized_mse": float(error[:, indices].mean()),
            "mean_feature_r2": float(feature_r2[indices].mean()),
        }
        for name, indices in groups.items()
        if indices
    }
    per_map = {
        f"map_{int(map_id)}": {
            "samples": int(keep.sum()),
            "standardized_mse": float(error[keep].mean()),
            "mean_feature_r2": float(
                (
                    1.0
                    - error[keep].mean(0)
                    / target[keep].var(0, unbiased=False).clamp_min(1e-12)
                ).mean()
            ),
        }
        for map_id in torch.unique(map_ids, sorted=True)
        for keep in [map_ids == map_id]
    }
    return {
        "standardized_mse": float(error.mean()),
        "mean_feature_r2": float(feature_r2.mean()),
        "groups": group_report,
        "per_feature": {
            name: {"standardized_mse": float(feature_mse[index]), "r2": float(feature_r2[index])}
            for index, name in enumerate(feature_names)
        },
        "per_map": per_map,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether Map ID improves lightweight prediction of real AoD/spatial statistics "
            "on the pre-registered receiver-spatial holdout."
        )
    )
    parser.add_argument("--scene-plan", required=True)
    parser.add_argument("--prepared-root", default="artifacts/prepared_d2los")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--target", choices=["aod", "spatial", "all"], default="all")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--feature-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.hidden_dim, args.embedding_dim, args.epochs, args.patience, args.batch_size) < 1:
        parser.error("model sizes, epochs, patience, and batch size must be positive")
    seeds = [int(value) for value in args.seeds.split(",")]
    if not seeds:
        parser.error("--seeds cannot be empty")

    device = resolve_device(args.device)
    plan = load_seen_map_spatial_plan(args.scene_plan)
    datasets = make_scene_id_datasets(args.prepared_root, plan)
    data_config = load_yaml(args.data_config)["data"]
    tx_array_shape = tuple(int(value) for value in data_config["system"]["tx_array_shape"])
    compact = {
        split: collect_learning_arrays(
            dataset,
            target=args.target,
            tx_array_shape=tx_array_shape,
            batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
        )
        for split, dataset in datasets.items()
    }
    feature_names = compact["train"]["target_names"]
    if any(payload["target_names"] != feature_names for payload in compact.values()):
        raise RuntimeError("target feature schemas differ across splits")
    condition_mean, condition_std = _condition_statistics(compact["train"])
    target_mean = compact["train"]["target"].mean(0)
    target_std = compact["train"]["target"].std(0, unbiased=False).clamp_min(1e-6)
    map_id_to_index = {int(key): int(value) for key, value in plan["map_id_to_index"].items()}
    arrays = {
        split: _prepare_arrays(
            payload,
            map_id_to_index,
            condition_mean,
            condition_std,
            target_mean,
            target_std,
        )
        for split, payload in compact.items()
    }
    map_count = len(map_id_to_index)
    if map_count < 2:
        raise ValueError("scene learnability requires at least two maps for the wrong-Map control")
    wrong_map = (arrays["test"]["map_index"] + 1) % map_count
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        physics_model, physics_fit = _fit(
            arrays,
            map_count=map_count,
            map_aware=False,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
            seed=seed,
        )
        scene_model, scene_fit = _fit(
            arrays,
            map_count=map_count,
            map_aware=True,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
            seed=seed,
        )
        target = arrays["test"]["y"]
        physics_prediction = _predict(physics_model, arrays["test"], device, args.batch_size)
        scene_prediction = _predict(scene_model, arrays["test"], device, args.batch_size)
        wrong_prediction = _predict(
            scene_model, arrays["test"], device, args.batch_size, map_override=wrong_map
        )
        physics_metrics = _metrics(
            physics_prediction, target, feature_names, arrays["test"]["map_id"]
        )
        scene_metrics = _metrics(
            scene_prediction, target, feature_names, arrays["test"]["map_id"]
        )
        wrong_metrics = _metrics(
            wrong_prediction, target, feature_names, arrays["test"]["map_id"]
        )
        group_reductions = {
            group: {
                "mse_reduction_vs_physics_only": (
                    physics_metrics["groups"][group]["standardized_mse"]
                    - scene_metrics["groups"][group]["standardized_mse"]
                ),
                "mse_reduction_vs_wrong_map": (
                    wrong_metrics["groups"][group]["standardized_mse"]
                    - scene_metrics["groups"][group]["standardized_mse"]
                ),
            }
            for group in scene_metrics["groups"]
        }
        map_names = sorted(scene_metrics["per_map"])
        runs.append(
            {
                "seed": seed,
                "fit": {"physics_only": physics_fit, "physics_plus_map": scene_fit},
                "test": {
                    "physics_only": physics_metrics,
                    "physics_plus_correct_map": scene_metrics,
                    "physics_plus_wrong_map": wrong_metrics,
                },
                "map_value": {
                    "mse_reduction_vs_physics_only": (
                        physics_metrics["standardized_mse"] - scene_metrics["standardized_mse"]
                    ),
                    "mse_reduction_vs_wrong_map": (
                        wrong_metrics["standardized_mse"] - scene_metrics["standardized_mse"]
                    ),
                    "groups": group_reductions,
                    "maps_correct_beats_physics_only": sum(
                        scene_metrics["per_map"][name]["standardized_mse"]
                        < physics_metrics["per_map"][name]["standardized_mse"]
                        for name in map_names
                    ),
                    "maps_correct_beats_wrong_map": sum(
                        scene_metrics["per_map"][name]["standardized_mse"]
                        < wrong_metrics["per_map"][name]["standardized_mse"]
                        for name in map_names
                    ),
                    "maps_total": len(map_names),
                },
            }
        )

    reductions_physics = torch.tensor(
        [run["map_value"]["mse_reduction_vs_physics_only"] for run in runs]
    )
    reductions_wrong = torch.tensor(
        [run["map_value"]["mse_reduction_vs_wrong_map"] for run in runs]
    )
    group_summary = {
        group: {
            "mean_mse_reduction_vs_physics_only": float(
                torch.tensor(
                    [run["map_value"]["groups"][group]["mse_reduction_vs_physics_only"] for run in runs]
                ).mean()
            ),
            "mean_mse_reduction_vs_wrong_map": float(
                torch.tensor(
                    [run["map_value"]["groups"][group]["mse_reduction_vs_wrong_map"] for run in runs]
                ).mean()
            ),
        }
        for group in runs[0]["map_value"]["groups"]
    }
    report = {
        "format_version": 1,
        "diagnostic": "real_scene_feature_learnability_without_diffusion",
        "scene_plan": str(Path(args.scene_plan).resolve()),
        "scene_plan_fingerprint": plan["fingerprint"],
        "prepared_root": str(Path(args.prepared_root).resolve()),
        "target": args.target,
        "target_features": list(feature_names),
        "split_samples": {split: int(values["y"].shape[0]) for split, values in arrays.items()},
        "protocol": {
            "input": "physical conditions plus optional categorical Map embedding",
            "output": "per-link real-data conditional statistics, standardized from training only",
            "holdout": "receiver-spatial train/validation/test plan shared with B3-S-ID",
            "models": "matched two-hidden-layer MLPs; no diffusion and no generated samples",
            "wrong_map": "cyclic shift over the stable plan mapping at test time",
        },
        "hyperparameters": {
            "hidden_dim": args.hidden_dim,
            "embedding_dim": args.embedding_dim,
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "seeds": seeds,
            "device": str(device),
        },
        "summary": {
            "mean_mse_reduction_correct_map_vs_physics_only": float(reductions_physics.mean()),
            "mean_mse_reduction_correct_map_vs_wrong_map": float(reductions_wrong.mean()),
            "seeds_correct_map_beats_physics_only": int((reductions_physics > 0).sum()),
            "seeds_correct_map_beats_wrong_map": int((reductions_wrong > 0).sum()),
            "seeds_total": len(seeds),
            "target_groups": group_summary,
            "interpretation": (
                "Stable positive reductions support learnable Map-specific statistics. A null result "
                "does not prove that geometry is irrelevant; it may reflect target noise or inadequate support."
            ),
        },
        "runs": runs,
    }
    atomic_json_dump(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
