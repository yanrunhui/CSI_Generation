from __future__ import annotations

import math
import random
from typing import Any, Literal, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from physics_csi.data.geometry import geometry_matrix_for_map_ids
from physics_csi.evaluation.research_diagnostics import (
    AOD_FEATURE_NAMES,
    POWER_AOD_FEATURE_NAMES,
    aod_statistical_features,
    condition_inputs,
    power_aod_joint_features,
    spatial_link_features,
)


ProbeMode = Literal["physics", "map", "geometry"]


class GeometryProbeRegressor(nn.Module):
    """Matched lightweight regressors for the P0/P1/P2 geometry probe."""

    def __init__(
        self,
        condition_dim: int,
        geometry_dim: int,
        target_dim: int,
        map_count: int,
        *,
        mode: ProbeMode,
        hidden_dim: int,
        context_dim: int,
        geometry_hidden_dim: int,
    ) -> None:
        super().__init__()
        if mode not in {"physics", "map", "geometry"}:
            raise ValueError(f"unsupported probe mode: {mode}")
        self.mode = mode
        # Construct the shared predictor first so a fixed seed gives every
        # ablation the same backbone initialization.
        self.predictor = nn.Sequential(
            nn.Linear(condition_dim + context_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, target_dim),
        )
        self.null_context = (
            nn.Parameter(torch.zeros(context_dim)) if mode == "physics" else None
        )
        self.map_embedding = (
            nn.Embedding(map_count, context_dim) if mode == "map" else None
        )
        self.geometry_encoder = (
            nn.Sequential(
                nn.Linear(geometry_dim, geometry_hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(geometry_hidden_dim),
                nn.Linear(geometry_hidden_dim, context_dim),
            )
            if mode == "geometry"
            else None
        )

    def forward(
        self,
        conditions: torch.Tensor,
        map_index: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "physics":
            assert self.null_context is not None
            context = self.null_context.unsqueeze(0).expand(conditions.shape[0], -1)
        elif self.mode == "map":
            assert self.map_embedding is not None
            if bool((map_index < 0).any()):
                raise ValueError("Map probe cannot evaluate an unseen Map ID")
            context = self.map_embedding(map_index)
        else:
            assert self.geometry_encoder is not None
            context = self.geometry_encoder(geometry)
        return self.predictor(torch.cat([conditions, context], dim=-1))


class ResidualGeometryProbe(nn.Module):
    """Frozen P0 plus a bounded, zero-output-initialized geometry correction."""

    def __init__(
        self,
        base_model: GeometryProbeRegressor,
        condition_dim: int,
        geometry_dim: int,
        target_dim: int,
        *,
        context_dim: int,
        geometry_hidden_dim: int,
        residual_hidden_dim: int,
        geometry_dropout: float,
        maximum_gate: float,
        initial_gate: float,
    ) -> None:
        super().__init__()
        if base_model.mode != "physics":
            raise ValueError("residual geometry probe requires a physics-only P0")
        if not 0.0 < initial_gate < maximum_gate:
            raise ValueError("initial_gate must be between zero and maximum_gate")
        if not 0.0 <= geometry_dropout < 1.0:
            raise ValueError("geometry_dropout must be in [0, 1)")
        self.base_model = base_model
        self.base_model.eval()
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.maximum_gate = float(maximum_gate)
        gate_fraction = initial_gate / maximum_gate
        self.gate_logit = nn.Parameter(
            torch.tensor(math.log(gate_fraction / (1.0 - gate_fraction)), dtype=torch.float32)
        )
        self.geometry_encoder = nn.Sequential(
            nn.Dropout(geometry_dropout),
            nn.Linear(geometry_dim, geometry_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(geometry_hidden_dim),
            nn.Linear(geometry_hidden_dim, context_dim),
        )
        self.residual_network = nn.Sequential(
            nn.Linear(condition_dim + context_dim, residual_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(residual_hidden_dim),
            nn.Linear(residual_hidden_dim, residual_hidden_dim),
            nn.SiLU(),
            nn.Linear(residual_hidden_dim, target_dim),
        )
        final = self.residual_network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @property
    def gate(self) -> torch.Tensor:
        return self.maximum_gate * torch.sigmoid(self.gate_logit)

    def train(self, mode: bool = True) -> "ResidualGeometryProbe":
        super().train(mode)
        self.base_model.eval()
        return self

    def forward(
        self,
        conditions: torch.Tensor,
        map_index: torch.Tensor,
        geometry: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            base = self.base_model(conditions, map_index, geometry)
        context = self.geometry_encoder(geometry)
        raw_delta = self.residual_network(torch.cat([conditions, context], dim=-1))
        correction = self.gate * torch.tanh(raw_delta)
        return base + correction, correction


def collect_probe_arrays(
    dataset: Dataset,
    *,
    tx_array_shape: tuple[int, int],
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    chunks: dict[str, list[torch.Tensor]] = {
        "conditions": [],
        "condition_validity": [],
        "map_id": [],
        "target": [],
    }
    target_names: tuple[str, ...] | None = None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    for batch in loader:
        spatial, spatial_names = spatial_link_features(batch["csi"], tx_array_shape)
        target = torch.cat(
            [
                aod_statistical_features(batch),
                power_aod_joint_features(batch),
                spatial,
            ],
            dim=-1,
        )
        names = tuple(AOD_FEATURE_NAMES) + tuple(POWER_AOD_FEATURE_NAMES) + spatial_names
        if target_names is None:
            target_names = names
        elif names != target_names:
            raise RuntimeError("geometry probe target schema changed between batches")
        chunks["conditions"].append(batch["conditions"].float())
        chunks["condition_validity"].append(batch["condition_validity"].bool())
        chunks["map_id"].append(batch["map_id"].long())
        chunks["target"].append(target.float())
    if target_names is None:
        raise ValueError("probe dataset is empty")
    return {key: torch.cat(value) for key, value in chunks.items()} | {
        "target_names": target_names
    }


def prepare_probe_arrays(
    raw: Mapping[str, Mapping[str, Any]],
    geometry_artifact: Mapping[str, Any],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    train = raw["train"]
    condition_mean = train["conditions"].float().mean(0)
    condition_std = train["conditions"].float().std(0, unbiased=False).clamp_min(1e-6)
    k_valid = train["condition_validity"][:, 2].bool()
    if bool(k_valid.any()):
        condition_mean[2] = train["conditions"][k_valid, 2].float().mean()
        condition_std[2] = train["conditions"][k_valid, 2].float().std(
            unbiased=False
        ).clamp_min(1e-6)
    target_mean = train["target"].float().mean(0)
    target_std = train["target"].float().std(0, unbiased=False).clamp_min(1e-6)
    training_maps = sorted(int(value) for value in train["map_id"].unique())
    map_id_to_index = {map_id: index for index, map_id in enumerate(training_maps)}
    unique_training_geometry = geometry_matrix_for_map_ids(
        geometry_artifact, torch.tensor(training_maps)
    )
    geometry_mean = unique_training_geometry.mean(0)
    raw_geometry_std = unique_training_geometry.std(0, unbiased=False)
    geometry_active = raw_geometry_std > 1e-6
    geometry_std = torch.where(
        geometry_active, raw_geometry_std, torch.ones_like(raw_geometry_std)
    )
    prepared: dict[str, dict[str, torch.Tensor]] = {}
    for split, values in raw.items():
        map_ids = values["map_id"].long()
        geometry = geometry_matrix_for_map_ids(geometry_artifact, map_ids)
        standardized_geometry = (geometry - geometry_mean) / geometry_std
        standardized_geometry[:, ~geometry_active] = 0.0
        prepared[split] = {
            "x": condition_inputs(
                values["conditions"],
                values["condition_validity"],
                condition_mean,
                condition_std,
            ),
            "map_index": torch.tensor(
                [map_id_to_index.get(int(value), -1) for value in map_ids],
                dtype=torch.long,
            ),
            "map_id": map_ids,
            "geometry": standardized_geometry,
            "y": (values["target"].float() - target_mean) / target_std,
        }
    metadata = {
        "training_maps": training_maps,
        "map_id_to_index": {str(key): value for key, value in map_id_to_index.items()},
        "condition_mean": condition_mean.tolist(),
        "condition_std": condition_std.tolist(),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "geometry_mean": geometry_mean.tolist(),
        "geometry_std": geometry_std.tolist(),
        "geometry_active_mask": geometry_active.tolist(),
        "geometry_active_features": int(geometry_active.sum()),
        "geometry_standardization": "unique training Maps receive equal weight",
    }
    return prepared, metadata


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(arrays: Mapping[str, torch.Tensor], batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        TensorDataset(
            arrays["x"], arrays["map_index"], arrays["geometry"], arrays["y"]
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


@torch.inference_mode()
def probe_loss(
    model: GeometryProbeRegressor,
    arrays: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for conditions, map_index, geometry, target in _loader(arrays, batch_size, False, 0):
        prediction = model(
            conditions.to(device), map_index.to(device), geometry.to(device)
        )
        total += float((prediction - target.to(device)).square().sum())
        count += target.numel()
    return total / max(count, 1)


def fit_probe(
    arrays: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    mode: ProbeMode,
    map_count: int,
    hidden_dim: int,
    context_dim: int,
    geometry_hidden_dim: int,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> tuple[GeometryProbeRegressor, dict[str, Any]]:
    _seed_everything(seed)
    model = GeometryProbeRegressor(
        arrays["train"]["x"].shape[1],
        arrays["train"]["geometry"].shape[1],
        arrays["train"]["y"].shape[1],
        map_count,
        mode=mode,
        hidden_dim=hidden_dim,
        context_dim=context_dim,
        geometry_hidden_dim=geometry_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_loss = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float | int]] = []
    train_loader = _loader(arrays["train"], batch_size, True, seed)
    for epoch in range(epochs):
        model.train()
        train_sum = 0.0
        train_count = 0
        for conditions, map_index, geometry, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                conditions.to(device), map_index.to(device), geometry.to(device)
            )
            loss = torch.nn.functional.mse_loss(prediction, target.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite geometry probe loss")
            loss.backward()
            optimizer.step()
            train_sum += float(loss.detach()) * target.numel()
            train_count += target.numel()
        validation_loss = probe_loss(
            model, arrays["validation"], device=device, batch_size=batch_size
        )
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
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("geometry probe did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_validation_standardized_mse": best_loss,
        "epochs_completed": len(history),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
    }


@torch.inference_mode()
def predict_probe(
    model: GeometryProbeRegressor,
    arrays: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
    map_override: torch.Tensor | None = None,
    geometry_override: torch.Tensor | None = None,
) -> torch.Tensor:
    model.eval()
    map_index = arrays["map_index"] if map_override is None else map_override
    geometry = arrays["geometry"] if geometry_override is None else geometry_override
    loader = DataLoader(
        TensorDataset(arrays["x"], map_index, geometry),
        batch_size=batch_size,
        shuffle=False,
    )
    chunks = []
    for conditions, current_map, current_geometry in loader:
        chunks.append(
            model(
                conditions.to(device),
                current_map.to(device),
                current_geometry.to(device),
            ).cpu()
        )
    return torch.cat(chunks)


def probe_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    feature_names: tuple[str, ...],
    map_ids: torch.Tensor,
) -> dict[str, Any]:
    squared_error = (prediction - target).square()
    target_variance = target.var(0, unbiased=False).clamp_min(1e-12)
    feature_mse = squared_error.mean(0)
    feature_r2 = 1.0 - feature_mse / target_variance
    prefixes = {
        "aod": "aod_",
        "power_aod": "power_aod_",
        "spatial_correlation": "tx_corr_",
    }
    groups: dict[str, Any] = {}
    for group, prefix in prefixes.items():
        indices = [index for index, name in enumerate(feature_names) if name.startswith(prefix)]
        groups[group] = {
            "features": len(indices),
            "standardized_mse": float(squared_error[:, indices].mean()),
            "mean_feature_r2": float(feature_r2[indices].mean()),
        }
    per_map: dict[str, Any] = {}
    for map_id in torch.unique(map_ids, sorted=True):
        keep = map_ids == map_id
        per_map[f"map_{int(map_id)}"] = {
            "samples": int(keep.sum()),
            "standardized_mse": float(squared_error[keep].mean()),
        }
    map_mse = torch.tensor([item["standardized_mse"] for item in per_map.values()])
    return {
        "samples": int(target.shape[0]),
        "maps": len(per_map),
        "standardized_mse_micro": float(squared_error.mean()),
        "standardized_mse_map_macro": float(map_mse.mean()),
        "mean_feature_r2": float(feature_r2.mean()),
        "groups": groups,
        "per_map": per_map,
    }


def cyclic_context_overrides(
    arrays: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return wrong Map indices and wrong geometry, shifted by distinct Map ID."""
    map_ids = arrays["map_id"].long()
    unique_maps = torch.unique(map_ids, sorted=True)
    if unique_maps.numel() < 2:
        raise ValueError("context shuffle requires at least two test maps")
    shifted = {
        int(map_id): int(unique_maps[(index + 1) % unique_maps.numel()])
        for index, map_id in enumerate(unique_maps)
    }
    wrong_geometry = torch.empty_like(arrays["geometry"])
    wrong_map_index = torch.empty_like(arrays["map_index"])
    for map_id in unique_maps:
        source_keep = map_ids == map_id
        donor_map = shifted[int(map_id)]
        donor_keep = map_ids == donor_map
        donor_row = torch.nonzero(donor_keep, as_tuple=False).flatten()[0]
        wrong_geometry[source_keep] = arrays["geometry"][donor_row]
        donor_map_indices = arrays["map_index"][donor_keep]
        wrong_map_index[source_keep] = donor_map_indices[0]
    return wrong_map_index, wrong_geometry


def geometry_variant_indices(
    feature_names: list[str] | tuple[str, ...], variant: str
) -> list[int]:
    """Nested feature sets: G1 density/size, G2 + orientation, G3 + layout."""
    if variant not in {"G1", "G2", "G3"}:
        raise ValueError("geometry variant must be G1, G2, or G3")
    orientation_prefix = "edge_orientation_length_fraction_"
    layout_prefix = "area_weighted_centroid_layout_"
    indices = []
    for index, name in enumerate(feature_names):
        is_orientation = name.startswith(orientation_prefix)
        is_layout = name.startswith(layout_prefix)
        if variant == "G1" and (is_orientation or is_layout):
            continue
        if variant == "G2" and is_layout:
            continue
        indices.append(index)
    if not indices:
        raise ValueError(f"{variant} selected no geometry features")
    return indices


def subset_geometry_arrays(
    arrays: Mapping[str, Mapping[str, torch.Tensor]], indices: list[int]
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        split: {
            key: value[:, indices] if key == "geometry" else value
            for key, value in values.items()
        }
        for split, values in arrays.items()
    }


def geometry_ood_audit(
    geometry_artifact: Mapping[str, Any],
    *,
    training_maps: list[int],
    evaluation_maps: list[int],
    feature_indices: list[int],
) -> dict[str, Any]:
    """Nearest training-Map distance in train-standardized geometry space."""
    training = geometry_matrix_for_map_ids(
        geometry_artifact, torch.tensor(training_maps, dtype=torch.long)
    )[:, feature_indices]
    evaluation = geometry_matrix_for_map_ids(
        geometry_artifact, torch.tensor(evaluation_maps, dtype=torch.long)
    )[:, feature_indices]
    mean = training.mean(0)
    raw_std = training.std(0, unbiased=False)
    active = raw_std > 1e-6
    if not bool(active.any()):
        raise ValueError("selected geometry variant has no varying training features")
    training_z = (training[:, active] - mean[active]) / raw_std[active]
    evaluation_z = (evaluation[:, active] - mean[active]) / raw_std[active]
    distance = torch.cdist(evaluation_z, training_z) / math.sqrt(int(active.sum()))
    nearest_distance, nearest_index = distance.min(dim=1)
    maximum_absolute_z = evaluation_z.abs().max(dim=1).values
    per_map = {
        f"map_{map_id}": {
            "nearest_training_map": int(training_maps[int(nearest_index[index])]),
            "nearest_standardized_rms_distance": float(nearest_distance[index]),
            "maximum_absolute_feature_z": float(maximum_absolute_z[index]),
        }
        for index, map_id in enumerate(evaluation_maps)
    }
    return {
        "selected_features": len(feature_indices),
        "active_training_features": int(active.sum()),
        "nearest_standardized_rms_distance": {
            "mean": float(nearest_distance.mean()),
            "median": float(nearest_distance.median()),
            "p90": float(torch.quantile(nearest_distance, 0.9)),
            "maximum": float(nearest_distance.max()),
        },
        "maximum_absolute_feature_z": {
            "mean": float(maximum_absolute_z.mean()),
            "p90": float(torch.quantile(maximum_absolute_z, 0.9)),
            "maximum": float(maximum_absolute_z.max()),
        },
        "per_map": per_map,
    }


@torch.inference_mode()
def residual_probe_loss(
    model: ResidualGeometryProbe,
    arrays: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for conditions, map_index, geometry, target in _loader(arrays, batch_size, False, 0):
        prediction, _ = model(
            conditions.to(device), map_index.to(device), geometry.to(device)
        )
        total += float((prediction - target.to(device)).square().sum())
        count += target.numel()
    return total / max(count, 1)


def fit_residual_geometry_probe(
    base_model: GeometryProbeRegressor,
    arrays: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    context_dim: int,
    geometry_hidden_dim: int,
    residual_hidden_dim: int,
    geometry_dropout: float,
    maximum_gate: float,
    initial_gate: float,
    correction_penalty: float,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> tuple[ResidualGeometryProbe, dict[str, Any]]:
    _seed_everything(seed ^ 0x47E0)
    model = ResidualGeometryProbe(
        base_model,
        arrays["train"]["x"].shape[1],
        arrays["train"]["geometry"].shape[1],
        arrays["train"]["y"].shape[1],
        context_dim=context_dim,
        geometry_hidden_dim=geometry_hidden_dim,
        residual_hidden_dim=residual_hidden_dim,
        geometry_dropout=geometry_dropout,
        maximum_gate=maximum_gate,
        initial_gate=initial_gate,
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=weight_decay
    )
    best_loss = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float | int]] = []
    train_loader = _loader(arrays["train"], batch_size, True, seed ^ 0x47E0)
    for epoch in range(epochs):
        model.train()
        train_mse_sum = 0.0
        train_correction_sum = 0.0
        count = 0
        for conditions, map_index, geometry, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction, correction = model(
                conditions.to(device), map_index.to(device), geometry.to(device)
            )
            mse = torch.nn.functional.mse_loss(prediction, target.to(device))
            correction_cost = correction.square().mean()
            loss = mse + float(correction_penalty) * correction_cost
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite residual geometry probe loss")
            loss.backward()
            optimizer.step()
            train_mse_sum += float(mse.detach()) * target.numel()
            train_correction_sum += float(correction_cost.detach()) * target.numel()
            count += target.numel()
        validation_loss = residual_probe_loss(
            model, arrays["validation"], device=device, batch_size=batch_size
        )
        history.append(
            {
                "epoch": epoch,
                "train_standardized_mse": train_mse_sum / max(count, 1),
                "train_correction_mean_square": train_correction_sum / max(count, 1),
                "validation_standardized_mse": validation_loss,
                "gate": float(model.gate.detach()),
            }
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("residual probe did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_validation_standardized_mse": best_loss,
        "epochs_completed": len(history),
        "gate": float(model.gate.detach()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "history": history,
    }


@torch.inference_mode()
def predict_residual_geometry_probe(
    model: ResidualGeometryProbe,
    arrays: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
    geometry_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    model.eval()
    geometry = arrays["geometry"] if geometry_override is None else geometry_override
    loader = DataLoader(
        TensorDataset(arrays["x"], arrays["map_index"], geometry),
        batch_size=batch_size,
        shuffle=False,
    )
    predictions = []
    corrections = []
    for conditions, map_index, current_geometry in loader:
        prediction, correction = model(
            conditions.to(device), map_index.to(device), current_geometry.to(device)
        )
        predictions.append(prediction.cpu())
        corrections.append(correction.cpu())
    correction = torch.cat(corrections)
    absolute = correction.abs()
    diagnostics = {
        "gate": float(model.gate.detach()),
        "correction_rms": float(correction.square().mean().sqrt()),
        "correction_absolute_mean": float(absolute.mean()),
        "correction_absolute_p95": float(torch.quantile(absolute, 0.95)),
        "correction_absolute_maximum": float(absolute.max()),
    }
    return torch.cat(predictions), diagnostics
