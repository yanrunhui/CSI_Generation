from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


CONDITION_NAMES = (
    "total_power_db",
    "delay_spread_ns",
    "k_factor_db",
    "los_status",
)


def direction_vectors(azimuth_deg: torch.Tensor, elevation_deg: torch.Tensor) -> torch.Tensor:
    """Convert spherical azimuth/elevation to Cartesian unit vectors."""
    azimuth = torch.deg2rad(azimuth_deg.float())
    elevation = torch.deg2rad(elevation_deg.float())
    horizontal = elevation.cos()
    return torch.stack(
        [
            horizontal * azimuth.cos(),
            horizontal * azimuth.sin(),
            elevation.sin(),
        ],
        dim=-1,
    )


def normalized_path_power(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask = payload["path_mask"].bool()
    power = payload["path_powers_linear"].float() * mask
    return power / power.sum(-1, keepdim=True).clamp_min(1e-20)


def aod_statistical_features(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Per-link AoD summaries without introducing an azimuth wrap discontinuity.

    The result contains unweighted and power-weighted Cartesian means, their
    concentrations, and power-weighted RMS angular spread.  These are useful
    conditional-statistics targets; they are not intended to reconstruct every
    path of a link.
    """
    mask = payload["path_mask"].bool()
    numeric_mask = mask.float()
    count = numeric_mask.sum(-1, keepdim=True).clamp_min(1.0)
    directions = direction_vectors(payload["aod_az_deg"], payload["aod_el_deg"])
    uniform_mean = (directions * numeric_mask.unsqueeze(-1)).sum(1) / count
    weights = normalized_path_power(payload)
    weighted_mean = (directions * weights.unsqueeze(-1)).sum(1)
    uniform_concentration = uniform_mean.norm(dim=-1, keepdim=True).clamp(0.0, 1.0)
    weighted_concentration = weighted_mean.norm(dim=-1, keepdim=True).clamp(0.0, 1.0)
    mean_direction = weighted_mean / weighted_concentration.clamp_min(1e-8)
    offset = torch.acos(
        (directions * mean_direction.unsqueeze(1)).sum(-1).clamp(-1.0, 1.0)
    )
    rms_spread_deg = torch.rad2deg(
        (weights * offset.square() * numeric_mask).sum(-1).clamp_min(0.0).sqrt()
    ).unsqueeze(-1)
    return torch.cat(
        [
            uniform_mean,
            weighted_mean,
            uniform_concentration,
            weighted_concentration,
            rms_spread_deg,
        ],
        dim=-1,
    )


AOD_FEATURE_NAMES = (
    "aod_uniform_mean_x",
    "aod_uniform_mean_y",
    "aod_uniform_mean_z",
    "aod_power_weighted_mean_x",
    "aod_power_weighted_mean_y",
    "aod_power_weighted_mean_z",
    "aod_uniform_concentration",
    "aod_power_weighted_concentration",
    "aod_power_weighted_rms_spread_deg",
)


POWER_AOD_FEATURE_NAMES = (
    "power_aod_second_moment_xx",
    "power_aod_second_moment_yy",
    "power_aod_second_moment_zz",
    "power_aod_second_moment_xy",
    "power_aod_second_moment_xz",
    "power_aod_second_moment_yz",
    "power_aod_strongest_path_direction_x",
    "power_aod_strongest_path_direction_y",
    "power_aod_strongest_path_direction_z",
    "power_aod_max_power_fraction",
    "power_aod_normalized_power_entropy",
)


def power_aod_joint_features(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Per-link moments that retain coupling between path power and AoD."""
    mask = payload["path_mask"].bool()
    weights = normalized_path_power(payload)
    directions = direction_vectors(payload["aod_az_deg"], payload["aod_el_deg"])
    x, y, z = directions.unbind(-1)
    second_moments = torch.stack(
        [
            (weights * x * x).sum(-1),
            (weights * y * y).sum(-1),
            (weights * z * z).sum(-1),
            (weights * x * y).sum(-1),
            (weights * x * z).sum(-1),
            (weights * y * z).sum(-1),
        ],
        dim=-1,
    )
    strongest_index = weights.argmax(dim=-1)
    batch_index = torch.arange(weights.shape[0], device=weights.device)
    strongest_direction = directions[batch_index, strongest_index]
    maximum_fraction = weights.max(dim=-1).values.unsqueeze(-1)
    entropy = -(weights.clamp_min(1e-20).log() * weights * mask).sum(-1)
    active_count = mask.sum(-1).float().clamp_min(1.0)
    normalized_entropy = torch.where(
        active_count > 1,
        entropy / active_count.log().clamp_min(1e-12),
        torch.zeros_like(entropy),
    ).unsqueeze(-1)
    return torch.cat(
        [second_moments, strongest_direction, maximum_fraction, normalized_entropy], dim=-1
    )


def spatial_link_features(
    csi: torch.Tensor,
    tx_array_shape: tuple[int, int],
    offsets: Iterable[tuple[int, int]] = ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0)),
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Low-dimensional complex Tx-correlation summaries for each link."""
    if csi.ndim != 4 or not csi.is_complex():
        raise ValueError("CSI must be complex [batch, rx, tx, subcarrier]")
    rows, columns = tx_array_shape
    if rows * columns != csi.shape[2]:
        raise ValueError("tx_array_shape does not match the CSI transmit dimension")
    grid = csi.permute(0, 1, 3, 2).reshape(
        csi.shape[0], csi.shape[1], csi.shape[3], rows, columns
    )
    energy = grid.abs().square().mean(dim=(1, 2, 3, 4)).clamp_min(1e-20)
    values: list[torch.Tensor] = []
    names: list[str] = []
    for row_offset, column_offset in offsets:
        if row_offset >= rows or column_offset >= columns:
            continue
        first = grid[..., : rows - row_offset, : columns - column_offset]
        second = grid[..., row_offset:, column_offset:]
        correlation = (first.conj() * second).mean(dim=(1, 2, 3, 4)) / energy
        values.extend([correlation.real, correlation.imag])
        names.extend(
            [
                f"tx_corr_drow_{row_offset}_dcol_{column_offset}_real",
                f"tx_corr_drow_{row_offset}_dcol_{column_offset}_imag",
            ]
        )
    return torch.stack(values, dim=-1).float(), tuple(names)


def condition_inputs(
    conditions: torch.Tensor,
    validity: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Five model inputs: standardized power/delay/K, LoS, and K-validity."""
    conditions = conditions.float()
    validity = validity.bool()
    continuous = (conditions[:, :3] - mean[:3]) / std[:3].clamp_min(1e-6)
    continuous[:, 2] = torch.where(
        validity[:, 2], continuous[:, 2], torch.zeros_like(continuous[:, 2])
    )
    return torch.cat(
        [
            continuous,
            (conditions[:, 3:4] >= 0.5).float(),
            validity[:, 2:3].float(),
        ],
        dim=-1,
    )


def collect_learning_arrays(
    dataset: Dataset,
    *,
    target: str,
    tx_array_shape: tuple[int, int],
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    """Materialize compact real-data features used by the diagnostic predictor."""
    if target not in {"aod", "spatial", "all"}:
        raise ValueError("target must be 'aod', 'spatial', or 'all'")
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
        feature_parts: list[torch.Tensor] = []
        names: list[str] = []
        if target in {"aod", "all"}:
            feature_parts.append(aod_statistical_features(batch))
            names.extend(AOD_FEATURE_NAMES)
        if target in {"spatial", "all"}:
            spatial, spatial_names = spatial_link_features(batch["csi"], tx_array_shape)
            feature_parts.append(spatial)
            names.extend(spatial_names)
        current_names = tuple(names)
        if target_names is None:
            target_names = current_names
        elif target_names != current_names:
            raise RuntimeError("target feature schema changed between batches")
        chunks["conditions"].append(batch["conditions"].float())
        chunks["condition_validity"].append(batch["condition_validity"].bool())
        chunks["map_id"].append(batch["map_id"].long())
        chunks["target"].append(torch.cat(feature_parts, dim=-1).float())
    if target_names is None:
        raise ValueError("dataset is empty")
    return {
        key: torch.cat(values, dim=0) for key, values in chunks.items()
    } | {"target_names": target_names}
