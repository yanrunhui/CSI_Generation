from __future__ import annotations

import torch
from torch import nn


def make_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    layers: int,
    dropout: float,
) -> nn.Sequential:
    if layers < 1:
        raise ValueError("layers must be at least one")
    modules: list[nn.Module] = []
    current = input_dim
    for _ in range(layers - 1):
        modules.extend(
            [
                nn.Linear(current, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
        )
        current = hidden_dim
    modules.append(nn.Linear(current, output_dim))
    return nn.Sequential(*modules)


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        mean: list[float],
        std: list[float],
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if len(mean) != 4 or len(std) != 4:
            raise ValueError("Canonical condition normalization requires four values")
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.network = make_mlp(8, hidden_dim, embedding_dim, layers=2, dropout=0.0)

    def forward(
        self, conditions: torch.Tensor, validity: torch.Tensor | None = None
    ) -> torch.Tensor:
        if conditions.shape[-1] != 4:
            raise ValueError("conditions must have shape [batch, 4]")
        if validity is None:
            validity = torch.ones_like(conditions, dtype=torch.bool)
        normalized = (conditions.float() - self.mean) / self.std.clamp_min(1e-6)
        normalized = normalized * validity.to(normalized.dtype)
        return self.network(torch.cat([normalized, validity.float()], dim=-1))


class PathEncoder(nn.Module):
    def __init__(
        self,
        path_feature_dim: int,
        path_embedding_dim: int,
        condition_embedding_dim: int,
        hidden_dim: int,
        latent_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.path_network = make_mlp(
            path_feature_dim,
            hidden_dim,
            path_embedding_dim,
            layers=layers,
            dropout=dropout,
        )
        posterior_input = 2 * path_embedding_dim + condition_embedding_dim
        self.posterior = make_mlp(
            posterior_input,
            hidden_dim,
            2 * latent_dim,
            layers=2,
            dropout=dropout,
        )

    def forward(
        self,
        path_features: torch.Tensor,
        path_mask: torch.Tensor,
        condition_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.path_network(path_features)
        mask = path_mask.bool().unsqueeze(-1)
        count = mask.sum(dim=1).clamp_min(1)
        mean_pool = (embedded * mask).sum(dim=1) / count
        max_pool = embedded.masked_fill(~mask, -torch.inf).amax(dim=1)
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        posterior = self.posterior(torch.cat([mean_pool, max_pool, condition_embedding], dim=-1))
        return posterior.chunk(2, dim=-1)


class MultipathDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        condition_embedding_dim: int,
        hidden_dim: int,
        max_paths: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_paths = max_paths
        self.trunk = make_mlp(
            latent_dim + condition_embedding_dim,
            hidden_dim,
            hidden_dim,
            layers=layers,
            dropout=dropout,
        )
        self.path_count_head = nn.Linear(hidden_dim, max_paths)
        self.validity_head = nn.Linear(hidden_dim, max_paths)
        self.delay_delta_head = nn.Linear(hidden_dim, max_paths)
        self.power_head = nn.Linear(hidden_dim, max_paths)
        self.phase_head = nn.Linear(hidden_dim, max_paths * 2)
        self.azimuth_head = nn.Linear(hidden_dim, max_paths * 4)
        self.elevation_head = nn.Linear(hidden_dim, max_paths * 2)
        self.direct_path_head = nn.Linear(hidden_dim, max_paths)

    def forward(self, latent: torch.Tensor, condition_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(torch.cat([latent, condition_embedding], dim=-1))
        batch = hidden.shape[0]
        phase_xy = self.phase_head(hidden).view(batch, self.max_paths, 2)
        azimuth_xy = self.azimuth_head(hidden).view(batch, self.max_paths, 4)
        elevations = self.elevation_head(hidden).view(batch, self.max_paths, 2)
        delay_deltas = torch.nn.functional.softplus(self.delay_delta_head(hidden))
        return {
            "path_count_logits": self.path_count_head(hidden),
            "validity_logits": self.validity_head(hidden),
            "direct_path_logits": self.direct_path_head(hidden),
            "delays_ns": torch.cumsum(delay_deltas, dim=-1),
            "positive_power_weights": torch.nn.functional.softplus(self.power_head(hidden)),
            "phases_rad": torch.atan2(phase_xy[..., 0], phase_xy[..., 1]),
            "aoa_az_deg": torch.rad2deg(torch.atan2(azimuth_xy[..., 0], azimuth_xy[..., 1])),
            "aod_az_deg": torch.rad2deg(torch.atan2(azimuth_xy[..., 2], azimuth_xy[..., 3])),
            "aoa_el_deg": 90.0 * torch.tanh(elevations[..., 0]),
            "aod_el_deg": 90.0 * torch.tanh(elevations[..., 1]),
        }


def mask_from_path_count(path_count: torch.Tensor, max_paths: int) -> torch.Tensor:
    indices = torch.arange(max_paths, device=path_count.device)
    return indices.unsqueeze(0) < path_count.long().unsqueeze(-1)

