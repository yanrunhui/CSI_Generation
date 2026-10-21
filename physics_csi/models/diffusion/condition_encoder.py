from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ConditionEncoder(nn.Module):
    """Encode canonical condition values and their validity mask.

    Invalid values are zeroed after normalization, while the explicit validity
    bits let the network distinguish a missing value from a genuine zero.
    """

    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if len(mean) != 4 or len(std) != 4:
            raise ValueError("condition mean/std must each contain four values")
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).clamp_min(1e-6)
        )
        self.network = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.null_embedding = nn.Parameter(torch.zeros(embedding_dim))

    def forward(
        self,
        values: torch.Tensor,
        validity: torch.Tensor | None = None,
        *,
        drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = values.float()
        if values.ndim != 2 or values.shape[-1] != 4:
            raise ValueError("conditions must have shape [batch, 4]")
        valid = (
            torch.ones_like(values, dtype=torch.bool)
            if validity is None
            else validity.to(device=values.device, dtype=torch.bool)
        )
        if valid.shape != values.shape:
            raise ValueError("condition validity must match condition values")
        normalized = (values - self.mean) / self.std
        normalized = torch.where(valid, normalized, torch.zeros_like(normalized))
        embedding = self.network(torch.cat([normalized, valid.float()], dim=-1))
        if drop_mask is not None:
            drop = drop_mask.to(device=values.device, dtype=torch.bool).reshape(-1, 1)
            if drop.shape[0] != values.shape[0]:
                raise ValueError("drop_mask must contain one value per batch item")
            null = self.null_embedding.unsqueeze(0).expand_as(embedding)
            embedding = torch.where(drop, null, embedding)
        return embedding

    def unconditional(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.null_embedding.to(device).unsqueeze(0).expand(batch_size, -1)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 4:
            raise ValueError("time embedding dimension must be at least four")
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        exponent = -torch.log(torch.tensor(10000.0, device=timesteps.device))
        frequencies = torch.exp(exponent * torch.arange(half, device=timesteps.device) / max(half - 1, 1))
        angles = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.shape[-1] < self.dimension:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding

