from __future__ import annotations

import torch
from torch import nn

from physics_csi.models.diffusion.condition_encoder import SinusoidalTimeEmbedding


class _ConditionedResidualBlock(nn.Module):
    def __init__(self, channels: int, embedding_dim: int, dropout: float, dilation: int) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.embedding = nn.Linear(embedding_dim, channels * 2)
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, value: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(torch.nn.functional.silu(self.norm1(value)))
        scale, shift = self.embedding(embedding).chunk(2, dim=-1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1.0 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        hidden = self.conv2(self.dropout(torch.nn.functional.silu(hidden)))
        return value + hidden


class AngleDelayDenoiser(nn.Module):
    """Shape-preserving 3-D residual denoiser for complex angle-delay tensors."""

    def __init__(
        self,
        condition_dim: int,
        channels: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        embedding_dim = channels * 4
        self.time = nn.Sequential(
            SinusoidalTimeEmbedding(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.condition = nn.Sequential(
            nn.Linear(condition_dim, embedding_dim), nn.SiLU(), nn.Linear(embedding_dim, embedding_dim)
        )
        self.input = nn.Conv3d(2, channels, 3, padding=1)
        self.blocks = nn.ModuleList(
            [
                _ConditionedResidualBlock(
                    channels, embedding_dim, dropout, dilation=1 if index % 2 == 0 else 2
                )
                for index in range(layers)
            ]
        )
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.output = nn.Sequential(
            nn.GroupNorm(groups, channels), nn.SiLU(), nn.Conv3d(channels, 2, 3, padding=1)
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self, noisy: torch.Tensor, timesteps: torch.Tensor, condition_embedding: torch.Tensor
    ) -> torch.Tensor:
        embedding = self.time(timesteps) + self.condition(condition_embedding)
        hidden = self.input(noisy)
        for block in self.blocks:
            hidden = block(hidden, embedding)
        return self.output(hidden)

