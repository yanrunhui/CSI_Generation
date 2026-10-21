from __future__ import annotations

import torch
from torch import nn

from physics_csi.models.diffusion.condition_encoder import SinusoidalTimeEmbedding


class MultipathDenoiser(nn.Module):
    """Small masked Transformer for the joint distribution of path slots."""

    def __init__(
        self,
        feature_dim: int,
        condition_dim: int,
        hidden_dim: int,
        max_paths: int,
        layers: int,
        heads: int,
        dropout: float,
        diffusion_steps: int,
    ) -> None:
        super().__init__()
        self.diffusion_steps = int(diffusion_steps)
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.randn(max_paths, hidden_dim) * 0.02)
        self.path_count_embedding = nn.Embedding(max_paths + 1, hidden_dim)
        self.time = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.condition = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, feature_dim))

    def forward(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        condition_embedding: torch.Tensor,
        path_count: torch.Tensor,
        path_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input_projection(noisy)
        hidden = hidden + self.position_embedding.unsqueeze(0)
        hidden = hidden + self.path_count_embedding(path_count).unsqueeze(1)
        hidden = hidden + self.time(timesteps).unsqueeze(1)
        hidden = hidden + self.condition(condition_embedding).unsqueeze(1)
        hidden = self.transformer(hidden, src_key_padding_mask=~path_mask.bool())
        return self.output(hidden)

