from __future__ import annotations

import torch
from torch import nn

from physics_csi.models.diffusion.condition_encoder import SinusoidalTimeEmbedding


class SceneResidualDenoiser(nn.Module):
    """A small zero-output-initialized correction on top of a frozen B3 denoiser.

    The adapter sees the current noisy path tokens, the B3 noise prediction,
    physical conditions and a categorical Map embedding.  Its residual scale is
    explicit so a newly initialized adapter starts exactly at the B3 function.
    """

    def __init__(
        self,
        feature_dim: int,
        condition_dim: int,
        hidden_dim: int,
        max_paths: int,
        num_scenes: int,
        scene_embedding_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        diffusion_steps: int,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if num_scenes <= 0:
            raise ValueError("num_scenes must be positive")
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        self.num_scenes = int(num_scenes)
        self.diffusion_steps = int(diffusion_steps)
        self.residual_scale = float(residual_scale)
        self.scene_embedding = nn.Embedding(num_scenes, scene_embedding_dim)
        self.input_projection = nn.Linear(feature_dim * 2, hidden_dim)
        self.position_embedding = nn.Parameter(torch.randn(max_paths, hidden_dim) * 0.02)
        self.path_count_embedding = nn.Embedding(max_paths + 1, hidden_dim)
        self.time = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context = nn.Sequential(
            nn.Linear(condition_dim + scene_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
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
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, feature_dim)
        # Exact functional preservation at initialization: S1 starts as B3.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        noisy: torch.Tensor,
        base_prediction: torch.Tensor,
        timesteps: torch.Tensor,
        physical_condition: torch.Tensor,
        scene_index: torch.Tensor,
        path_count: torch.Tensor,
        path_mask: torch.Tensor,
        *,
        drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``Delta epsilon`` with shape ``[batch, paths, features]``."""
        scene_index = scene_index.to(device=noisy.device, dtype=torch.long).reshape(-1)
        if scene_index.shape != (noisy.shape[0],):
            raise ValueError("scene_index must contain one value per batch item")
        if bool(((scene_index < 0) | (scene_index >= self.num_scenes)).any()):
            raise ValueError("scene_index is outside the learned adapter table")
        scene = self.scene_embedding(scene_index)
        hidden = self.input_projection(torch.cat([noisy, base_prediction], dim=-1))
        hidden = hidden + self.position_embedding.unsqueeze(0)
        hidden = hidden + self.path_count_embedding(path_count).unsqueeze(1)
        hidden = hidden + self.time(timesteps).unsqueeze(1)
        hidden = hidden + self.context(
            torch.cat([physical_condition, scene], dim=-1)
        ).unsqueeze(1)
        hidden = self.transformer(hidden, src_key_padding_mask=~path_mask.bool())
        residual = self.output(self.output_norm(hidden)) * self.residual_scale
        residual = residual * path_mask.unsqueeze(-1).to(residual.dtype)
        if drop_mask is not None:
            keep = ~drop_mask.to(device=noisy.device, dtype=torch.bool).reshape(-1, 1, 1)
            if keep.shape[0] != noisy.shape[0]:
                raise ValueError("drop_mask must contain one value per batch item")
            residual = residual * keep
        return residual
