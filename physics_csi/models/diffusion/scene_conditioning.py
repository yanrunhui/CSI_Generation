from __future__ import annotations

import torch
from torch import nn


class SceneConditionFusion(nn.Module):
    """Fuse a categorical scene identity with a physical-condition embedding."""

    def __init__(
        self,
        condition_dim: int,
        num_scenes: int,
        scene_embedding_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_scenes <= 0:
            raise ValueError("num_scenes must be positive")
        if scene_embedding_dim <= 0:
            raise ValueError("scene_embedding_dim must be positive")
        self.num_scenes = int(num_scenes)
        self.scene_embedding = nn.Embedding(num_scenes, scene_embedding_dim)
        self.fusion = nn.Sequential(
            nn.Linear(condition_dim + scene_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, condition_dim),
        )
        # A small residual scene branch preserves the scale/meaning of a B3
        # condition embedding while still providing nonzero gradients to the
        # new Map Embedding from the first optimizer step.
        final = self.fusion[-1]
        assert isinstance(final, nn.Linear)
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias)
        self.null_embedding = nn.Parameter(torch.zeros(condition_dim))

    def forward(
        self,
        physical_embedding: torch.Tensor,
        scene_index: torch.Tensor,
        *,
        drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if physical_embedding.ndim != 2:
            raise ValueError("physical embedding must have shape [batch, dimension]")
        scene_index = scene_index.to(
            device=physical_embedding.device, dtype=torch.long
        ).reshape(-1)
        if scene_index.shape[0] != physical_embedding.shape[0]:
            raise ValueError("scene_index must contain one value per batch item")
        if bool(((scene_index < 0) | (scene_index >= self.num_scenes)).any()):
            raise ValueError("scene_index is outside the learned embedding table")
        scene_embedding = self.scene_embedding(scene_index)
        fused = physical_embedding + self.fusion(
            torch.cat([physical_embedding, scene_embedding], dim=-1)
        )
        if drop_mask is not None:
            drop = drop_mask.to(
                device=physical_embedding.device, dtype=torch.bool
            ).reshape(-1, 1)
            if drop.shape[0] != physical_embedding.shape[0]:
                raise ValueError("drop_mask must contain one value per batch item")
            null = self.null_embedding.unsqueeze(0).expand_as(fused)
            fused = torch.where(drop, null, fused)
        return fused

    def unconditional(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.null_embedding.to(device).unsqueeze(0).expand(batch_size, -1)
