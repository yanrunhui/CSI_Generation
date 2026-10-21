from __future__ import annotations

import torch
from torch import nn


class PathCountPredictor(nn.Module):
    """Categorical p(path_count | physical condition)."""

    def __init__(self, condition_dim: int, hidden_dim: int, max_paths: int, dropout: float) -> None:
        super().__init__()
        self.max_paths = int(max_paths)
        self.network = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.max_paths),
        )

    def forward(self, condition_embedding: torch.Tensor) -> torch.Tensor:
        return self.network(condition_embedding)

    def probabilities(
        self,
        condition_embedding: torch.Tensor,
        *,
        temperature: float = 1.0,
        logit_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return calibrated categorical probabilities over path counts 1..Lmax."""
        if temperature <= 0:
            raise ValueError("path-count temperature must be positive")
        logits = self(condition_embedding)
        if logit_bias is not None:
            if logit_bias.shape != (self.max_paths,):
                raise ValueError("path-count logit bias must have shape [max_paths]")
            logits = logits + logit_bias.to(device=logits.device, dtype=logits.dtype)
        return torch.softmax(logits / temperature, dim=-1)

    @torch.no_grad()
    def sample(
        self,
        condition_embedding: torch.Tensor,
        *,
        temperature: float = 1.0,
        logit_bias: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        probabilities = self.probabilities(
            condition_embedding, temperature=temperature, logit_bias=logit_bias
        )
        return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1) + 1

