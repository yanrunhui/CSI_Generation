from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn


@dataclass
class AoDResidualOutput:
    aod_az_deg: torch.Tensor
    aod_el_deg: torch.Tensor
    azimuth_residual_deg: torch.Tensor
    elevation_residual_deg: torch.Tensor
    azimuth_gate: torch.Tensor
    elevation_gate: torch.Tensor


class SceneConditionedAoDResidual(nn.Module):
    """Per-path periodic AoD transport conditioned on projected Delay--Power.

    This module never changes path topology, Delay, Power, Phase or AoA.  Its
    zero-initialized heads preserve the DP-only sample exactly at initialization.
    A learnable global gate starts small and bounded residual angles are wrapped
    with atan2, avoiding an azimuth discontinuity at -180/180 degrees.
    """

    def __init__(
        self,
        *,
        max_paths: int,
        map_id_to_index: Mapping[str | int, int],
        condition_mean: list[float] | torch.Tensor,
        condition_std: list[float] | torch.Tensor,
        hidden_dim: int = 128,
        map_embedding_dim: int = 32,
        condition_hidden_dim: int = 64,
        layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.05,
        maximum_azimuth_residual_deg: float = 20.0,
        maximum_elevation_residual_deg: float = 10.0,
        initial_gate: float = 0.10,
    ) -> None:
        super().__init__()
        self.max_paths = int(max_paths)
        if self.max_paths < 1:
            raise ValueError("AoD residual requires at least one path")
        self.map_id_to_index = {
            int(map_id): int(index) for map_id, index in map_id_to_index.items()
        }
        indices = sorted(self.map_id_to_index.values())
        if indices != list(range(len(indices))) or not indices:
            raise ValueError("Map indices must be non-empty, unique and contiguous")
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if maximum_azimuth_residual_deg <= 0 or maximum_elevation_residual_deg <= 0:
            raise ValueError("maximum angle residuals must be positive")
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must lie strictly between zero and one")
        self.maximum_azimuth_residual_deg = float(maximum_azimuth_residual_deg)
        self.maximum_elevation_residual_deg = float(maximum_elevation_residual_deg)

        mean = torch.as_tensor(condition_mean, dtype=torch.float32)
        std = torch.as_tensor(condition_std, dtype=torch.float32)
        if mean.shape != (4,) or std.shape != (4,):
            raise ValueError("condition statistics must contain four canonical fields")
        self.register_buffer("condition_mean", mean)
        self.register_buffer("condition_std", std.clamp_min(1e-6))

        self.map_embedding = nn.Embedding(len(indices), map_embedding_dim)
        self.condition_encoder = nn.Sequential(
            nn.Linear(5, condition_hidden_dim),
            nn.SiLU(),
            nn.Linear(condition_hidden_dim, condition_hidden_dim),
            nn.SiLU(),
        )
        context_dim = condition_hidden_dim + map_embedding_dim
        # token = normalized relative delay, log power fraction, AoD xyz,
        # rank, normalized path count, direct flag, active flag.
        self.token_input = nn.Linear(9 + context_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.path_encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.azimuth_head = nn.Linear(hidden_dim, 1)
        self.elevation_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.azimuth_head.weight)
        nn.init.zeros_(self.azimuth_head.bias)
        nn.init.zeros_(self.elevation_head.weight)
        nn.init.zeros_(self.elevation_head.bias)
        gate_logit = math.log(initial_gate / (1.0 - initial_gate))
        self.azimuth_gate_logit = nn.Parameter(torch.tensor(gate_logit))
        self.elevation_gate_logit = nn.Parameter(torch.tensor(gate_logit))

    def map_indices(self, map_id: torch.Tensor | int) -> torch.Tensor:
        external = torch.as_tensor(map_id, dtype=torch.long).reshape(-1)
        values: list[int] = []
        unknown: set[int] = set()
        for item in external.detach().cpu().tolist():
            key = int(item)
            if key not in self.map_id_to_index:
                unknown.add(key)
            else:
                values.append(self.map_id_to_index[key])
        if unknown:
            names = ", ".join(f"map_{value}" for value in sorted(unknown))
            raise ValueError(f"AoD residual has no learned embedding for {names}")
        return torch.tensor(values, device=external.device, dtype=torch.long)

    def external_map_ids(self, map_index: torch.Tensor) -> torch.Tensor:
        inverse = {index: map_id for map_id, index in self.map_id_to_index.items()}
        values = [inverse[int(item)] for item in map_index.detach().cpu().tolist()]
        return torch.tensor(values, device=map_index.device, dtype=torch.long)

    @staticmethod
    def _direction(azimuth_deg: torch.Tensor, elevation_deg: torch.Tensor) -> torch.Tensor:
        azimuth = torch.deg2rad(azimuth_deg.float())
        elevation = torch.deg2rad(elevation_deg.float())
        horizontal = elevation.cos()
        return torch.stack(
            [horizontal * azimuth.cos(), horizontal * azimuth.sin(), elevation.sin()],
            dim=-1,
        )

    def forward(
        self,
        dp_payload: Mapping[str, torch.Tensor],
        target_conditions: torch.Tensor,
        target_condition_validity: torch.Tensor,
        *,
        map_id: torch.Tensor | int | None = None,
        map_index: torch.Tensor | None = None,
    ) -> AoDResidualOutput:
        conditions = target_conditions.float()
        validity = target_condition_validity.bool()
        batch_size = conditions.shape[0]
        mask = dp_payload["path_mask"].bool()
        if mask.shape != (batch_size, self.max_paths):
            raise ValueError("path mask must have shape [batch, max_paths]")
        if not bool(mask.any(-1).all()):
            raise ValueError("every AoD residual sample must contain an active path")
        if map_index is None:
            if map_id is None:
                raise ValueError("map_id or map_index is required")
            map_index = self.map_indices(map_id)
        map_index = map_index.to(device=conditions.device, dtype=torch.long).reshape(-1)
        if map_index.shape != (batch_size,):
            raise ValueError("map_index must have shape [batch]")
        if bool(((map_index < 0) | (map_index >= len(self.map_id_to_index))).any()):
            raise ValueError("map_index is outside the learned embedding table")

        delays = dp_payload["path_delays_ns"].float()
        first = torch.where(mask, delays, torch.full_like(delays, torch.inf)).amin(-1)
        relative = (delays - first.unsqueeze(-1)).clamp_min(0.0) * mask
        spread = conditions[:, 1:2].clamp_min(1e-6)
        normalized_delay = torch.log1p(relative / spread)
        power = dp_payload["path_powers_linear"].float().clamp_min(0.0) * mask
        fraction = power / power.sum(-1, keepdim=True).clamp_min(1e-20)
        direction = self._direction(dp_payload["aod_az_deg"], dp_payload["aod_el_deg"])
        direct = dp_payload.get("direct_path_mask", torch.zeros_like(mask)).bool() & mask

        normalized_condition = (conditions - self.condition_mean) / self.condition_std
        normalized_condition = normalized_condition.clone()
        normalized_condition[:, 2] = torch.where(
            validity[:, 2], normalized_condition[:, 2], torch.zeros_like(normalized_condition[:, 2])
        )
        condition_input = torch.cat(
            [normalized_condition, validity[:, 2:3].float()], dim=-1
        )
        context = torch.cat(
            [self.condition_encoder(condition_input), self.map_embedding(map_index)], dim=-1
        )
        rank = torch.arange(
            self.max_paths, device=mask.device, dtype=delays.dtype
        ).unsqueeze(0).expand(batch_size, -1)
        rank = rank / max(self.max_paths - 1, 1)
        path_count = mask.sum(-1, keepdim=True).to(delays.dtype) / self.max_paths
        token = torch.cat(
            [
                normalized_delay.unsqueeze(-1),
                torch.log(fraction.clamp_min(1e-20)).unsqueeze(-1),
                direction,
                rank.unsqueeze(-1),
                path_count.unsqueeze(1).expand(-1, self.max_paths, -1),
                direct.float().unsqueeze(-1),
                mask.float().unsqueeze(-1),
            ],
            dim=-1,
        )
        encoded = self.token_input(
            torch.cat([token, context.unsqueeze(1).expand(-1, self.max_paths, -1)], dim=-1)
        )
        encoded = self.path_encoder(encoded, src_key_padding_mask=~mask)
        encoded = self.output_norm(encoded)

        azimuth_gate = torch.sigmoid(self.azimuth_gate_logit)
        elevation_gate = torch.sigmoid(self.elevation_gate_logit)
        azimuth_residual = (
            self.maximum_azimuth_residual_deg
            * azimuth_gate
            * torch.tanh(self.azimuth_head(encoded).squeeze(-1))
            * mask
        )
        elevation_residual = (
            self.maximum_elevation_residual_deg
            * elevation_gate
            * torch.tanh(self.elevation_head(encoded).squeeze(-1))
            * mask
        )
        original_azimuth = torch.deg2rad(dp_payload["aod_az_deg"].float())
        corrected_azimuth = original_azimuth + torch.deg2rad(azimuth_residual)
        corrected_azimuth = torch.atan2(corrected_azimuth.sin(), corrected_azimuth.cos())
        corrected_elevation = (
            dp_payload["aod_el_deg"].float() + elevation_residual
        ).clamp(-90.0, 90.0)
        return AoDResidualOutput(
            aod_az_deg=torch.rad2deg(corrected_azimuth) * mask,
            aod_el_deg=corrected_elevation * mask,
            azimuth_residual_deg=azimuth_residual,
            elevation_residual_deg=elevation_residual,
            azimuth_gate=azimuth_gate,
            elevation_gate=elevation_gate,
        )

    @staticmethod
    def apply_to_payload(
        dp_payload: Mapping[str, torch.Tensor], output: AoDResidualOutput
    ) -> dict[str, torch.Tensor]:
        result = {
            key: value
            for key, value in dp_payload.items()
            if isinstance(value, torch.Tensor)
        }
        result["aod_az_deg"] = output.aod_az_deg
        result["aod_el_deg"] = output.aod_el_deg
        return result
