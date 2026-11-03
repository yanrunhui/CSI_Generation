from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn

from physics_csi.models.scene_aod_residual import AoDResidualOutput


class SceneConditionedLinkGlobalAoDResidual(nn.Module):
    """One shared AoD rotation per generated link.

    The link representation pools the already-corrected Delay--Power sample and
    its base AoD cluster.  The network therefore changes the cluster's overall
    propagation direction without changing any within-link relative angles.
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
        dropout: float = 0.05,
        maximum_azimuth_residual_deg: float = 20.0,
        maximum_elevation_residual_deg: float = 10.0,
        initial_gate: float = 0.10,
    ) -> None:
        super().__init__()
        self.max_paths = int(max_paths)
        if self.max_paths < 1:
            raise ValueError("Link-global AoD residual requires at least one path")
        self.map_id_to_index = {
            int(map_id): int(index) for map_id, index in map_id_to_index.items()
        }
        indices = sorted(self.map_id_to_index.values())
        if indices != list(range(len(indices))) or not indices:
            raise ValueError("Map indices must be non-empty, unique and contiguous")
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
        # 13 pooled fields: weighted direction mean/variance (6), normalized
        # delay mean/RMS/max (3), max power and entropy (2), path count (1),
        # and direct-path power fraction (1).
        self.pool_encoder = nn.Sequential(
            nn.Linear(13, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.output_network = nn.Sequential(
            nn.Linear(hidden_dim + condition_hidden_dim + map_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        final = self.output_network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
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
            raise ValueError(f"Link-global AoD residual has no learned embedding for {names}")
        return torch.tensor(values, device=external.device, dtype=torch.long)

    def external_map_ids(self, map_index: torch.Tensor) -> torch.Tensor:
        inverse = {index: map_id for map_id, index in self.map_id_to_index.items()}
        return torch.tensor(
            [inverse[int(item)] for item in map_index.detach().cpu().tolist()],
            device=map_index.device,
            dtype=torch.long,
        )

    @staticmethod
    def _direction(azimuth_deg: torch.Tensor, elevation_deg: torch.Tensor) -> torch.Tensor:
        azimuth = torch.deg2rad(azimuth_deg.float())
        elevation = torch.deg2rad(elevation_deg.float())
        horizontal = elevation.cos()
        return torch.stack(
            [horizontal * azimuth.cos(), horizontal * azimuth.sin(), elevation.sin()],
            dim=-1,
        )

    def _pooled_features(
        self,
        payload: Mapping[str, torch.Tensor],
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        mask = payload["path_mask"].bool()
        delay = payload["path_delays_ns"].float()
        first = torch.where(mask, delay, torch.full_like(delay, torch.inf)).amin(-1)
        relative = (delay - first.unsqueeze(-1)).clamp_min(0.0) * mask
        normalized_delay = torch.log1p(
            relative / conditions[:, 1:2].float().clamp_min(1e-6)
        )
        power = payload["path_powers_linear"].float().clamp_min(0.0) * mask
        fraction = power / power.sum(-1, keepdim=True).clamp_min(1e-20)
        direction = self._direction(payload["aod_az_deg"], payload["aod_el_deg"])
        direction_mean = (fraction.unsqueeze(-1) * direction).sum(1)
        direction_variance = (
            fraction.unsqueeze(-1)
            * (direction - direction_mean.unsqueeze(1)).square()
        ).sum(1)
        delay_mean = (fraction * normalized_delay).sum(-1)
        delay_rms = (
            fraction * (normalized_delay - delay_mean.unsqueeze(-1)).square()
        ).sum(-1).clamp_min(1e-20).sqrt()
        delay_max = torch.where(
            mask, normalized_delay, torch.full_like(normalized_delay, -torch.inf)
        ).amax(-1)
        maximum_power = fraction.amax(-1)
        entropy = -(fraction * torch.log(fraction.clamp_min(1e-20))).sum(-1)
        entropy = entropy / math.log(max(self.max_paths, 2))
        path_count = mask.sum(-1).float() / self.max_paths
        direct = payload.get("direct_path_mask", torch.zeros_like(mask)).bool() & mask
        direct_power = (fraction * direct.float()).sum(-1)
        return torch.cat(
            [
                direction_mean,
                direction_variance,
                torch.stack([delay_mean, delay_rms, delay_max], dim=-1),
                torch.stack(
                    [maximum_power, entropy, path_count, direct_power], dim=-1
                ),
            ],
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
            raise ValueError("every Link-global AoD sample must contain an active path")
        if map_index is None:
            if map_id is None:
                raise ValueError("map_id or map_index is required")
            map_index = self.map_indices(map_id)
        map_index = map_index.to(device=conditions.device, dtype=torch.long).reshape(-1)
        if map_index.shape != (batch_size,):
            raise ValueError("map_index must have shape [batch]")
        if bool(((map_index < 0) | (map_index >= len(self.map_id_to_index))).any()):
            raise ValueError("map_index is outside the learned embedding table")

        normalized_condition = (conditions - self.condition_mean) / self.condition_std
        normalized_condition = normalized_condition.clone()
        normalized_condition[:, 2] = torch.where(
            validity[:, 2],
            normalized_condition[:, 2],
            torch.zeros_like(normalized_condition[:, 2]),
        )
        condition_input = torch.cat(
            [normalized_condition, validity[:, 2:3].float()], dim=-1
        )
        context = torch.cat(
            [
                self.pool_encoder(self._pooled_features(dp_payload, conditions)),
                self.condition_encoder(condition_input),
                self.map_embedding(map_index),
            ],
            dim=-1,
        )
        raw = torch.tanh(self.output_network(context))
        azimuth_gate = torch.sigmoid(self.azimuth_gate_logit)
        elevation_gate = torch.sigmoid(self.elevation_gate_logit)
        link_azimuth = self.maximum_azimuth_residual_deg * azimuth_gate * raw[:, 0]
        link_elevation = self.maximum_elevation_residual_deg * elevation_gate * raw[:, 1]
        azimuth_residual = link_azimuth.unsqueeze(-1).expand_as(mask) * mask
        elevation_residual = link_elevation.unsqueeze(-1).expand_as(mask) * mask

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
            key: value for key, value in dp_payload.items() if isinstance(value, torch.Tensor)
        }
        result["aod_az_deg"] = output.aod_az_deg
        result["aod_el_deg"] = output.aod_el_deg
        return result
