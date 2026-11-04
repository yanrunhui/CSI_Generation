from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn

from physics_csi.models.scene_aod_residual import AoDResidualOutput


class SceneConditionedRelativeAoDResidual(nn.Module):
    """Low-dimensional relative-AoD deformation with a fixed global center.

    The input is the output of the frozen Link-global AoD model.  The network
    predicts only two link-level scale factors in the local east/up tangent
    plane.  After the anisotropic deformation, a minimal 3-D rotation restores
    the original power-weighted direction center exactly.  Consequently this
    module can change angular spread but cannot learn another global rotation.
    """

    def __init__(
        self,
        *,
        max_paths: int,
        map_id_to_index: Mapping[str | int, int],
        condition_mean: list[float] | torch.Tensor,
        condition_std: list[float] | torch.Tensor,
        hidden_dim: int = 96,
        map_embedding_dim: int = 24,
        condition_hidden_dim: int = 48,
        dropout: float = 0.05,
        maximum_log_scale: float = 0.50,
        initial_gate: float = 0.10,
        maximum_azimuth_residual_deg: float = 20.0,
        maximum_elevation_residual_deg: float = 10.0,
    ) -> None:
        super().__init__()
        self.max_paths = int(max_paths)
        if self.max_paths < 1:
            raise ValueError("Relative-AoD residual requires at least one path")
        self.map_id_to_index = {
            int(map_id): int(index) for map_id, index in map_id_to_index.items()
        }
        indices = sorted(self.map_id_to_index.values())
        if indices != list(range(len(indices))) or not indices:
            raise ValueError("Map indices must be non-empty, unique and contiguous")
        if maximum_log_scale <= 0:
            raise ValueError("maximum_log_scale must be positive")
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must lie strictly between zero and one")
        self.maximum_log_scale = float(maximum_log_scale)
        # These bounds normalize the existing residual-strength loss.  They do
        # not clip the tangent-plane deformation.
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
        # weighted direction center (3), covariance (9), normalized Delay
        # moments (3), Power moments (3), path count/direct fraction (2).
        self.cluster_encoder = nn.Sequential(
            nn.Linear(20, hidden_dim),
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
        # Keep the familiar names for the shared loss/evaluation interface.
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
            raise ValueError(f"Relative-AoD residual has no learned embedding for {names}")
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

    @staticmethod
    def _angles(direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        direction = direction / direction.square().sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12).sqrt()
        azimuth = torch.rad2deg(torch.atan2(direction[..., 1], direction[..., 0]))
        elevation = torch.rad2deg(
            torch.asin(direction[..., 2].clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        )
        return azimuth, elevation

    @staticmethod
    def _center(direction: torch.Tensor, fraction: torch.Tensor) -> torch.Tensor:
        center = (fraction.unsqueeze(-1) * direction).sum(1)
        return center / center.square().sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12).sqrt()

    @staticmethod
    def _minimal_rotation(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Stable Rodrigues map without a norm of the zero cross product."""
        source = source / source.square().sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12).sqrt()
        target = target / target.square().sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12).sqrt()
        cross = torch.linalg.cross(source, target, dim=-1)
        cosine = (source * target).sum(-1).clamp(-1.0, 1.0)
        batch = source.shape[0]
        skew = torch.zeros(batch, 3, 3, dtype=source.dtype, device=source.device)
        skew[:, 0, 1] = -cross[:, 2]
        skew[:, 0, 2] = cross[:, 1]
        skew[:, 1, 0] = cross[:, 2]
        skew[:, 1, 2] = -cross[:, 0]
        skew[:, 2, 0] = -cross[:, 1]
        skew[:, 2, 1] = cross[:, 0]
        identity = torch.eye(3, dtype=source.dtype, device=source.device).expand(batch, -1, -1)
        # R = I + K + K^2/(1 + dot).  Unlike the equivalent
        # (1-dot)/||cross||^2 form, this is differentiable at the identity.
        factor = (1.0 / (1.0 + cosine).clamp_min(1e-5)).view(-1, 1, 1)
        rotation = identity + skew + skew @ skew * factor
        return rotation

    @staticmethod
    def _tangent_basis(center: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        north = torch.tensor([0.0, 0.0, 1.0], device=center.device, dtype=center.dtype)
        north = north.expand_as(center)
        east = torch.linalg.cross(north, center, dim=-1)
        near_pole = east.square().sum(-1) < 1e-10
        fallback = torch.tensor([1.0, 0.0, 0.0], device=center.device, dtype=center.dtype)
        fallback = fallback.expand_as(center)
        east = torch.where(
            near_pole.unsqueeze(-1),
            torch.linalg.cross(fallback, center, dim=-1),
            east,
        )
        east = east / east.square().sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12).sqrt()
        up = torch.linalg.cross(center, east, dim=-1)
        up = up / up.square().sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12).sqrt()
        return east, up

    def _cluster_features(
        self,
        payload: Mapping[str, torch.Tensor],
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        mask = payload["path_mask"].bool()
        power = payload["path_powers_linear"].float().clamp_min(0.0) * mask
        fraction = power / power.sum(-1, keepdim=True).clamp_min(1e-20)
        direction = self._direction(payload["aod_az_deg"], payload["aod_el_deg"])
        center = self._center(direction, fraction)
        centered = direction - center.unsqueeze(1)
        covariance = torch.einsum("bl,bli,blj->bij", fraction, centered, centered)

        delay = payload["path_delays_ns"].float()
        first = torch.where(mask, delay, torch.full_like(delay, torch.inf)).amin(-1)
        relative = (delay - first.unsqueeze(-1)).clamp_min(0.0) * mask
        normalized = torch.log1p(relative / conditions[:, 1:2].clamp_min(1e-6))
        delay_mean = (fraction * normalized).sum(-1)
        delay_rms = (
            fraction * (normalized - delay_mean.unsqueeze(-1)).square()
        ).sum(-1).clamp_min(0.0).sqrt()
        delay_max = torch.where(
            mask, normalized, torch.full_like(normalized, -torch.inf)
        ).amax(-1)
        maximum_power = fraction.amax(-1)
        entropy = -(fraction * torch.log(fraction.clamp_min(1e-20))).sum(-1)
        entropy = entropy / math.log(max(self.max_paths, 2))
        power_concentration = fraction.square().sum(-1)
        path_count = mask.sum(-1).float() / self.max_paths
        direct = payload.get("direct_path_mask", torch.zeros_like(mask)).bool() & mask
        direct_fraction = (fraction * direct).sum(-1)
        return torch.cat(
            [
                center,
                covariance.flatten(1),
                torch.stack([delay_mean, delay_rms, delay_max], dim=-1),
                torch.stack(
                    [
                        maximum_power,
                        entropy,
                        power_concentration,
                        path_count,
                        direct_fraction,
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )

    def forward(
        self,
        link_global_payload: Mapping[str, torch.Tensor],
        target_conditions: torch.Tensor,
        target_condition_validity: torch.Tensor,
        *,
        map_id: torch.Tensor | int | None = None,
        map_index: torch.Tensor | None = None,
    ) -> AoDResidualOutput:
        conditions = target_conditions.float()
        validity = target_condition_validity.bool()
        batch = conditions.shape[0]
        mask = link_global_payload["path_mask"].bool()
        if mask.shape != (batch, self.max_paths):
            raise ValueError("path mask must have shape [batch, max_paths]")
        if not bool(mask.any(-1).all()):
            raise ValueError("every Relative-AoD sample must contain an active path")
        if map_index is None:
            if map_id is None:
                raise ValueError("map_id or map_index is required")
            map_index = self.map_indices(map_id)
        map_index = map_index.to(device=conditions.device, dtype=torch.long).reshape(-1)
        if map_index.shape != (batch,):
            raise ValueError("map_index must have shape [batch]")

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
                self.cluster_encoder(
                    self._cluster_features(link_global_payload, conditions)
                ),
                self.condition_encoder(condition_input),
                self.map_embedding(map_index),
            ],
            dim=-1,
        )
        raw_scale = torch.tanh(self.output_network(context))
        azimuth_gate = torch.sigmoid(self.azimuth_gate_logit)
        elevation_gate = torch.sigmoid(self.elevation_gate_logit)
        scale = torch.exp(
            self.maximum_log_scale
            * raw_scale
            * torch.stack([azimuth_gate, elevation_gate])
        )

        power = link_global_payload["path_powers_linear"].float().clamp_min(0.0) * mask
        fraction = power / power.sum(-1, keepdim=True).clamp_min(1e-20)
        direction = self._direction(
            link_global_payload["aod_az_deg"], link_global_payload["aod_el_deg"]
        )
        center = self._center(direction, fraction)
        east, up = self._tangent_basis(center)
        cosine = (direction * center.unsqueeze(1)).sum(-1).clamp(
            -1.0 + 1e-6, 1.0 - 1e-6
        )
        angle = torch.acos(cosine)
        tangent = direction - cosine.unsqueeze(-1) * center.unsqueeze(1)
        tangent = tangent * (
            angle / angle.sin().abs().clamp_min(1e-6)
        ).unsqueeze(-1)
        east_coordinate = (tangent * east.unsqueeze(1)).sum(-1) * scale[:, 0:1]
        up_coordinate = (tangent * up.unsqueeze(1)).sum(-1) * scale[:, 1:2]
        scaled_tangent = (
            east_coordinate.unsqueeze(-1) * east.unsqueeze(1)
            + up_coordinate.unsqueeze(-1) * up.unsqueeze(1)
        )
        scaled_angle_squared = scaled_tangent.square().sum(-1)
        safe_angle = scaled_angle_squared.clamp_min(1e-12).sqrt().clamp_max(
            torch.pi - 1e-4
        )
        small = scaled_angle_squared < 1e-6
        cosine_scale = torch.where(
            small,
            1.0 - scaled_angle_squared / 2.0
            + scaled_angle_squared.square() / 24.0,
            safe_angle.cos(),
        )
        sinc_scale = torch.where(
            small,
            1.0 - scaled_angle_squared / 6.0
            + scaled_angle_squared.square() / 120.0,
            safe_angle.sin() / safe_angle,
        )
        deformed = (
            cosine_scale.unsqueeze(-1) * center.unsqueeze(1)
            + sinc_scale.unsqueeze(-1) * scaled_tangent
        )
        deformed = torch.where(mask.unsqueeze(-1), deformed, direction)

        # Numerical and weighting asymmetry can move the centroid.  Restore it
        # with one shared SO(3) rotation, which cannot alter relative geometry.
        deformed_center = self._center(deformed, fraction)
        rotation = self._minimal_rotation(deformed_center, center)
        corrected = torch.einsum("bij,blj->bli", rotation, deformed)
        corrected = corrected / corrected.square().sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12).sqrt()
        corrected_azimuth, corrected_elevation = self._angles(corrected)
        original_azimuth = link_global_payload["aod_az_deg"].float()
        original_elevation = link_global_payload["aod_el_deg"].float()
        azimuth_residual = torch.rad2deg(
            torch.atan2(
                torch.deg2rad(corrected_azimuth - original_azimuth).sin(),
                torch.deg2rad(corrected_azimuth - original_azimuth).cos(),
            )
        ) * mask
        elevation_residual = (corrected_elevation - original_elevation) * mask
        return AoDResidualOutput(
            aod_az_deg=corrected_azimuth * mask,
            aod_el_deg=corrected_elevation * mask,
            azimuth_residual_deg=azimuth_residual,
            elevation_residual_deg=elevation_residual,
            azimuth_gate=azimuth_gate,
            elevation_gate=elevation_gate,
        )

    @staticmethod
    def apply_to_payload(
        link_global_payload: Mapping[str, torch.Tensor], output: AoDResidualOutput
    ) -> dict[str, torch.Tensor]:
        result = {
            key: value
            for key, value in link_global_payload.items()
            if isinstance(value, torch.Tensor)
        }
        result["aod_az_deg"] = output.aod_az_deg
        result["aod_el_deg"] = output.aod_el_deg
        return result

