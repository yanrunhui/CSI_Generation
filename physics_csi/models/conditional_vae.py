from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from physics_csi.models.blocks import (
    ConditionEncoder,
    MultipathDecoder,
    PathEncoder,
    mask_from_path_count,
)
from physics_csi.physics.constraints import PhysicsConstraintLayer, check_condition_feasibility
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig


class PhysicsConditionedVAE(nn.Module):
    condition_fields = (
        "total_power_db",
        "delay_spread_ns",
        "k_factor_db",
        "los_status",
    )

    def __init__(self, model_config: Mapping[str, Any], system_config: SystemConfig) -> None:
        super().__init__()
        self.max_paths = int(model_config["max_paths"])
        self.latent_dim = int(model_config["latent_dim"])
        hidden_dim = int(model_config["hidden_dim"])
        condition_dim = int(model_config["condition_embedding_dim"])
        path_embedding_dim = int(model_config["path_embedding_dim"])
        dropout = float(model_config["dropout"])
        normalization = model_config["condition_normalization"]
        self.path_scales = dict(model_config["path_scales"])
        self.condition_encoder = ConditionEncoder(
            normalization["mean"], normalization["std"], condition_dim, hidden_dim
        )
        self.encoder = PathEncoder(
            path_feature_dim=12,
            path_embedding_dim=path_embedding_dim,
            condition_embedding_dim=condition_dim,
            hidden_dim=hidden_dim,
            latent_dim=self.latent_dim,
            layers=int(model_config["encoder_layers"]),
            dropout=dropout,
        )
        self.decoder = MultipathDecoder(
            latent_dim=self.latent_dim,
            condition_embedding_dim=condition_dim,
            hidden_dim=hidden_dim,
            max_paths=self.max_paths,
            layers=int(model_config["decoder_layers"]),
            dropout=dropout,
        )
        constraint_config = dict(model_config["constraints"])
        self.constraints = PhysicsConstraintLayer(**constraint_config)
        self.synthesizer = MIMOOFDMSynthesizer(system_config)

    def encode(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        condition_embedding = self.condition_encoder(
            batch["conditions"], batch.get("condition_validity")
        )
        features = self._path_features(batch)
        mean, log_variance = self.encoder(
            features, batch["path_mask"], condition_embedding
        )
        return mean, log_variance, condition_embedding

    @staticmethod
    def reparameterize(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        standard_deviation = torch.exp(0.5 * log_variance)
        return mean + torch.randn_like(standard_deviation) * standard_deviation

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mean, log_variance, condition_embedding = self.encode(batch)
        latent = self.reparameterize(mean, log_variance)
        decoded = self.decoder(latent, condition_embedding)
        path_mask = batch["path_mask"].bool()
        constrained = self.constraints(
            decoded["delays_ns"],
            decoded["positive_power_weights"],
            path_mask,
            batch["conditions"],
            target_condition_validity=batch.get("condition_validity"),
            direct_path_mask=batch.get("direct_path_mask"),
        )
        csi = self._synthesize(decoded, constrained, path_mask)
        return {
            **decoded,
            "path_mask": path_mask,
            "path_delays_ns": constrained.delays_ns,
            "path_powers_linear": constrained.powers_linear,
            "direct_path_mask": constrained.direct_path_mask,
            "actual_conditions": constrained.actual_conditions,
            "actual_condition_validity": constrained.condition_validity,
            "constraint_feasible": constrained.feasible,
            "csi": csi,
            "latent_mean": mean,
            "latent_log_variance": log_variance,
            "latent": latent,
        }

    @torch.no_grad()
    def generate(
        self,
        conditions: Mapping[str, float] | torch.Tensor,
        num_samples: int,
        *,
        condition_validity: torch.Tensor | None = None,
        sample_path_count: bool = True,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        device = next(self.parameters()).device
        if isinstance(conditions, Mapping):
            los = float(conditions["los_status"])
            check_condition_feasibility(conditions, k_factor_valid=los >= 0.5)
            row = torch.tensor(
                [float(conditions[field]) for field in self.condition_fields],
                dtype=torch.float32,
                device=device,
            )
            condition_tensor = row.unsqueeze(0).expand(num_samples, -1)
            if condition_validity is None:
                condition_validity = torch.ones_like(condition_tensor, dtype=torch.bool)
                condition_validity[:, 2] = los >= 0.5
        else:
            condition_tensor = conditions.to(device=device, dtype=torch.float32)
            if condition_tensor.ndim == 1:
                condition_tensor = condition_tensor.unsqueeze(0)
            if condition_tensor.shape[0] == 1:
                condition_tensor = condition_tensor.expand(num_samples, -1)
            if condition_tensor.shape != (num_samples, 4):
                raise ValueError(
                    f"Expected conditions shape {(num_samples, 4)}, got {tuple(condition_tensor.shape)}"
                )
            if condition_validity is None:
                condition_validity = torch.ones_like(condition_tensor, dtype=torch.bool)
                condition_validity[:, 2] = condition_tensor[:, 3] >= 0.5
        if condition_validity.ndim == 1:
            condition_validity = condition_validity.unsqueeze(0)
        if condition_validity.shape[0] == 1:
            condition_validity = condition_validity.expand(num_samples, -1)
        if condition_validity.shape != (num_samples, 4):
            raise ValueError(
                "condition_validity must have shape [4], [1, 4], or [num_samples, 4]"
            )
        condition_validity = condition_validity.to(device=device, dtype=torch.bool)
        embedding = self.condition_encoder(condition_tensor, condition_validity)
        latent = torch.randn(
            num_samples, self.latent_dim, device=device, generator=generator
        )
        decoded = self.decoder(latent, embedding)
        if sample_path_count:
            probabilities = torch.softmax(decoded["path_count_logits"], dim=-1)
            count = torch.multinomial(
                probabilities, 1, replacement=True, generator=generator
            ).squeeze(-1) + 1
        else:
            count = decoded["path_count_logits"].argmax(-1) + 1
        path_mask = mask_from_path_count(count, self.max_paths)
        constrained = self.constraints(
            decoded["delays_ns"],
            decoded["positive_power_weights"],
            path_mask,
            condition_tensor,
            target_condition_validity=condition_validity,
        )
        csi = self._synthesize(decoded, constrained, path_mask)
        return {
            "csi": csi,
            "path_count": count,
            "path_mask": path_mask,
            "path_delays_ns": constrained.delays_ns,
            "path_powers_linear": constrained.powers_linear,
            "path_phases_rad": decoded["phases_rad"],
            "aoa_az_deg": decoded["aoa_az_deg"],
            "aoa_el_deg": decoded["aoa_el_deg"],
            "aod_az_deg": decoded["aod_az_deg"],
            "aod_el_deg": decoded["aod_el_deg"],
            "direct_path_mask": constrained.direct_path_mask,
            "target_conditions": condition_tensor,
            "target_condition_validity": condition_validity,
            "actual_conditions": constrained.actual_conditions,
            "actual_condition_validity": constrained.condition_validity,
            "constraint_feasible": constrained.feasible,
            "latent": latent,
        }

    def _path_features(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        power_db = 10.0 * torch.log10(batch["path_powers_linear"].clamp_min(1e-20))
        delay = batch["path_delays_ns"] / float(self.path_scales["delay_ns"])
        power = power_db / float(self.path_scales["log_power_db"])
        phase = batch["path_phases_rad"]
        angles = [
            torch.deg2rad(batch["aoa_az_deg"]),
            torch.deg2rad(batch["aoa_el_deg"]),
            torch.deg2rad(batch["aod_az_deg"]),
            torch.deg2rad(batch["aod_el_deg"]),
        ]
        features = [delay, power, torch.sin(phase), torch.cos(phase)]
        for angle in angles:
            features.extend([torch.sin(angle), torch.cos(angle)])
        return torch.stack(features, dim=-1)

    def _synthesize(self, decoded, constrained, path_mask):
        return self.synthesizer(
            constrained.delays_ns,
            constrained.powers_linear,
            decoded["phases_rad"],
            decoded["aoa_az_deg"],
            decoded["aoa_el_deg"],
            decoded["aod_az_deg"],
            decoded["aod_el_deg"],
            path_mask,
        )

