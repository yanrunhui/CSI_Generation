from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from physics_csi.models.diffusion.angle_delay_denoiser import AngleDelayDenoiser
from physics_csi.models.diffusion.angle_delay_transform import AngleDelayTransform
from physics_csi.models.diffusion.condition_encoder import ConditionEncoder
from physics_csi.models.diffusion.noise_schedule import GaussianDiffusion
from physics_csi.physics.constraints import check_condition_feasibility
from physics_csi.physics.synthesizer import SystemConfig


class AngleDelayDiffusionModel(nn.Module):
    condition_fields = ("total_power_db", "delay_spread_ns", "k_factor_db", "los_status")

    def __init__(
        self,
        config: Mapping[str, Any],
        system: SystemConfig,
        statistics: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.config = dict(config)
        condition_dim = int(config["condition_embedding_dim"])
        hidden_dim = int(config.get("condition_hidden_dim", config["channels"] * 4))
        self.condition_dropout = float(config.get("condition_dropout", 0.1))
        self.condition_encoder = ConditionEncoder(
            statistics["condition_mean"], statistics["condition_std"], condition_dim, hidden_dim
        )
        self.transform = AngleDelayTransform(system, float(statistics["angle_delay_scale"]))
        self.denoiser = AngleDelayDenoiser(
            condition_dim,
            int(config["channels"]),
            int(config["layers"]),
            float(config.get("dropout", 0.0)),
        )
        self.diffusion = GaussianDiffusion(
            int(config["diffusion_steps"]), float(config.get("cosine_offset", 0.008))
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        conditions = batch["conditions"]
        validity = batch.get("condition_validity")
        clean = self.transform(batch["csi"])
        drop_mask = torch.zeros(conditions.shape[0], device=conditions.device, dtype=torch.bool)
        if self.training and self.condition_dropout > 0.0:
            drop_mask = (
                torch.rand(conditions.shape[0], device=conditions.device)
                < self.condition_dropout
            )
        embedding = self.condition_encoder(conditions, validity, drop_mask=drop_mask)

        def predict(noisy: torch.Tensor, timestep: torch.Tensor, unconditional: bool) -> torch.Tensor:
            selected = (
                self.condition_encoder.unconditional(noisy.shape[0], noisy.device)
                if unconditional
                else embedding
            )
            return self.denoiser(noisy, timestep, selected)

        diffusion_loss = self.diffusion.training_loss(clean, predict)
        return {"loss": diffusion_loss, "diffusion_loss": diffusion_loss}

    @torch.no_grad()
    def generate(
        self,
        conditions: Mapping[str, float] | torch.Tensor,
        num_samples: int,
        *,
        condition_validity: torch.Tensor | None = None,
        sampler: str = "ddim",
        sampling_steps: int = 50,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
        calibration: str = "none",
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        values, validity = self._prepare_conditions(conditions, num_samples, condition_validity)
        embedding = self.condition_encoder(values, validity)

        def predict(noisy: torch.Tensor, timestep: torch.Tensor, unconditional: bool) -> torch.Tensor:
            selected = (
                self.condition_encoder.unconditional(num_samples, noisy.device)
                if unconditional
                else embedding
            )
            return self.denoiser(noisy, timestep, selected)

        representation = self.diffusion.sample(
            (num_samples, *self.transform.tensor_shape),
            predict,
            device=values.device,
            sampler=sampler,
            sampling_steps=sampling_steps,
            eta=eta,
            guidance_scale=guidance_scale,
            clip_denoised=float(self.config.get("clip_denoised", 5.0)),
            generator=generator,
        )
        csi = self.transform.inverse(representation)
        proxy_before = csi.abs().square().mean(dim=(1, 2, 3))
        if calibration == "csi_energy_proxy":
            target = torch.pow(10.0, values[:, 0] / 10.0)
            scale = (target / proxy_before.clamp_min(1e-20)).sqrt()
            csi = csi * scale[:, None, None, None]
        elif calibration != "none":
            raise ValueError("calibration must be 'none' or 'csi_energy_proxy'")
        proxy_after = csi.abs().square().mean(dim=(1, 2, 3))
        if not torch.isfinite(csi.real).all() or not torch.isfinite(csi.imag).all():
            raise FloatingPointError("angle-delay inverse transform produced non-finite CSI")
        return {
            "csi": csi.to(torch.complex64),
            "angle_delay": representation,
            "target_conditions": values,
            "target_condition_validity": validity,
            "csi_energy_proxy_before": proxy_before,
            "csi_energy_proxy_after": proxy_after,
            "calibration_mode": calibration,
        }

    def _prepare_conditions(
        self,
        conditions: Mapping[str, float] | torch.Tensor,
        num_samples: int,
        validity: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        device = next(self.parameters()).device
        if isinstance(conditions, Mapping):
            los = float(conditions["los_status"])
            check_condition_feasibility(conditions, k_factor_valid=los >= 0.5)
            row = torch.tensor([float(conditions[name]) for name in self.condition_fields], device=device)
            values = row.unsqueeze(0).expand(num_samples, -1)
            if validity is None:
                validity = torch.ones_like(values, dtype=torch.bool)
                validity[:, 2] = los >= 0.5
        else:
            values = conditions.to(device=device, dtype=torch.float32)
            if values.ndim == 1:
                values = values.unsqueeze(0)
            if values.shape[0] == 1:
                values = values.expand(num_samples, -1)
            if values.shape != (num_samples, 4):
                raise ValueError("conditions must have shape [num_samples, 4]")
            if validity is None:
                validity = torch.ones_like(values, dtype=torch.bool)
                validity[:, 2] = values[:, 3] >= 0.5
        validity = validity.to(device=device, dtype=torch.bool)
        if validity.ndim == 1:
            validity = validity.unsqueeze(0)
        if validity.shape[0] == 1:
            validity = validity.expand(num_samples, -1)
        if validity.shape != values.shape:
            raise ValueError("condition validity must match conditions")
        return values.float(), validity
