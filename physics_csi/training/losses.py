from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(eps)


def circular_error_degrees(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    difference = torch.deg2rad(prediction - target)
    return 1.0 - torch.cos(difference)


def circular_error_radians(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.cos(prediction - target)


def complex_nmse(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    numerator = (prediction - target).abs().square().flatten(1).sum(-1)
    denominator = target.abs().square().flatten(1).sum(-1).clamp_min(eps)
    return (numerator / denominator).mean()


class CVAELoss(nn.Module):
    def __init__(
        self,
        weights: Mapping[str, float],
        *,
        delay_scale_ns: float = 1000.0,
        physics_scales: tuple[float, float, float, float] = (20.0, 100.0, 10.0, 1.0),
    ) -> None:
        super().__init__()
        self.weights = {key: float(value) for key, value in weights.items()}
        self.delay_scale_ns = delay_scale_ns
        self.register_buffer("physics_scales", torch.tensor(physics_scales))

    def forward(
        self,
        output: Mapping[str, torch.Tensor],
        target: Mapping[str, torch.Tensor],
        *,
        beta: float,
    ) -> dict[str, torch.Tensor]:
        mask = target["path_mask"].bool()
        delay = masked_mean(
            F.smooth_l1_loss(
                output["path_delays_ns"] / self.delay_scale_ns,
                target["path_delays_ns"] / self.delay_scale_ns,
                reduction="none",
            ),
            mask,
        )
        predicted_power_db = 10.0 * torch.log10(output["path_powers_linear"].clamp_min(1e-20))
        target_power_db = 10.0 * torch.log10(target["path_powers_linear"].clamp_min(1e-20))
        power = masked_mean(
            F.smooth_l1_loss(predicted_power_db, target_power_db, reduction="none") / 20.0,
            mask,
        )
        phase = masked_mean(
            circular_error_radians(output["phases_rad"], target["path_phases_rad"]), mask
        )
        angle_terms = []
        for key in ("aoa_az_deg", "aoa_el_deg", "aod_az_deg", "aod_el_deg"):
            angle_terms.append(masked_mean(circular_error_degrees(output[key], target[key]), mask))
        angle = torch.stack(angle_terms).mean()
        path_reconstruction = delay + power + phase + angle

        path_count = F.cross_entropy(
            output["path_count_logits"], target["path_count"].long() - 1
        )
        validity = F.binary_cross_entropy_with_logits(
            output["validity_logits"], mask.float()
        )
        direct_path = F.binary_cross_entropy_with_logits(
            output["direct_path_logits"], target["direct_path_mask"].float()
        )
        csi = complex_nmse(output["csi"], target["csi"])
        mean = output["latent_mean"]
        log_variance = output["latent_log_variance"]
        kl = -0.5 * (1.0 + log_variance - mean.square() - log_variance.exp()).sum(-1).mean()

        condition_validity = target["condition_validity"].bool()
        normalized_difference = (
            output["actual_conditions"] - target["conditions"]
        ) / self.physics_scales.clamp_min(1e-6)
        physics = masked_mean(normalized_difference.square(), condition_validity)

        total = (
            self.weights["path"] * path_reconstruction
            + self.weights["csi"] * csi
            + self.weights["kl"] * beta * kl
            + self.weights["physics"] * physics
            + self.weights["path_count"] * path_count
            + self.weights["validity"] * validity
            + self.weights["direct_path"] * direct_path
        )
        return {
            "loss": total,
            "path": path_reconstruction.detach(),
            "delay": delay.detach(),
            "power": power.detach(),
            "phase": phase.detach(),
            "angle": angle.detach(),
            "csi_nmse": csi.detach(),
            "kl": kl.detach(),
            "physics": physics.detach(),
            "path_count": path_count.detach(),
            "validity": validity.detach(),
            "direct_path": direct_path.detach(),
        }

