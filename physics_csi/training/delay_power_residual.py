from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from physics_csi.models.scene_delay_power_residual import DelayPowerResidualOutput


def _relative_delay_and_fraction(
    payload: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = payload["path_mask"].bool()
    delay = payload["path_delays_ns"].float()
    first = torch.where(mask, delay, torch.full_like(delay, torch.inf)).amin(-1)
    relative = (delay - first.unsqueeze(-1)).clamp_min(0.0) * mask
    power = payload["path_powers_linear"].float().clamp_min(0.0) * mask
    fraction = power / power.sum(-1, keepdim=True).clamp_min(1e-20)
    return mask, relative, fraction


def _fixed_directions(dimensions: int, projections: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    directions = torch.randn(projections, dimensions, generator=generator)
    return directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _sliced_wasserstein(
    candidate: torch.Tensor, reference: torch.Tensor, directions: torch.Tensor
) -> torch.Tensor:
    if candidate.ndim != 2 or reference.ndim != 2:
        raise ValueError("Sliced Wasserstein inputs must be matrices")
    if candidate.shape[1] != reference.shape[1] or candidate.shape[1] != directions.shape[1]:
        raise ValueError("Sliced Wasserstein feature dimensions do not match")
    candidate_projection = (candidate @ directions.T).sort(dim=0).values
    reference_projection = (reference @ directions.T).sort(dim=0).values
    if candidate_projection.shape[0] != reference_projection.shape[0]:
        sample_count = min(candidate_projection.shape[0], reference_projection.shape[0])
        if sample_count < 2:
            raise ValueError("At least two samples are required for distribution loss")
        candidate_projection = F.interpolate(
            candidate_projection.T.unsqueeze(0),
            size=sample_count,
            mode="linear",
            align_corners=True,
        ).squeeze(0).T
        reference_projection = F.interpolate(
            reference_projection.T.unsqueeze(0),
            size=sample_count,
            mode="linear",
            align_corners=True,
        ).squeeze(0).T
    return (candidate_projection - reference_projection).abs().mean()


@dataclass(frozen=True)
class DelayPowerLossWeights:
    path_swd: float = 1.0
    link_swd: float = 1.0
    power_weighted_delay: float = 0.5
    shuffled_margin: float = 0.5
    residual_strength: float = 0.01
    diversity: float = 0.1


class DelayPowerDistributionLoss(nn.Module):
    """Unpaired group loss evaluated strictly after physical projection."""

    def __init__(
        self,
        *,
        max_paths: int,
        power_db_mean: float,
        power_db_std: float,
        projections: int = 64,
        weighted_delay_bins: int = 64,
        weighted_delay_maximum: float = 8.0,
        weighted_delay_temperature: float = 0.08,
        margin: float = 0.02,
        diversity_minimum_ratio: float = 0.8,
        weights: DelayPowerLossWeights | None = None,
        seed: int = 2026,
    ) -> None:
        super().__init__()
        self.max_paths = int(max_paths)
        self.power_db_mean = float(power_db_mean)
        self.power_db_std = max(float(power_db_std), 1e-6)
        self.margin = float(margin)
        self.diversity_minimum_ratio = float(diversity_minimum_ratio)
        self.weighted_delay_temperature = float(weighted_delay_temperature)
        self.weights = weights or DelayPowerLossWeights()
        self.register_buffer("path_directions", _fixed_directions(2, projections, seed))
        self.register_buffer(
            "link_directions", _fixed_directions(3 * self.max_paths, projections, seed + 1)
        )
        self.register_buffer(
            "weighted_delay_grid",
            torch.linspace(0.0, weighted_delay_maximum, weighted_delay_bins),
        )

    def _features(
        self, payload: Mapping[str, torch.Tensor], conditions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask, relative, fraction = _relative_delay_and_fraction(payload)
        spread = conditions[:, 1:2].float().clamp_min(1e-6)
        normalized_delay = torch.log1p(relative / spread)
        power_db = 10.0 * torch.log10(fraction.clamp_min(1e-20))
        normalized_power = (power_db - self.power_db_mean) / self.power_db_std
        path = torch.stack([normalized_delay, normalized_power], dim=-1)[mask]
        link = torch.cat(
            [mask.float(), normalized_delay * mask, normalized_power * mask], dim=-1
        )
        return path, link, normalized_delay, fraction

    def distribution_terms(
        self,
        candidate: Mapping[str, torch.Tensor],
        reference: Mapping[str, torch.Tensor],
        conditions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        candidate_path, candidate_link, candidate_delay, candidate_fraction = self._features(
            candidate, conditions
        )
        reference_path, reference_link, reference_delay, reference_fraction = self._features(
            reference, conditions
        )
        path_swd = _sliced_wasserstein(
            candidate_path, reference_path, self.path_directions.to(candidate_path)
        )
        link_swd = _sliced_wasserstein(
            candidate_link, reference_link, self.link_directions.to(candidate_link)
        )
        grid = self.weighted_delay_grid.to(candidate_delay)
        temperature = max(self.weighted_delay_temperature, 1e-4)

        def cdf(delay: torch.Tensor, fraction: torch.Tensor) -> torch.Tensor:
            soft_indicator = torch.sigmoid(
                (grid.reshape(1, 1, -1) - delay.unsqueeze(-1)) / temperature
            )
            return (fraction.unsqueeze(-1) * soft_indicator).sum(1).mean(0)

        weighted_delay = (
            cdf(candidate_delay, candidate_fraction)
            - cdf(reference_delay, reference_fraction)
        ).abs().mean()
        score = (
            self.weights.path_swd * path_swd
            + self.weights.link_swd * link_swd
            + self.weights.power_weighted_delay * weighted_delay
        )
        return {
            "path_delay_power_swd": path_swd,
            "link_delay_power_swd": link_swd,
            "power_weighted_delay_cdf": weighted_delay,
            "distribution_score": score,
        }

    @staticmethod
    def _pairwise_mean(features: torch.Tensor) -> torch.Tensor:
        if features.shape[0] < 2:
            return features.sum() * 0.0
        return torch.pdist(features, p=2).mean()

    def forward(
        self,
        *,
        correct_payload: Mapping[str, torch.Tensor],
        shuffled_payload: Mapping[str, torch.Tensor],
        base_payload: Mapping[str, torch.Tensor],
        reference_payload: Mapping[str, torch.Tensor],
        conditions: torch.Tensor,
        correct_output: DelayPowerResidualOutput,
    ) -> dict[str, torch.Tensor]:
        correct = self.distribution_terms(correct_payload, reference_payload, conditions)
        shuffled = self.distribution_terms(shuffled_payload, reference_payload, conditions)
        margin_loss = F.relu(
            correct["distribution_score"] + self.margin - shuffled["distribution_score"]
        )
        mask = correct_payload["path_mask"].bool()
        gap_mask = mask.clone()
        gap_mask[:, 0] = False
        residual_strength = (
            correct_output.delay_log_gap_residual[gap_mask].square().mean()
            if bool(gap_mask.any())
            else correct_output.delay_log_gap_residual.sum() * 0.0
        )
        residual_strength = residual_strength + correct_output.power_logit_residual[mask].square().mean()

        _, base_link, _, _ = self._features(base_payload, conditions)
        _, correct_link, _, _ = self._features(correct_payload, conditions)
        base_diversity = self._pairwise_mean(base_link).detach()
        correct_diversity = self._pairwise_mean(correct_link)
        diversity_loss = F.relu(
            self.diversity_minimum_ratio * base_diversity - correct_diversity
        ).square() / base_diversity.square().clamp_min(1e-8)

        total = (
            correct["distribution_score"]
            + self.weights.shuffled_margin * margin_loss
            + self.weights.residual_strength * residual_strength
            + self.weights.diversity * diversity_loss
        )
        return {
            "loss": total,
            **{f"correct_{key}": value for key, value in correct.items()},
            **{f"shuffled_{key}": value for key, value in shuffled.items()},
            "correct_vs_shuffled_margin_loss": margin_loss,
            "correct_advantage": shuffled["distribution_score"] - correct["distribution_score"],
            "residual_strength": residual_strength,
            "diversity_loss": diversity_loss,
            "base_link_diversity": base_diversity,
            "correct_link_diversity": correct_diversity,
            "projection_feasible_rate": correct_output.constrained.feasible.float().mean(),
        }
