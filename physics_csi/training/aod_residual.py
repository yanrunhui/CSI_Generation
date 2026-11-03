from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from physics_csi.evaluation.research_diagnostics import (
    direction_vectors,
    normalized_path_power,
    power_aod_joint_features,
)
from physics_csi.models.scene_aod_residual import AoDResidualOutput


def _fixed_directions(dimensions: int, projections: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(projections, dimensions, generator=generator)
    return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _swd(candidate: torch.Tensor, reference: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    candidate_projection = (candidate @ directions.T).sort(dim=0).values
    reference_projection = (reference @ directions.T).sort(dim=0).values
    if candidate_projection.shape[0] != reference_projection.shape[0]:
        count = min(candidate_projection.shape[0], reference_projection.shape[0])
        if count < 2:
            raise ValueError("at least two samples are required for AoD distribution loss")
        candidate_projection = F.interpolate(
            candidate_projection.T.unsqueeze(0), size=count, mode="linear", align_corners=True
        ).squeeze(0).T
        reference_projection = F.interpolate(
            reference_projection.T.unsqueeze(0), size=count, mode="linear", align_corners=True
        ).squeeze(0).T
    return (candidate_projection - reference_projection).abs().mean()


def _weighted_quantiles(
    projected: torch.Tensor, weights: torch.Tensor, quantiles: torch.Tensor
) -> torch.Tensor:
    values, order = projected.T.sort(dim=-1)
    sorted_weights = weights[order]
    cumulative = sorted_weights.cumsum(-1)
    cumulative = cumulative / cumulative[:, -1:].clamp_min(1e-20)
    queries = quantiles.unsqueeze(0).expand(values.shape[0], -1).contiguous()
    positions = torch.searchsorted(cumulative.contiguous(), queries, right=False)
    return values.gather(1, positions.clamp_max(values.shape[1] - 1))


@dataclass(frozen=True)
class AoDLossWeights:
    path_power_aod: float = 0.5
    delay_power_aod: float = 0.5
    link_power_aod: float = 1.0
    power_weighted_aod: float = 2.0
    shuffled_margin: float = 0.5
    residual_strength: float = 0.02
    diversity: float = 0.10


class AoDDistributionLoss(nn.Module):
    """Unpaired Power--AoD objective for the frozen B3+DP hierarchy."""

    def __init__(
        self,
        *,
        max_paths: int,
        power_db_mean: float,
        power_db_std: float,
        projections: int = 64,
        weighted_quantiles: int = 128,
        margin: float = 0.01,
        diversity_minimum_ratio: float = 0.85,
        maximum_azimuth_residual_deg: float = 20.0,
        maximum_elevation_residual_deg: float = 10.0,
        weights: AoDLossWeights | None = None,
        seed: int = 2026,
    ) -> None:
        super().__init__()
        self.max_paths = int(max_paths)
        self.power_db_mean = float(power_db_mean)
        self.power_db_std = max(float(power_db_std), 1e-6)
        self.margin = float(margin)
        self.diversity_minimum_ratio = float(diversity_minimum_ratio)
        self.maximum_azimuth_residual_deg = float(maximum_azimuth_residual_deg)
        self.maximum_elevation_residual_deg = float(maximum_elevation_residual_deg)
        self.weights = weights or AoDLossWeights()
        self.register_buffer("path_directions", _fixed_directions(4, projections, seed))
        self.register_buffer("joint_directions", _fixed_directions(5, projections, seed + 1))
        self.register_buffer("link_directions", _fixed_directions(11, projections, seed + 2))
        self.register_buffer("weighted_directions", _fixed_directions(3, projections, seed + 3))
        self.register_buffer(
            "quantiles", (torch.arange(weighted_quantiles, dtype=torch.float32) + 0.5) / weighted_quantiles
        )

    def _path_features(self, payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
        mask = payload["path_mask"].bool()
        direction = direction_vectors(payload["aod_az_deg"], payload["aod_el_deg"])
        fraction = normalized_path_power(payload)
        power_db = 10.0 * torch.log10(fraction.clamp_min(1e-20))
        normalized_power = (power_db - self.power_db_mean) / self.power_db_std
        return torch.cat([direction, normalized_power.unsqueeze(-1)], dim=-1)[mask]

    def _weighted_aod(self, payload: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        mask = payload["path_mask"].bool()
        points = direction_vectors(payload["aod_az_deg"], payload["aod_el_deg"])[mask]
        weights = normalized_path_power(payload)
        weights = (weights / max(mask.shape[0], 1))[mask]
        return points, weights

    def _delay_power_aod_features(
        self, payload: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        mask = payload["path_mask"].bool()
        delay = payload["path_delays_ns"].float()
        first = torch.where(mask, delay, torch.full_like(delay, torch.inf)).amin(-1)
        relative = (delay - first.unsqueeze(-1)).clamp_min(0.0) * mask
        fraction = normalized_path_power(payload)
        mean = (fraction * relative).sum(-1, keepdim=True)
        rms = (
            fraction * (relative - mean).square()
        ).sum(-1, keepdim=True).clamp_min(1e-20).sqrt()
        delay_shape = torch.log1p(relative / rms.clamp_min(1e-6))
        power_db = 10.0 * torch.log10(fraction.clamp_min(1e-20))
        normalized_power = (power_db - self.power_db_mean) / self.power_db_std
        direction = direction_vectors(payload["aod_az_deg"], payload["aod_el_deg"])
        return torch.cat(
            [delay_shape.unsqueeze(-1), normalized_power.unsqueeze(-1), direction], dim=-1
        )[mask]

    def distribution_terms(
        self,
        candidate: Mapping[str, torch.Tensor],
        reference: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        candidate_path = self._path_features(candidate)
        reference_path = self._path_features(reference)
        path_swd = _swd(candidate_path, reference_path, self.path_directions.to(candidate_path))
        candidate_joint = self._delay_power_aod_features(candidate)
        reference_joint = self._delay_power_aod_features(reference)
        joint_swd = _swd(
            candidate_joint, reference_joint, self.joint_directions.to(candidate_joint)
        )
        candidate_link = power_aod_joint_features(candidate)
        reference_link = power_aod_joint_features(reference)
        link_swd = _swd(candidate_link, reference_link, self.link_directions.to(candidate_link))
        candidate_points, candidate_weights = self._weighted_aod(candidate)
        reference_points, reference_weights = self._weighted_aod(reference)
        directions = self.weighted_directions.to(candidate_points)
        quantiles = self.quantiles.to(candidate_points)
        candidate_quantiles = _weighted_quantiles(
            candidate_points @ directions.T, candidate_weights, quantiles
        )
        reference_quantiles = _weighted_quantiles(
            reference_points @ directions.T, reference_weights, quantiles
        )
        weighted_swd = (candidate_quantiles - reference_quantiles).abs().mean()
        score = (
            self.weights.path_power_aod * path_swd
            + self.weights.delay_power_aod * joint_swd
            + self.weights.link_power_aod * link_swd
            + self.weights.power_weighted_aod * weighted_swd
        )
        return {
            "path_power_aod_swd": path_swd,
            "delay_power_aod_swd": joint_swd,
            "link_power_aod_swd": link_swd,
            "power_weighted_aod_swd": weighted_swd,
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
        dp_payload: Mapping[str, torch.Tensor],
        reference_payload: Mapping[str, torch.Tensor],
        correct_output: AoDResidualOutput,
    ) -> dict[str, torch.Tensor]:
        correct = self.distribution_terms(correct_payload, reference_payload)
        shuffled = self.distribution_terms(shuffled_payload, reference_payload)
        margin_loss = F.relu(
            correct["distribution_score"] + self.margin - shuffled["distribution_score"]
        )
        mask = correct_payload["path_mask"].bool()
        residual_strength = (
            (correct_output.azimuth_residual_deg[mask] / self.maximum_azimuth_residual_deg).square().mean()
            + (correct_output.elevation_residual_deg[mask] / self.maximum_elevation_residual_deg).square().mean()
        )
        dp_link = power_aod_joint_features(dp_payload)
        correct_link = power_aod_joint_features(correct_payload)
        base_diversity = self._pairwise_mean(dp_link).detach()
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
            "dp_link_diversity": base_diversity,
            "correct_link_diversity": correct_diversity,
            "azimuth_gate": correct_output.azimuth_gate,
            "elevation_gate": correct_output.elevation_gate,
        }
