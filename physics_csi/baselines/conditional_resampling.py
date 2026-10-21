from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass
class ResamplingResult:
    samples: dict[str, torch.Tensor]
    source_indices: torch.Tensor
    condition_distance: torch.Tensor


class ConditionalResampler:
    """Training-only empirical baseline for conditional selection and sampling."""

    def __init__(
        self,
        reference: Mapping[str, torch.Tensor],
        scales: torch.Tensor | None = None,
    ) -> None:
        required = {"conditions", "condition_validity"}
        if not required.issubset(reference):
            raise ValueError(f"Reference payload is missing {sorted(required - reference.keys())}")
        self.reference = dict(reference)
        self.conditions = reference["conditions"].float().cpu()
        self.validity = reference["condition_validity"].bool().cpu()
        self.scales = (
            scales.float().cpu()
            if scales is not None
            else torch.tensor([20.0, 100.0, 10.0, 1.0])
        )

    def sample(
        self,
        target_conditions: torch.Tensor,
        num_samples: int,
        *,
        method: str = "nearest",
        threshold: torch.Tensor | None = None,
        bandwidth: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> ResamplingResult:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        target = target_conditions.float().cpu()
        if target.ndim != 1 or target.numel() != 4:
            raise ValueError("target_conditions must have shape [4]")
        comparable = self.validity.clone()
        if target[3] < 0.5:
            comparable[:, 2] = False
        difference = (self.conditions - target) / self.scales
        difference = torch.where(comparable, difference, torch.zeros_like(difference))
        distance = difference.square().sum(-1).sqrt()
        los_match = (self.conditions[:, 3] >= 0.5) == (target[3] >= 0.5)
        required_validity = torch.ones_like(los_match)
        if target[3] >= 0.5:
            required_validity &= self.validity[:, 2]
        eligible = los_match & required_validity
        distance = distance.masked_fill(~eligible, torch.inf)
        if not torch.isfinite(distance).any():
            raise ValueError("No eligible training sample has compatible condition validity")

        if method == "nearest":
            eligible_indices = torch.nonzero(
                torch.isfinite(distance), as_tuple=False
            ).flatten()
            local = torch.topk(
                distance[eligible_indices],
                k=min(max(num_samples, 1), eligible_indices.numel()),
                largest=False,
            ).indices
            candidates = eligible_indices[local]
            indices = candidates[
                torch.randint(candidates.numel(), (num_samples,), generator=generator)
            ]
        elif method == "threshold":
            if threshold is None:
                raise ValueError("threshold method requires a four-element threshold")
            absolute = (self.conditions - target).abs()
            keep = ((absolute <= threshold.cpu()) | ~comparable).all(-1) & eligible
            candidates = torch.nonzero(keep, as_tuple=False).flatten()
            if candidates.numel() == 0:
                raise ValueError("No training sample satisfies the requested thresholds")
            indices = candidates[
                torch.randint(candidates.numel(), (num_samples,), generator=generator)
            ]
        elif method == "kernel":
            if bandwidth is None:
                raise ValueError("kernel method requires four bandwidths")
            normalized = torch.where(
                comparable,
                (self.conditions - target) / bandwidth.cpu().clamp_min(1e-6),
                torch.zeros_like(self.conditions),
            )
            log_weight = -0.5 * normalized.square().sum(-1)
            log_weight = log_weight.masked_fill(~eligible, -torch.inf)
            probabilities = torch.softmax(log_weight, dim=0)
            if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
                raise ValueError("Kernel weights are degenerate for the requested condition")
            indices = torch.multinomial(
                probabilities, num_samples, replacement=True, generator=generator
            )
        else:
            raise ValueError(f"Unknown resampling method {method!r}")

        samples = {
            key: value[indices]
            for key, value in self.reference.items()
            if isinstance(value, torch.Tensor) and value.shape[0] == self.conditions.shape[0]
        }
        return ResamplingResult(samples, indices, distance[indices])

