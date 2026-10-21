from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from physics_csi.data.physical_labels import compute_physical_conditions


@dataclass
class ConstraintResult:
    delays_ns: torch.Tensor
    powers_linear: torch.Tensor
    direct_path_mask: torch.Tensor
    actual_conditions: torch.Tensor
    condition_validity: torch.Tensor
    feasible: torch.Tensor


def check_condition_feasibility(
    conditions: Mapping[str, float],
    *,
    path_count: int | None = None,
    k_factor_valid: bool = True,
) -> None:
    required = {"total_power_db", "delay_spread_ns", "k_factor_db", "los_status"}
    missing = required - conditions.keys()
    if missing:
        raise ValueError(f"Missing physical conditions: {sorted(missing)}")
    values = {key: float(conditions[key]) for key in required}
    if not all(torch.isfinite(torch.tensor(value)) for value in values.values()):
        raise ValueError("All supplied condition values must be finite")
    if values["delay_spread_ns"] < 0:
        raise ValueError("delay_spread_ns cannot be negative")
    if values["los_status"] not in {0.0, 1.0}:
        raise ValueError("los_status must be 0 or 1")
    if path_count is not None:
        if path_count < 1:
            raise ValueError("At least one active path is required")
        if values["delay_spread_ns"] > 0 and path_count < 2:
            raise ValueError("Positive delay spread requires at least two active paths")
        if k_factor_valid and path_count < 2:
            raise ValueError("A finite direct-to-remaining K-factor requires at least two paths")
    if k_factor_valid and values["los_status"] == 0:
        raise ValueError("Rician K-factor control is undefined for NLoS under the default policy")


def first_active_path_mask(path_mask: torch.Tensor) -> torch.Tensor:
    mask = path_mask.bool()
    return mask & (mask.to(torch.int64).cumsum(dim=-1) == 1)


class PhysicsConstraintLayer(nn.Module):
    def __init__(
        self,
        *,
        enforce_total_power: bool = True,
        enforce_delay_spread: bool = True,
        enforce_k_factor: bool = True,
        enforce_los_direct_path: bool = True,
        epsilon: float = 1e-20,
        invalid_k_fill_db: float = 0.0,
    ) -> None:
        super().__init__()
        self.enforce_total_power = enforce_total_power
        self.enforce_delay_spread = enforce_delay_spread
        self.enforce_k_factor = enforce_k_factor
        self.enforce_los_direct_path = enforce_los_direct_path
        # Scientific-notation values in YAML can be returned as strings by
        # some PyYAML resolver versions, so normalize them at the boundary.
        self.epsilon = float(epsilon)
        self.invalid_k_fill_db = float(invalid_k_fill_db)

    def forward(
        self,
        delays_ns: torch.Tensor,
        positive_power_weights: torch.Tensor,
        path_mask: torch.Tensor,
        target_conditions: torch.Tensor,
        *,
        target_condition_validity: torch.Tensor | None = None,
        direct_path_mask: torch.Tensor | None = None,
    ) -> ConstraintResult:
        mask = path_mask.bool()
        if target_conditions.shape[-1] != 4:
            raise ValueError("target_conditions must use four-field canonical order")
        if direct_path_mask is None:
            direct = first_active_path_mask(mask) & (target_conditions[:, 3:4] >= 0.5)
        else:
            direct = direct_path_mask.bool() & mask
        if self.enforce_los_direct_path:
            los = target_conditions[:, 3:4] >= 0.5
            direct = torch.where(los, first_active_path_mask(mask), torch.zeros_like(mask))

        raw_power = positive_power_weights.float().clamp_min(self.epsilon)
        raw_power = raw_power * mask.to(raw_power.dtype)
        target_total = torch.pow(10.0, target_conditions[:, 0].float() / 10.0)
        powers = self._normalize_total(raw_power, mask, target_total)

        k_valid = (
            target_condition_validity[:, 2].bool()
            if target_condition_validity is not None
            else (target_conditions[:, 3] >= 0.5)
        )
        k_feasible = direct.any(dim=-1) & (mask & ~direct).any(dim=-1)
        apply_k = k_valid & k_feasible & self.enforce_k_factor
        if apply_k.any():
            powers = self._project_k_factor(
                powers, mask, direct, target_total, target_conditions[:, 2], apply_k
            )

        projected_delays, delay_feasible = self._project_delay_spread(
            delays_ns.float(), powers, mask, target_conditions[:, 1].float()
        )
        actual, validity = compute_physical_conditions(
            powers,
            projected_delays,
            mask,
            direct,
            eps=self.epsilon,
            invalid_k_fill_db=self.invalid_k_fill_db,
        )
        feasible = mask.any(dim=-1) & delay_feasible & (~k_valid | k_feasible)
        return ConstraintResult(
            delays_ns=projected_delays,
            powers_linear=powers,
            direct_path_mask=direct,
            actual_conditions=actual,
            condition_validity=validity,
            feasible=feasible,
        )

    def _normalize_total(
        self,
        weights: torch.Tensor,
        mask: torch.Tensor,
        target_total: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enforce_total_power:
            return weights
        denominator = weights.sum(dim=-1, keepdim=True).clamp_min(self.epsilon)
        return weights / denominator * target_total.unsqueeze(-1)

    def _project_k_factor(
        self,
        powers: torch.Tensor,
        mask: torch.Tensor,
        direct: torch.Tensor,
        target_total: torch.Tensor,
        target_k_db: torch.Tensor,
        apply: torch.Tensor,
    ) -> torch.Tensor:
        k_linear = torch.pow(10.0, target_k_db.float() / 10.0)
        direct_total = target_total * k_linear / (1.0 + k_linear)
        remaining_total = target_total / (1.0 + k_linear)
        direct_weights = powers * direct.to(powers.dtype)
        remaining_weights = powers * (mask & ~direct).to(powers.dtype)
        direct_projected = direct_weights / direct_weights.sum(-1, keepdim=True).clamp_min(
            self.epsilon
        ) * direct_total.unsqueeze(-1)
        remaining_projected = remaining_weights / remaining_weights.sum(
            -1, keepdim=True
        ).clamp_min(self.epsilon) * remaining_total.unsqueeze(-1)
        projected = direct_projected + remaining_projected
        return torch.where(apply.unsqueeze(-1), projected, powers)

    def _project_delay_spread(
        self,
        delays_ns: torch.Tensor,
        powers: torch.Tensor,
        mask: torch.Tensor,
        target_spread_ns: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enforce_delay_spread:
            return delays_ns * mask, mask.any(dim=-1)
        weights = powers * mask.to(powers.dtype)
        total = weights.sum(-1).clamp_min(self.epsilon)
        mean = (weights * delays_ns).sum(-1) / total
        centered = delays_ns - mean.unsqueeze(-1)
        variance = (weights * centered.square()).sum(-1) / total
        # A single active path has exactly zero delay variance.  sqrt(0) has
        # an infinite derivative, which can turn a subsequently masked zero
        # gradient into NaN during backward.  Clamp before sqrt for a finite
        # derivative, while using the unclamped variance for feasibility.
        has_nonzero_spread = variance > self.epsilon
        current_spread = variance.clamp_min(self.epsilon).sqrt()
        active_count = mask.sum(-1)
        positive_target = target_spread_ns > self.epsilon
        feasible = (~positive_target) | (
            (active_count >= 2) & has_nonzero_spread
        )
        scale = torch.where(
            positive_target & feasible,
            target_spread_ns / current_spread.clamp_min(self.epsilon),
            torch.zeros_like(target_spread_ns),
        )
        first_delay = torch.where(
            mask, delays_ns, torch.full_like(delays_ns, torch.inf)
        ).amin(-1)
        relative_delay = delays_ns - first_delay.unsqueeze(-1)
        projected = first_delay.unsqueeze(-1) + relative_delay * scale.unsqueeze(-1)
        projected = torch.where(mask, projected.clamp_min(0.0), torch.zeros_like(projected))
        projected = torch.where(feasible.unsqueeze(-1), projected, delays_ns * mask)
        return projected, feasible

