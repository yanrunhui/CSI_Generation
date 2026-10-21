from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from physics_csi.physics.constraints import (
    ConstraintResult,
    PhysicsConstraintLayer,
    first_active_path_mask,
)


@dataclass
class ParameterizationResult:
    constrained: ConstraintResult
    parameterized_delays_ns: torch.Tensor
    parameterized_powers_linear: torch.Tensor
    delay_shape_rms: torch.Tensor
    delay_scale_ns: torch.Tensor
    delay_shape_repaired: torch.Tensor


class PhysicsAwareParameterization(nn.Module):
    """Turn learned dimensionless path structure into physical parameters.

    The diffusion model remains responsible for the joint relative-delay,
    power, phase and angle structure.  Deterministic identities for total
    power, RMS delay spread and the project-specific LoS K-factor definition
    are applied analytically before the legacy constraint layer is used as a
    final safety check.
    """

    def __init__(
        self,
        constraints: PhysicsConstraintLayer,
        config: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        values = dict(config or {})
        mode = str(values.get("mode", "legacy_projection"))
        if mode not in {"legacy_projection", "structured"}:
            raise ValueError(f"unsupported physics parameterization mode {mode!r}")
        self.mode = mode
        self.structured_delay = bool(values.get("structured_delay", mode == "structured"))
        self.structured_power = bool(values.get("structured_power", mode == "structured"))
        self.structured_k_factor = bool(
            values.get("structured_k_factor", mode == "structured")
        )
        self.epsilon = float(values.get("epsilon", constraints.epsilon))
        self.constraints = constraints

    def forward(
        self,
        raw: Mapping[str, torch.Tensor],
        target_conditions: torch.Tensor,
        target_validity: torch.Tensor,
    ) -> ParameterizationResult:
        mask = raw["path_mask"].bool()
        los = target_conditions[:, 3:4] >= 0.5
        direct = first_active_path_mask(mask) & los

        weights = raw["positive_power_weights"].float().clamp_min(self.epsilon) * mask
        target_total = torch.pow(10.0, target_conditions[:, 0].float() / 10.0)
        if self.structured_power:
            powers = weights / weights.sum(-1, keepdim=True).clamp_min(self.epsilon)
            powers = powers * target_total.unsqueeze(-1)
            k_valid = target_validity[:, 2].bool() & los.squeeze(-1)
            k_feasible = direct.any(-1) & (mask & ~direct).any(-1)
            apply_k = k_valid & k_feasible & self.structured_k_factor
            if apply_k.any():
                k_linear = torch.pow(10.0, target_conditions[:, 2].float() / 10.0)
                direct_fraction = k_linear / (1.0 + k_linear)
                scatter_fraction = 1.0 / (1.0 + k_linear)
                scatter_weights = weights * (mask & ~direct)
                scatter_weights = scatter_weights / scatter_weights.sum(
                    -1, keepdim=True
                ).clamp_min(self.epsilon)
                projected = (
                    direct.to(weights.dtype) * direct_fraction.unsqueeze(-1)
                    + scatter_weights * scatter_fraction.unsqueeze(-1)
                ) * target_total.unsqueeze(-1)
                powers = torch.where(apply_k.unsqueeze(-1), projected, powers)
        else:
            powers = weights

        delay_powers = powers
        if self.structured_delay and not self.structured_power:
            # The delay RMS must be computed using the same final power
            # fractions that the safety constraint will use.  This keeps the
            # B2 delay ablation isolated even though power construction itself
            # remains assigned to the legacy constraint layer.
            delay_powers = weights / weights.sum(-1, keepdim=True).clamp_min(self.epsilon)
            k_valid = target_validity[:, 2].bool() & los.squeeze(-1)
            k_feasible = direct.any(-1) & (mask & ~direct).any(-1)
            apply_k = k_valid & k_feasible & self.constraints.enforce_k_factor
            if apply_k.any():
                k_linear = torch.pow(10.0, target_conditions[:, 2].float() / 10.0)
                direct_fraction = k_linear / (1.0 + k_linear)
                scatter_fraction = 1.0 / (1.0 + k_linear)
                scatter = weights * (mask & ~direct)
                scatter = scatter / scatter.sum(-1, keepdim=True).clamp_min(self.epsilon)
                projected = (
                    direct.to(weights.dtype) * direct_fraction.unsqueeze(-1)
                    + scatter * scatter_fraction.unsqueeze(-1)
                )
                delay_powers = torch.where(apply_k.unsqueeze(-1), projected, delay_powers)

        delay_shape_rms = torch.zeros(mask.shape[0], device=mask.device)
        delay_scale = torch.ones_like(delay_shape_rms)
        repaired = torch.zeros_like(delay_shape_rms, dtype=torch.bool)
        if self.structured_delay:
            shape = raw["relative_delay_shape"].float().clamp_min(0.0) * mask
            fractions = delay_powers / delay_powers.sum(-1, keepdim=True).clamp_min(self.epsilon)
            mean = (fractions * shape).sum(-1, keepdim=True)
            variance = (fractions * (shape - mean).square()).sum(-1)
            delay_shape_rms = variance.clamp_min(0.0).sqrt()
            positive_target = target_conditions[:, 1] > self.epsilon
            feasible_count = mask.sum(-1) >= 2
            degenerate = positive_target & feasible_count & (delay_shape_rms <= 1e-8)
            if degenerate.any():
                ranks = torch.arange(
                    mask.shape[-1], device=mask.device, dtype=shape.dtype
                ).unsqueeze(0)
                shape = torch.where(degenerate.unsqueeze(-1), ranks * mask, shape)
                mean = (fractions * shape).sum(-1, keepdim=True)
                variance = (fractions * (shape - mean).square()).sum(-1)
                delay_shape_rms = variance.clamp_min(0.0).sqrt()
                repaired = degenerate
            delay_scale = torch.where(
                positive_target & feasible_count,
                target_conditions[:, 1].float()
                / delay_shape_rms.clamp_min(self.epsilon),
                torch.zeros_like(delay_shape_rms),
            )
            first = raw["first_delay_ns"].float().clamp_min(0.0)
            delays = (first.unsqueeze(-1) + shape * delay_scale.unsqueeze(-1)) * mask
        else:
            delays = raw["path_delays_ns"].float() * mask

        constrained = self.constraints(
            delays,
            powers,
            mask,
            target_conditions,
            target_condition_validity=target_validity,
            direct_path_mask=direct,
        )
        return ParameterizationResult(
            constrained=constrained,
            parameterized_delays_ns=delays,
            parameterized_powers_linear=powers,
            delay_shape_rms=delay_shape_rms,
            delay_scale_ns=delay_scale,
            delay_shape_repaired=repaired,
        )
