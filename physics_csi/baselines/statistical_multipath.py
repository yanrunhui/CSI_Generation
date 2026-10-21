from __future__ import annotations

from typing import Mapping

import torch

from physics_csi.baselines.conditional_resampling import ConditionalResampler
from physics_csi.physics.constraints import PhysicsConstraintLayer
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer


class StatisticalMultipathBaseline:
    """Local empirical bootstrap with phase/angle perturbation and hard constraints."""

    def __init__(
        self,
        reference: Mapping[str, torch.Tensor],
        synthesizer: MIMOOFDMSynthesizer,
        constraints: PhysicsConstraintLayer,
    ) -> None:
        self.resampler = ConditionalResampler(reference)
        self.synthesizer = synthesizer
        self.constraints = constraints

    @torch.no_grad()
    def generate(
        self,
        target_conditions: torch.Tensor,
        num_samples: int,
        *,
        angle_jitter_std_deg: float = 2.0,
        delay_jitter_std_ns: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        selected = self.resampler.sample(
            target_conditions,
            num_samples,
            method="kernel",
            bandwidth=torch.tensor([5.0, 30.0, 3.0, 0.1]),
            generator=generator,
        ).samples
        conditions = target_conditions.float().reshape(1, 4).expand(num_samples, -1)
        validity = torch.ones_like(conditions, dtype=torch.bool)
        validity[:, 2] = conditions[:, 3] >= 0.5
        mask = selected["path_mask"].bool()
        delays = (
            selected["path_delays_ns"]
            + torch.randn(selected["path_delays_ns"].shape, generator=generator)
            * delay_jitter_std_ns
        ).clamp_min(0.0)
        power_weights = selected["path_powers_linear"].clamp_min(1e-12)
        constrained = self.constraints(
            delays,
            power_weights,
            mask,
            conditions,
            target_condition_validity=validity,
        )
        result = dict(selected)
        for key in ("aoa_az_deg", "aoa_el_deg", "aod_az_deg", "aod_el_deg"):
            result[key] = selected[key] + torch.randn(
                selected[key].shape, generator=generator
            ) * angle_jitter_std_deg
        result["path_phases_rad"] = (
            selected["path_phases_rad"]
            + 2.0 * torch.pi * torch.rand(selected["path_phases_rad"].shape, generator=generator)
        )
        result["path_delays_ns"] = constrained.delays_ns
        result["path_powers_linear"] = constrained.powers_linear
        result["direct_path_mask"] = constrained.direct_path_mask
        result["target_conditions"] = conditions
        result["actual_conditions"] = constrained.actual_conditions
        result["csi"] = self.synthesizer(
            result["path_delays_ns"],
            result["path_powers_linear"],
            result["path_phases_rad"],
            result["aoa_az_deg"],
            result["aoa_el_deg"],
            result["aod_az_deg"],
            result["aod_el_deg"],
            mask,
        )
        return result

