from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import torch


AoDControlMode = Literal["mean", "strongest"]


def wrap_degrees(angle_deg: torch.Tensor) -> torch.Tensor:
    """Wrap degrees to [-180, 180), including stable behavior at the boundary."""
    angle = angle_deg.float()
    return torch.remainder(angle + 180.0, 360.0) - 180.0


def circular_difference_degrees(
    first_deg: torch.Tensor, second_deg: torch.Tensor
) -> torch.Tensor:
    return wrap_degrees(first_deg.float() - second_deg.float())


@dataclass(frozen=True)
class AoDControlOutput:
    aod_az_deg: torch.Tensor
    target_az_deg: torch.Tensor
    reference_before_deg: torch.Tensor
    reference_after_deg: torch.Tensor
    rotation_deg: torch.Tensor
    circular_error_deg: torch.Tensor
    control_valid: torch.Tensor
    mean_resultant_concentration: torch.Tensor
    strongest_path_index: torch.Tensor
    strongest_power_margin_fraction: torch.Tensor
    strongest_near_tie: torch.Tensor
    mode: AoDControlMode


@dataclass(frozen=True)
class AoDCandidateSelection:
    indices: torch.Tensor
    angular_distance_deg: torch.Tensor
    eligible_mask: torch.Tensor
    pool_measurement: AoDControlOutput
    requested_samples: int
    candidate_samples: int
    eligible_samples: int


class AoDControlProjector:
    """Parameter-free common-azimuth rotation for generated multipath links.

    Mean mode controls the power-weighted circular mean. Strongest mode controls
    the deterministic first-argmax power path. Both modes rotate every active
    path equally and therefore preserve all relative azimuth intervals.
    """

    def __init__(
        self,
        *,
        minimum_mean_concentration: float = 0.10,
        strongest_near_tie_margin: float = 0.02,
    ) -> None:
        if not 0.0 <= minimum_mean_concentration <= 1.0:
            raise ValueError("minimum_mean_concentration must lie in [0, 1]")
        if not 0.0 <= strongest_near_tie_margin <= 1.0:
            raise ValueError("strongest_near_tie_margin must lie in [0, 1]")
        self.minimum_mean_concentration = float(minimum_mean_concentration)
        self.strongest_near_tie_margin = float(strongest_near_tie_margin)

    @staticmethod
    def _inputs(
        payload: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = payload["path_mask"].bool()
        azimuth = payload["aod_az_deg"].float()
        power = payload["path_powers_linear"].float().clamp_min(0.0) * mask
        if mask.ndim != 2 or azimuth.shape != mask.shape or power.shape != mask.shape:
            raise ValueError("path mask, AoD azimuth and path power must share [batch, path]")
        if not bool(mask.any(-1).all()):
            raise ValueError("every AoD-controlled sample must contain an active path")
        total = power.sum(-1, keepdim=True)
        if bool((total <= 0).any()):
            raise ValueError("every AoD-controlled sample must have positive active power")
        fraction = power / total.clamp_min(1e-20)
        radians = torch.deg2rad(azimuth)
        sine = (fraction * radians.sin()).sum(-1)
        cosine = (fraction * radians.cos()).sum(-1)
        concentration = torch.sqrt(sine.square() + cosine.square()).clamp(0.0, 1.0)
        mean = torch.rad2deg(torch.atan2(sine, cosine))
        return mask, azimuth, fraction, torch.stack([mean, concentration], dim=-1)

    @staticmethod
    def _strongest_audit(
        azimuth: torch.Tensor,
        fraction: torch.Tensor,
        mask: torch.Tensor,
        near_tie_margin: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        masked_power = fraction.masked_fill(~mask, -torch.inf)
        strongest_index = masked_power.argmax(-1)
        strongest_angle = azimuth.gather(1, strongest_index.unsqueeze(-1)).squeeze(-1)
        active_count = mask.sum(-1)
        top = masked_power.topk(min(2, mask.shape[1]), dim=-1).values
        first = top[:, 0]
        second = top[:, 1] if top.shape[1] > 1 else torch.zeros_like(first)
        second = torch.where(active_count >= 2, second, torch.zeros_like(second))
        relative_margin = (first - second).clamp_min(0.0) / first.clamp_min(1e-20)
        near_tie = (active_count >= 2) & (relative_margin <= near_tie_margin)
        return strongest_index, strongest_angle, relative_margin, near_tie

    def __call__(
        self,
        payload: Mapping[str, torch.Tensor],
        target_az_deg: float | torch.Tensor,
        *,
        mode: AoDControlMode = "mean",
    ) -> AoDControlOutput:
        if mode not in {"mean", "strongest"}:
            raise ValueError("AoD control mode must be 'mean' or 'strongest'")
        mask, azimuth, fraction, mean_audit = self._inputs(payload)
        mean_before = mean_audit[:, 0]
        concentration = mean_audit[:, 1]
        strongest_index, strongest_before, margin, near_tie = self._strongest_audit(
            azimuth,
            fraction,
            mask,
            self.strongest_near_tie_margin,
        )
        target = torch.as_tensor(
            target_az_deg, device=azimuth.device, dtype=azimuth.dtype
        ).reshape(-1)
        if target.numel() == 1:
            target = target.expand(azimuth.shape[0])
        if target.shape != (azimuth.shape[0],):
            raise ValueError("target_az_deg must be scalar or have shape [batch]")
        if not bool(torch.isfinite(target).all()):
            raise ValueError("target AoD azimuth must be finite")
        target = wrap_degrees(target)
        reference_before = mean_before if mode == "mean" else strongest_before
        rotation = circular_difference_degrees(target, reference_before)
        corrected = wrap_degrees(azimuth + rotation.unsqueeze(-1)) * mask

        corrected_payload = dict(payload)
        corrected_payload["aod_az_deg"] = corrected
        _, corrected_azimuth, corrected_fraction, corrected_mean_audit = self._inputs(
            corrected_payload
        )
        _, strongest_after, _, _ = self._strongest_audit(
            corrected_azimuth,
            corrected_fraction,
            mask,
            self.strongest_near_tie_margin,
        )
        reference_after = (
            corrected_mean_audit[:, 0] if mode == "mean" else strongest_after
        )
        error = circular_difference_degrees(reference_after, target).abs()
        valid = (
            concentration >= self.minimum_mean_concentration
            if mode == "mean"
            else torch.ones_like(concentration, dtype=torch.bool)
        )
        return AoDControlOutput(
            aod_az_deg=corrected,
            target_az_deg=target,
            reference_before_deg=reference_before,
            reference_after_deg=reference_after,
            rotation_deg=rotation,
            circular_error_deg=error,
            control_valid=valid,
            mean_resultant_concentration=concentration,
            strongest_path_index=strongest_index,
            strongest_power_margin_fraction=margin,
            strongest_near_tie=near_tie,
            mode=mode,
        )

    @staticmethod
    def apply_to_payload(
        payload: Mapping[str, torch.Tensor], output: AoDControlOutput
    ) -> dict[str, torch.Tensor]:
        result = {
            key: value for key, value in payload.items() if isinstance(value, torch.Tensor)
        }
        result["aod_az_deg"] = output.aod_az_deg
        return result


def select_nearest_aod_candidates(
    payload: Mapping[str, torch.Tensor],
    target_az_deg: float | torch.Tensor,
    *,
    samples: int,
    mode: AoDControlMode,
    projector: AoDControlProjector,
    exclude_strongest_near_ties: bool = False,
) -> AoDCandidateSelection:
    """Select unique generated links requiring the smallest common rotation.

    Selection is deterministic and without replacement. Mean-mode links with an
    unstable circular direction are excluded. Strongest-mode near ties remain
    eligible by default because the first-argmax definition is deterministic,
    but callers may exclude them for a stricter control policy.
    """
    if samples < 1:
        raise ValueError("samples must be positive")
    measured = projector(payload, target_az_deg, mode=mode)
    eligible = measured.control_valid.clone()
    if mode == "strongest" and exclude_strongest_near_ties:
        eligible &= ~measured.strongest_near_tie
    eligible_count = int(eligible.sum())
    if eligible_count < samples:
        raise ValueError(
            f"AoD candidate pool has only {eligible_count} eligible unique links, "
            f"but {samples} were requested"
        )
    distance = measured.rotation_deg.abs()
    ranked_distance = distance.masked_fill(~eligible, torch.inf)
    # Stable sorting makes equal-distance selection reproducible and retains the
    # original independent generation order rather than duplicating candidates.
    order = torch.argsort(ranked_distance, stable=True)
    indices = order[:samples]
    if int(torch.unique(indices).numel()) != samples:
        raise RuntimeError("AoD resampling selected a duplicate candidate")
    return AoDCandidateSelection(
        indices=indices,
        angular_distance_deg=distance[indices],
        eligible_mask=eligible,
        pool_measurement=measured,
        requested_samples=samples,
        candidate_samples=int(distance.numel()),
        eligible_samples=eligible_count,
    )


def select_aod_candidates_within_tolerance(
    payload: Mapping[str, torch.Tensor],
    target_az_deg: float | torch.Tensor,
    *,
    samples: int,
    tolerance_deg: float,
    mode: AoDControlMode,
    projector: AoDControlProjector,
    exclude_strongest_near_ties: bool = False,
) -> AoDCandidateSelection:
    """Accept natural links inside a fixed angular tolerance without projection.

    Eligible samples are retained in their independent generation order. This
    avoids silently tightening the requested tolerance by always choosing the
    closest subset, and it never duplicates a generated link.
    """
    if samples < 1:
        raise ValueError("samples must be positive")
    if not 0.0 <= tolerance_deg <= 180.0:
        raise ValueError("tolerance_deg must lie in [0, 180]")
    measured = projector(payload, target_az_deg, mode=mode)
    distance = measured.rotation_deg.abs()
    eligible = measured.control_valid & (distance <= float(tolerance_deg))
    if mode == "strongest" and exclude_strongest_near_ties:
        eligible &= ~measured.strongest_near_tie
    eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
    eligible_count = int(eligible_indices.numel())
    if eligible_count < samples:
        acceptance = eligible_count / max(int(distance.numel()), 1)
        raise ValueError(
            f"AoD candidate pool accepted {eligible_count}/{distance.numel()} "
            f"unique links ({acceptance:.3%}) within ±{tolerance_deg:g}°, "
            f"but {samples} were requested; increase the candidate multiplier"
        )
    indices = eligible_indices[:samples]
    return AoDCandidateSelection(
        indices=indices,
        angular_distance_deg=distance[indices],
        eligible_mask=eligible,
        pool_measurement=measured,
        requested_samples=samples,
        candidate_samples=int(distance.numel()),
        eligible_samples=eligible_count,
    )


def select_payload_samples(
    payload: Mapping[str, torch.Tensor], indices: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Select a common batch subset while ignoring non-batched tensors."""
    sample_count = int(payload["path_mask"].shape[0])
    return {
        key: value[indices.to(value.device)]
        for key, value in payload.items()
        if isinstance(value, torch.Tensor) and value.shape[:1] == (sample_count,)
    }
