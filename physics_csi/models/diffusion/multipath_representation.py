from __future__ import annotations

from collections.abc import Mapping

import torch


FEATURE_NAMES = (
    "first_delay_log_ns",
    "relative_delay_log_ns",
    "log_power_fraction_db",
    "phase_sin",
    "phase_cos",
    "aoa_az_sin",
    "aoa_az_cos",
    "aoa_el_scaled",
    "aod_az_sin",
    "aod_az_cos",
    "aod_el_scaled",
    "direct_flag",
)


class MultipathRepresentation:
    """Encode/decode delay-sorted padded paths without circular discontinuities."""

    feature_dim = len(FEATURE_NAMES)

    def __init__(
        self,
        statistics: Mapping[str, float],
        max_paths: int,
        *,
        delay_mode: str = "legacy_ns",
    ) -> None:
        self.max_paths = int(max_paths)
        if delay_mode not in {"legacy_ns", "rms_normalized_shape"}:
            raise ValueError(f"unsupported delay representation {delay_mode!r}")
        self.delay_mode = delay_mode
        self.relative_delay_statistic = (
            "relative_delay_shape_log"
            if delay_mode == "rms_normalized_shape"
            else "relative_delay_log_ns"
        )
        self.statistics = {key: float(value) for key, value in statistics.items()}
        for prefix in (
            "first_delay_log_ns",
            self.relative_delay_statistic,
            "log_power_fraction_db",
        ):
            if f"{prefix}_mean" not in self.statistics or f"{prefix}_std" not in self.statistics:
                raise KeyError(f"missing representation statistic for {prefix}")

    def _standardize(self, value: torch.Tensor, name: str) -> torch.Tensor:
        return (value - self.statistics[f"{name}_mean"]) / max(
            self.statistics[f"{name}_std"], 1e-6
        )

    def _restore(self, value: torch.Tensor, name: str) -> torch.Tensor:
        return value * max(self.statistics[f"{name}_std"], 1e-6) + self.statistics[f"{name}_mean"]

    def feature_mask(self, path_mask: torch.Tensor) -> torch.Tensor:
        mask = path_mask.bool()
        feature_mask = mask.unsqueeze(-1).expand(*mask.shape, self.feature_dim).clone()
        feature_mask[:, :, 0] = False
        feature_mask[:, 0, 0] = mask[:, 0]
        return feature_mask

    def encode(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        mask = batch["path_mask"].bool()
        if mask.shape[-1] != self.max_paths:
            raise ValueError("path mask width does not match max_paths")
        delays = batch["path_delays_ns"].float()
        first = torch.where(mask, delays, torch.full_like(delays, torch.inf)).amin(-1)
        relative = (delays - first.unsqueeze(-1)).clamp_min(0.0)
        powers = batch["path_powers_linear"].float() * mask
        fractions = powers / powers.sum(-1, keepdim=True).clamp_min(1e-20)
        if self.delay_mode == "rms_normalized_shape":
            mean = (fractions * relative).sum(-1, keepdim=True)
            spread = (
                fractions * (relative - mean).square()
            ).sum(-1, keepdim=True).clamp_min(0.0).sqrt()
            relative = torch.where(spread > 1e-8, relative / spread.clamp_min(1e-8), relative)
        phase = batch["path_phases_rad"].float()
        aoa_az = torch.deg2rad(batch["aoa_az_deg"].float())
        aod_az = torch.deg2rad(batch["aod_az_deg"].float())
        features = torch.stack(
            [
                torch.zeros_like(delays),
                self._standardize(
                    torch.log1p(relative), self.relative_delay_statistic
                ),
                self._standardize(10.0 * torch.log10(fractions.clamp_min(1e-20)), "log_power_fraction_db"),
                phase.sin(),
                phase.cos(),
                aoa_az.sin(),
                aoa_az.cos(),
                batch["aoa_el_deg"].float() / 90.0,
                aod_az.sin(),
                aod_az.cos(),
                batch["aod_el_deg"].float() / 90.0,
                batch["direct_path_mask"].float() * 2.0 - 1.0,
            ],
            dim=-1,
        )
        features[:, 0, 0] = self._standardize(torch.log1p(first), "first_delay_log_ns")
        feature_mask = self.feature_mask(mask)
        return features * feature_mask, feature_mask

    def decode(self, features: torch.Tensor, path_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 3 or features.shape[1:] != (self.max_paths, self.feature_dim):
            raise ValueError("invalid multipath diffusion tensor shape")
        if not torch.isfinite(features).all():
            raise FloatingPointError("multipath diffusion features contain NaN or Inf")
        # Diffusion operates in normalized coordinates.  Keep extrapolation
        # bounded before exp/expm1 decoding; otherwise a single imperfect
        # high-noise prediction can overflow all subsequent physics metrics.
        features = features.clamp(-6.0, 6.0)
        mask = path_mask.bool()
        first = torch.expm1(self._restore(features[:, 0, 0], "first_delay_log_ns")).clamp_min(0.0)
        relative = torch.expm1(
            self._restore(features[:, :, 1], self.relative_delay_statistic)
        ).clamp_min(0.0)
        relative[:, 0] = 0.0
        order = torch.where(mask, relative, torch.full_like(relative, torch.inf)).argsort(dim=-1)
        gather = order.unsqueeze(-1).expand_as(features)
        sorted_features = features.gather(1, gather)
        sorted_relative = relative.gather(1, order)
        sorted_mask = mask.gather(1, order)
        sorted_relative = sorted_relative - sorted_relative[:, :1]

        power_db = self._restore(sorted_features[:, :, 2], "log_power_fraction_db")
        power_weights = torch.pow(10.0, power_db / 10.0).clamp_min(1e-20) * sorted_mask

        def circular(sin_index: int, cos_index: int) -> torch.Tensor:
            return torch.atan2(
                sorted_features[:, :, sin_index], sorted_features[:, :, cos_index]
            )

        result = {
            "path_mask": sorted_mask,
            "first_delay_ns": first,
            "positive_power_weights": power_weights,
            "path_phases_rad": circular(3, 4),
            "aoa_az_deg": torch.rad2deg(circular(5, 6)),
            "aoa_el_deg": (sorted_features[:, :, 7] * 90.0).clamp(-90.0, 90.0),
            "aod_az_deg": torch.rad2deg(circular(8, 9)),
            "aod_el_deg": (sorted_features[:, :, 10] * 90.0).clamp(-90.0, 90.0),
            "direct_path_mask": (sorted_features[:, :, 11] >= 0.0) & sorted_mask,
        }
        if self.delay_mode == "rms_normalized_shape":
            result["relative_delay_shape"] = sorted_relative * sorted_mask
        else:
            result["path_delays_ns"] = (
                first.unsqueeze(-1) + sorted_relative
            ) * sorted_mask
        return result

