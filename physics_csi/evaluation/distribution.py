from __future__ import annotations

import torch


def multipath_features(payload: dict[str, torch.Tensor]) -> torch.Tensor:
    mask = payload["path_mask"].bool()
    powers = payload["path_powers_linear"] * mask
    normalized_power = powers / powers.sum(-1, keepdim=True).clamp_min(1e-12)
    delays = payload["path_delays_ns"] * mask
    angle_features = []
    for key in ("aoa_az_deg", "aod_az_deg"):
        radians = torch.deg2rad(payload[key])
        angle_features.extend([torch.sin(radians) * mask, torch.cos(radians) * mask])
    return torch.cat(
        [
            mask.sum(-1, keepdim=True).float(),
            delays,
            normalized_power,
            *angle_features,
        ],
        dim=-1,
    ).float()


def full_multipath_features(payload: dict[str, torch.Tensor]) -> torch.Tensor:
    """Ordered link-level features covering every generated path parameter."""
    mask = payload["path_mask"].bool()
    numeric_mask = mask.float()
    powers = payload["path_powers_linear"].float() * numeric_mask
    fractions = powers / powers.sum(-1, keepdim=True).clamp_min(1e-20)
    delays = payload["path_delays_ns"].float() * numeric_mask
    first = torch.where(mask, delays, torch.full_like(delays, torch.inf)).amin(-1)
    relative = (delays - first.unsqueeze(-1)).clamp_min(0.0) * numeric_mask
    values = [numeric_mask, first.unsqueeze(-1), relative, fractions]
    phase = payload["path_phases_rad"].float()
    values.extend([phase.sin() * numeric_mask, phase.cos() * numeric_mask])
    for azimuth_key, elevation_key in (
        ("aoa_az_deg", "aoa_el_deg"),
        ("aod_az_deg", "aod_el_deg"),
    ):
        azimuth = torch.deg2rad(payload[azimuth_key].float())
        values.extend(
            [
                azimuth.sin() * numeric_mask,
                azimuth.cos() * numeric_mask,
                payload[elevation_key].float() / 90.0 * numeric_mask,
            ]
        )
    return torch.cat(values, dim=-1).float()


def _active_path_features(payload: dict[str, torch.Tensor]) -> torch.Tensor:
    mask = payload["path_mask"].bool()
    powers = payload["path_powers_linear"].float() * mask
    fractions = powers / powers.sum(-1, keepdim=True).clamp_min(1e-20)
    phase = payload["path_phases_rad"].float()
    aoa_az = torch.deg2rad(payload["aoa_az_deg"].float())
    aod_az = torch.deg2rad(payload["aod_az_deg"].float())
    features = torch.stack(
        [
            payload["path_delays_ns"].float(),
            10.0 * torch.log10(fractions.clamp_min(1e-20)),
            phase.sin(),
            phase.cos(),
            aoa_az.sin(),
            aoa_az.cos(),
            payload["aoa_el_deg"].float() / 90.0,
            aod_az.sin(),
            aod_az.cos(),
            payload["aod_el_deg"].float() / 90.0,
        ],
        dim=-1,
    )
    return features[mask]


def _correlation_matrix(values: torch.Tensor) -> torch.Tensor:
    if values.shape[0] < 2:
        return torch.zeros(values.shape[-1], values.shape[-1])
    standardized = (values - values.mean(0, keepdim=True)) / values.std(
        0, unbiased=False, keepdim=True
    ).clamp_min(1e-6)
    return torch.nan_to_num(torch.corrcoef(standardized.T))


def _scalar_wasserstein(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    count = min(first.numel(), second.numel())
    quantiles = torch.linspace(0, 1, count, device=first.device)
    return (
        torch.quantile(first.float(), quantiles)
        - torch.quantile(second.float().to(first.device), quantiles)
    ).abs().mean()


def marginal_path_metrics(
    generated: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]
) -> dict[str, float]:
    generated_mask = generated["path_mask"].bool()
    reference_mask = reference["path_mask"].bool()
    generated_power = generated["path_powers_linear"].float() * generated_mask
    reference_power = reference["path_powers_linear"].float() * reference_mask
    generated_fraction = generated_power / generated_power.sum(-1, keepdim=True).clamp_min(1e-20)
    reference_fraction = reference_power / reference_power.sum(-1, keepdim=True).clamp_min(1e-20)
    metrics: dict[str, float] = {}
    scalar_values = {
        "delay_ns": (
            generated["path_delays_ns"][generated_mask],
            reference["path_delays_ns"][reference_mask],
        ),
        "power_fraction": (
            generated_fraction[generated_mask], reference_fraction[reference_mask]
        ),
        "aoa_elevation_deg": (
            generated["aoa_el_deg"][generated_mask], reference["aoa_el_deg"][reference_mask]
        ),
        "aod_elevation_deg": (
            generated["aod_el_deg"][generated_mask], reference["aod_el_deg"][reference_mask]
        ),
    }
    for name, (first, second) in scalar_values.items():
        metrics[f"{name}_wasserstein"] = float(_scalar_wasserstein(first, second))
    for name, key, radians in (
        ("phase", "path_phases_rad", True),
        ("aoa_azimuth", "aoa_az_deg", False),
        ("aod_azimuth", "aod_az_deg", False),
    ):
        first = generated[key][generated_mask].float()
        second = reference[key][reference_mask].float()
        if not radians:
            first = torch.deg2rad(first)
            second = torch.deg2rad(second)
        first_circular = torch.stack([first.sin(), first.cos()], dim=-1)
        second_circular = torch.stack([second.sin(), second.cos()], dim=-1)
        metrics[f"{name}_circular_sliced_wasserstein"] = float(
            sliced_wasserstein(first_circular, second_circular)
        )
    return metrics


def rbf_mmd(
    generated: torch.Tensor,
    reference: torch.Tensor,
    bandwidths: list[float],
) -> torch.Tensor:
    generated = generated.float()
    reference = reference.float()
    xx = torch.cdist(generated, generated).square()
    yy = torch.cdist(reference, reference).square()
    xy = torch.cdist(generated, reference).square()
    values = []
    for bandwidth in bandwidths:
        denominator = 2.0 * bandwidth**2
        values.append(
            torch.exp(-xx / denominator).mean()
            + torch.exp(-yy / denominator).mean()
            - 2.0 * torch.exp(-xy / denominator).mean()
        )
    return torch.stack(values).mean()


def sliced_wasserstein(
    generated: torch.Tensor,
    reference: torch.Tensor,
    projections: int = 128,
    seed: int = 0,
) -> torch.Tensor:
    dimension = generated.shape[-1]
    random = torch.Generator(device=generated.device).manual_seed(seed)
    directions = torch.randn(
        projections, dimension, generator=random, device=generated.device
    )
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    generated_projected = generated @ directions.T
    reference_projected = reference @ directions.T
    count = min(generated_projected.shape[0], reference_projected.shape[0])
    generated_quantiles = torch.quantile(
        generated_projected, torch.linspace(0, 1, count, device=generated.device), dim=0
    )
    reference_quantiles = torch.quantile(
        reference_projected, torch.linspace(0, 1, count, device=reference.device), dim=0
    )
    return (generated_quantiles - reference_quantiles).abs().mean()


def correlation_curve(csi: torch.Tensor, dimension: str) -> torch.Tensor:
    if dimension == "frequency":
        samples = csi.flatten(1, 2)
    elif dimension == "space":
        samples = csi.permute(0, 3, 1, 2).flatten(2)
    else:
        raise ValueError("dimension must be 'frequency' or 'space'")
    samples = samples / samples.abs().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
    length = samples.shape[-1]
    return torch.stack(
        [
            (samples[..., : length - lag] * samples[..., lag:].conj()).mean().abs()
            for lag in range(length)
        ]
    )


def distribution_metrics(
    generated: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    bandwidths: list[float],
) -> dict[str, float]:
    generated_features = multipath_features(generated)
    reference_features = multipath_features(reference)
    feature_mean = reference_features.mean(dim=0, keepdim=True)
    feature_std = reference_features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    generated_features = (generated_features - feature_mean) / feature_std
    reference_features = (reference_features - feature_mean) / feature_std
    generated_full = full_multipath_features(generated)
    reference_full = full_multipath_features(reference)
    full_mean = reference_full.mean(dim=0, keepdim=True)
    full_std = reference_full.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    generated_full = (generated_full - full_mean) / full_std
    reference_full = (reference_full - full_mean) / full_std
    generated_active = _active_path_features(generated)
    reference_active = _active_path_features(reference)
    frequency_generated = correlation_curve(generated["csi"], "frequency")
    frequency_reference = correlation_curve(reference["csi"], "frequency")
    spatial_generated = correlation_curve(generated["csi"], "space")
    spatial_reference = correlation_curve(reference["csi"], "space")
    return {
        "mmd_rbf": float(rbf_mmd(generated_features, reference_features, bandwidths)),
        "sliced_wasserstein": float(
            sliced_wasserstein(generated_features, reference_features)
        ),
        "full_joint_mmd_rbf": float(
            rbf_mmd(generated_full, reference_full, bandwidths)
        ),
        "full_joint_sliced_wasserstein": float(
            sliced_wasserstein(generated_full, reference_full)
        ),
        "active_path_correlation_mae": float(
            (
                _correlation_matrix(generated_active)
                - _correlation_matrix(reference_active)
            ).abs().mean()
        ),
        "frequency_correlation_mae": float(
            (frequency_generated - frequency_reference).abs().mean()
        ),
        "spatial_correlation_mae": float(
            (spatial_generated - spatial_reference).abs().mean()
        ),
        **marginal_path_metrics(generated, reference),
    }

