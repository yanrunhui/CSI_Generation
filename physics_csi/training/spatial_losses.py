from __future__ import annotations

import torch


def _validate_csi(csi: torch.Tensor, tx_array_shape: tuple[int, int]) -> None:
    if csi.ndim != 4 or not csi.is_complex():
        raise ValueError("CSI must be complex with shape [batch, rx, tx, subcarrier]")
    if csi.shape[2] != tx_array_shape[0] * tx_array_shape[1]:
        raise ValueError("CSI Tx dimension does not match tx_array_shape")


def normalized_beamspace_power(
    csi: torch.Tensor,
    tx_array_shape: tuple[int, int],
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Return a per-link unit-mass 2-D DFT beam-power map."""
    _validate_csi(csi, tx_array_shape)
    rows, columns = tx_array_shape
    grid = csi.reshape(csi.shape[0], csi.shape[1], rows, columns, csi.shape[3])
    beam = torch.fft.fft2(grid, dim=(2, 3), norm="ortho")
    power = beam.abs().square().mean(dim=(1, 4))
    return power / power.sum(dim=(1, 2), keepdim=True).clamp_min(epsilon)


def normalized_tx_covariance(
    csi: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Return the phase-sensitive Tx covariance with unit trace per link."""
    if csi.ndim != 4 or not csi.is_complex():
        raise ValueError("CSI must be complex with shape [batch, rx, tx, subcarrier]")
    # [B, Tx, Rx*F], so covariance preserves the complex antenna relationship.
    samples = csi.permute(0, 2, 1, 3).reshape(csi.shape[0], csi.shape[2], -1)
    covariance = samples @ samples.conj().transpose(-1, -2)
    covariance = covariance / max(samples.shape[-1], 1)
    trace = covariance.diagonal(dim1=-2, dim2=-1).real.sum(-1)
    return covariance / trace.clamp_min(epsilon).reshape(-1, 1, 1)


def paired_spatial_losses(
    predicted_csi: torch.Tensor,
    target_csi: torch.Tensor,
    tx_array_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Beam total-variation and relative complex-covariance losses.

    These losses are intentionally paired with the clean sample used to create
    the noisy diffusion input.  They are not used to match an arbitrary sample
    from the same condition, which would encourage conditional mean collapse.
    """
    _validate_csi(predicted_csi, tx_array_shape)
    _validate_csi(target_csi, tx_array_shape)
    if predicted_csi.shape != target_csi.shape:
        raise ValueError("predicted and target CSI must have identical shapes")
    predicted_beam = normalized_beamspace_power(predicted_csi, tx_array_shape)
    target_beam = normalized_beamspace_power(target_csi, tx_array_shape)
    beamspace = 0.5 * (predicted_beam - target_beam).abs().flatten(1).sum(-1)

    predicted_covariance = normalized_tx_covariance(predicted_csi)
    target_covariance = normalized_tx_covariance(target_csi)
    difference = predicted_covariance - target_covariance
    # Express |z|^2 directly through real/imaginary components.  This avoids
    # routing the backward pass through complex abs at z == 0 while remaining
    # exactly equivalent to the squared complex Frobenius norm.
    numerator = (
        difference.real.square() + difference.imag.square()
    ).flatten(1).sum(-1)
    denominator = (
        target_covariance.real.square() + target_covariance.imag.square()
    ).flatten(1).sum(-1).clamp_min(1e-12)
    spatial = numerator / denominator
    return beamspace.mean(), spatial.mean()
