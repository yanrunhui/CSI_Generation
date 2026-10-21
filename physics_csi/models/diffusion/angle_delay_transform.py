from __future__ import annotations

import torch
from torch import nn

from physics_csi.physics.synthesizer import SystemConfig


class AngleDelayTransform(nn.Module):
    """Reversible Tx spatial-DFT / frequency-IFFT representation."""

    def __init__(self, system: SystemConfig, scale: float = 1.0) -> None:
        super().__init__()
        # Derive this from the dataclass fields instead of relying on the
        # convenience ``num_rx`` property.  This keeps the diffusion module
        # compatible with prepared-data installations created by older
        # project revisions whose SystemConfig did not expose that property.
        num_rx = int(system.rx_array_shape[0]) * int(system.rx_array_shape[1])
        if num_rx != 1:
            raise ValueError("angle-delay diffusion currently requires one Rx antenna")
        self.tx_rows, self.tx_cols = system.tx_array_shape
        self.num_subcarriers = system.num_subcarriers
        self.register_buffer("scale", torch.tensor(float(scale)).clamp_min(1e-12))

    @property
    def tensor_shape(self) -> tuple[int, int, int, int]:
        return (2, self.tx_rows, self.tx_cols, self.num_subcarriers)

    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        expected = (1, self.tx_rows * self.tx_cols, self.num_subcarriers)
        if csi.ndim != 4 or tuple(csi.shape[1:]) != expected:
            raise ValueError(f"expected complex CSI shape [B, {expected[0]}, {expected[1]}, {expected[2]}]")
        grid = csi[:, 0].reshape(-1, self.tx_rows, self.tx_cols, self.num_subcarriers)
        spatial = torch.fft.fftshift(
            torch.fft.fft2(grid, dim=(1, 2), norm="ortho"), dim=(1, 2)
        )
        angle_delay = torch.fft.ifft(spatial, dim=-1, norm="ortho")
        packed = torch.stack([angle_delay.real, angle_delay.imag], dim=1)
        return packed.float() / self.scale

    def inverse(self, representation: torch.Tensor) -> torch.Tensor:
        expected = self.tensor_shape
        if representation.ndim != 5 or tuple(representation.shape[1:]) != expected:
            raise ValueError(f"expected angle-delay shape [B, {expected}]")
        complex_tensor = torch.complex(representation[:, 0], representation[:, 1]) * self.scale
        spatial = torch.fft.fft(complex_tensor, dim=-1, norm="ortho")
        grid = torch.fft.ifft2(
            torch.fft.ifftshift(spatial, dim=(1, 2)), dim=(1, 2), norm="ortho"
        )
        return grid.reshape(-1, 1, self.tx_rows * self.tx_cols, self.num_subcarriers).to(torch.complex64)
