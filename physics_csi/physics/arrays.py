from __future__ import annotations

import torch


def array_positions_wavelengths(
    shape: tuple[int, int],
    spacing: float,
    plane: str = "xy",
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    rows, cols = shape
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Array dimensions must be positive, got {shape}")
    if spacing <= 0:
        raise ValueError("Array spacing must be positive")
    plane = plane.lower()
    if plane not in {"xy", "xz", "yz"}:
        raise ValueError(f"Unsupported array plane {plane!r}")
    row = torch.arange(rows, dtype=torch.float32, device=device) - (rows - 1) / 2
    col = torch.arange(cols, dtype=torch.float32, device=device) - (cols - 1) / 2
    grid_row, grid_col = torch.meshgrid(row, col, indexing="ij")
    positions = torch.zeros(rows, cols, 3, dtype=torch.float32, device=device)
    axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane]
    positions[..., axes[0]] = grid_row * spacing
    positions[..., axes[1]] = grid_col * spacing
    return positions.reshape(rows * cols, 3)


def direction_vectors(azimuth_deg: torch.Tensor, elevation_deg: torch.Tensor) -> torch.Tensor:
    azimuth = torch.deg2rad(azimuth_deg.float())
    elevation = torch.deg2rad(elevation_deg.float())
    cos_elevation = torch.cos(elevation)
    return torch.stack(
        [
            cos_elevation * torch.cos(azimuth),
            cos_elevation * torch.sin(azimuth),
            torch.sin(elevation),
        ],
        dim=-1,
    )


def array_response(
    azimuth_deg: torch.Tensor,
    elevation_deg: torch.Tensor,
    positions_wavelengths: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    # ComplexHalf support is incomplete in PyTorch/CUDA.  Array steering is a
    # physics boundary and should remain float32/complex64 even when the
    # surrounding neural network runs under AMP.
    with torch.autocast(device_type=azimuth_deg.device.type, enabled=False):
        directions = direction_vectors(azimuth_deg.float(), elevation_deg.float())
        positions = positions_wavelengths.to(device=directions.device, dtype=torch.float32)
        phase = 2.0 * torch.pi * torch.einsum("...d,nd->...n", directions, positions)
        response = torch.polar(torch.ones_like(phase), phase)
        if normalize:
            response = response / positions.shape[0] ** 0.5
        return response.to(torch.complex64)

