from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from physics_csi.physics.arrays import array_positions_wavelengths, array_response


@dataclass(frozen=True)
class SystemConfig:
    carrier_hz: float = 3.5e9
    bandwidth_hz: float = 100e6
    num_subcarriers: int = 64
    tx_array_shape: tuple[int, int] = (8, 8)
    rx_array_shape: tuple[int, int] = (1, 1)
    tx_array_plane: str = "xy"
    rx_array_plane: str = "xy"
    tx_spacing_wavelengths: float = 0.5
    rx_spacing_wavelengths: float = 0.5
    center_subcarriers: bool = True
    normalize_array_response: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SystemConfig":
        payload = dict(values)
        # PyYAML may keep values such as ``3.5e9`` and ``100.0e6`` as
        # strings.  Normalize the public configuration boundary instead of
        # relying on a particular YAML numeric resolver.
        payload["carrier_hz"] = float(payload["carrier_hz"])
        payload["bandwidth_hz"] = float(payload["bandwidth_hz"])
        payload["num_subcarriers"] = int(payload["num_subcarriers"])
        payload["tx_array_shape"] = tuple(int(v) for v in payload["tx_array_shape"])
        payload["rx_array_shape"] = tuple(int(v) for v in payload["rx_array_shape"])
        payload["tx_spacing_wavelengths"] = float(payload["tx_spacing_wavelengths"])
        payload["rx_spacing_wavelengths"] = float(payload["rx_spacing_wavelengths"])
        return cls(**payload)

    def validate(self) -> None:
        if self.carrier_hz <= 0 or self.bandwidth_hz <= 0:
            raise ValueError("Carrier frequency and bandwidth must be positive")
        if self.num_subcarriers <= 0:
            raise ValueError("num_subcarriers must be positive")
        if min(*self.tx_array_shape, *self.rx_array_shape) <= 0:
            raise ValueError("Array dimensions must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "carrier_hz": self.carrier_hz,
            "bandwidth_hz": self.bandwidth_hz,
            "num_subcarriers": self.num_subcarriers,
            "tx_array_shape": list(self.tx_array_shape),
            "rx_array_shape": list(self.rx_array_shape),
            "tx_array_plane": self.tx_array_plane,
            "rx_array_plane": self.rx_array_plane,
            "tx_spacing_wavelengths": self.tx_spacing_wavelengths,
            "rx_spacing_wavelengths": self.rx_spacing_wavelengths,
            "center_subcarriers": self.center_subcarriers,
            "normalize_array_response": self.normalize_array_response,
        }


class MIMOOFDMSynthesizer(nn.Module):
    """Differentiable batched geometric MIMO-OFDM channel synthesis."""

    def __init__(self, config: SystemConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        tx_positions = array_positions_wavelengths(
            config.tx_array_shape,
            config.tx_spacing_wavelengths,
            config.tx_array_plane,
        )
        rx_positions = array_positions_wavelengths(
            config.rx_array_shape,
            config.rx_spacing_wavelengths,
            config.rx_array_plane,
        )
        self.register_buffer("tx_positions", tx_positions, persistent=True)
        self.register_buffer("rx_positions", rx_positions, persistent=True)
        spacing = config.bandwidth_hz / config.num_subcarriers
        indices = torch.arange(config.num_subcarriers, dtype=torch.float32)
        if config.center_subcarriers:
            indices = indices - (config.num_subcarriers - 1) / 2
        self.register_buffer("subcarrier_offsets_hz", indices * spacing, persistent=True)

    @property
    def num_tx(self) -> int:
        return self.config.tx_array_shape[0] * self.config.tx_array_shape[1]

    @property
    def num_rx(self) -> int:
        return self.config.rx_array_shape[0] * self.config.rx_array_shape[1]

    def forward(
        self,
        path_delays_ns: torch.Tensor,
        path_powers_linear: torch.Tensor,
        path_phases_rad: torch.Tensor,
        aoa_az_deg: torch.Tensor,
        aoa_el_deg: torch.Tensor,
        aod_az_deg: torch.Tensor,
        aod_el_deg: torch.Tensor,
        path_mask: torch.Tensor,
    ) -> torch.Tensor:
        tensors = (
            path_delays_ns,
            path_powers_linear,
            path_phases_rad,
            aoa_az_deg,
            aoa_el_deg,
            aod_az_deg,
            aod_el_deg,
            path_mask,
        )
        if any(tensor.ndim != 2 for tensor in tensors):
            raise ValueError("All path tensors must have shape [batch, max_paths]")
        shape = path_delays_ns.shape
        if any(tensor.shape != shape for tensor in tensors[1:]):
            raise ValueError("All path tensors must share the same shape")

        # Keep explicit complex channel construction out of AMP.  PyTorch
        # 2.5 does not implement CUDA polar for float16, and complex64 is also
        # the intended numerical contract of this synthesizer.
        with torch.autocast(device_type=path_delays_ns.device.type, enabled=False):
            delays_s = path_delays_ns.float() * 1e-9
            powers = path_powers_linear.float().clamp_min(0.0)
            phases = path_phases_rad.float()
            active = path_mask.bool()
            mask = active.to(dtype=torch.float32)
            # Invalid padded paths have exactly zero power.  Differentiating
            # sqrt(0) yields an infinite derivative, and multiplying that by
            # the zero mask can still produce NaN gradients.  Give inactive
            # paths a constant dummy power before sqrt; torch.where also cuts
            # their gradient path, while their final complex gain remains 0.
            safe_powers = torch.where(
                active,
                powers.clamp_min(1e-20),
                torch.ones_like(powers),
            )
            amplitude = safe_powers.sqrt() * mask
            complex_gain = torch.polar(amplitude, phases).to(torch.complex64)
            frequency_phase = (
                -2.0
                * torch.pi
                * delays_s.unsqueeze(-1)
                * self.subcarrier_offsets_hz.float()
            )
            frequency_response = torch.polar(
                torch.ones_like(frequency_phase), frequency_phase
            ).to(torch.complex64)
            rx_response = array_response(
                aoa_az_deg,
                aoa_el_deg,
                self.rx_positions,
                normalize=self.config.normalize_array_response,
            )
            tx_response = array_response(
                aod_az_deg,
                aod_el_deg,
                self.tx_positions,
                normalize=self.config.normalize_array_response,
            )
            channel = torch.einsum(
                "bl,blf,blr,blt->brtf",
                complex_gain,
                frequency_response,
                rx_response,
                tx_response.conj(),
            )
            return channel.to(torch.complex64)


def synthesize_mimo_ofdm(
    path_delays: torch.Tensor,
    path_powers: torch.Tensor,
    path_phases: torch.Tensor,
    aoa: tuple[torch.Tensor, torch.Tensor],
    aod: tuple[torch.Tensor, torch.Tensor],
    path_mask: torch.Tensor,
    config: SystemConfig | Mapping[str, Any],
) -> torch.Tensor:
    system = config if isinstance(config, SystemConfig) else SystemConfig.from_mapping(config)
    synthesizer = MIMOOFDMSynthesizer(system).to(path_delays.device)
    return synthesizer(
        path_delays,
        path_powers,
        path_phases,
        aoa[0],
        aoa[1],
        aod[0],
        aod[1],
        path_mask,
    )

