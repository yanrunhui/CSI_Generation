import torch

from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig


def test_system_config_normalizes_numeric_strings():
    config = SystemConfig.from_mapping(
        {
            "carrier_hz": "3.5e9",
            "bandwidth_hz": "100.0e6",
            "num_subcarriers": "64",
            "tx_array_shape": ["8", "8"],
            "rx_array_shape": ["1", "1"],
            "tx_array_plane": "xy",
            "rx_array_plane": "xy",
            "tx_spacing_wavelengths": "0.5",
            "rx_spacing_wavelengths": "0.5",
            "center_subcarriers": True,
            "normalize_array_response": True,
        }
    )

    assert config.carrier_hz == 3.5e9
    assert config.bandwidth_hz == 100.0e6
    assert config.num_subcarriers == 64
    assert config.tx_array_shape == (8, 8)
    assert config.tx_spacing_wavelengths == 0.5


def test_synthesizer_shape_dtype_and_gradient():
    synthesizer = MIMOOFDMSynthesizer(
        SystemConfig(num_subcarriers=64, tx_array_shape=(8, 8), rx_array_shape=(1, 1))
    )
    shape = (2, 3)
    delays = torch.tensor([[0.0, 20.0, 50.0], [5.0, 30.0, 0.0]], requires_grad=True)
    powers = torch.full(shape, 1 / 3, requires_grad=True)
    phases = torch.zeros(shape, requires_grad=True)
    angles = torch.zeros(shape)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    channel = synthesizer(delays, powers, phases, angles, angles, angles, angles, mask)
    assert channel.shape == (2, 1, 64, 64)
    assert channel.dtype == torch.complex64
    channel.abs().square().mean().backward()
    assert delays.grad is not None
    assert powers.grad is not None
    assert phases.grad is not None


def test_single_path_has_constant_frequency_magnitude():
    synthesizer = MIMOOFDMSynthesizer(SystemConfig(num_subcarriers=16))
    one = torch.ones(1, 1)
    zero = torch.zeros(1, 1)
    channel = synthesizer(zero, one, zero, zero, zero, zero, zero, one.bool())
    magnitude = channel.abs()
    assert torch.allclose(magnitude, magnitude[..., :1].expand_as(magnitude), atol=1e-6)


def test_synthesizer_keeps_complex64_inside_autocast():
    synthesizer = MIMOOFDMSynthesizer(SystemConfig(num_subcarriers=8))
    one = torch.ones(1, 1)
    zero = torch.zeros(1, 1)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        channel = synthesizer(zero, one, zero, zero, zero, zero, zero, one.bool())
    assert channel.dtype == torch.complex64


def test_masked_zero_power_has_finite_gradient():
    synthesizer = MIMOOFDMSynthesizer(SystemConfig(num_subcarriers=8))
    delays = torch.tensor([[0.0, 10.0]], requires_grad=True)
    powers = torch.tensor([[1.0, 0.0]], requires_grad=True)
    phases = torch.zeros(1, 2, requires_grad=True)
    angles = torch.zeros(1, 2)
    mask = torch.tensor([[True, False]])
    channel = synthesizer(
        delays, powers, phases, angles, angles, angles, angles, mask
    )
    channel.abs().square().mean().backward()
    assert torch.isfinite(delays.grad).all()
    assert torch.isfinite(powers.grad).all()
    assert torch.isfinite(phases.grad).all()

