from __future__ import annotations

import torch

from physics_csi.models.diffusion.angle_delay_transform import AngleDelayTransform
from physics_csi.models.diffusion.condition_encoder import ConditionEncoder
from physics_csi.models.diffusion.multipath_representation import MultipathRepresentation
from physics_csi.physics.synthesizer import SystemConfig


def _statistics() -> dict[str, object]:
    return {
        "condition_mean": [-80.0, 50.0, 3.0, 0.5],
        "condition_std": [10.0, 20.0, 5.0, 0.5],
        "multipath": {
            "first_delay_log_ns_mean": 3.0,
            "first_delay_log_ns_std": 1.0,
            "relative_delay_log_ns_mean": 2.0,
            "relative_delay_log_ns_std": 1.0,
            "relative_delay_shape_log_mean": 0.5,
            "relative_delay_shape_log_std": 0.5,
            "log_power_fraction_db_mean": -6.0,
            "log_power_fraction_db_std": 4.0,
        },
        "angle_delay_scale": 0.25,
    }


def _path_batch() -> dict[str, torch.Tensor]:
    mask = torch.tensor([[True, True, False], [True, True, True]])
    return {
        "path_mask": mask,
        "path_delays_ns": torch.tensor([[10.0, 30.0, 0.0], [4.0, 12.0, 45.0]]),
        "path_powers_linear": torch.tensor([[0.8, 0.2, 0.0], [0.6, 0.3, 0.1]]),
        "path_phases_rad": torch.tensor([[3.13, -3.13, 0.0], [0.0, 1.0, -2.0]]),
        "aoa_az_deg": torch.tensor([[179.0, -179.0, 0.0], [20.0, -30.0, 80.0]]),
        "aoa_el_deg": torch.tensor([[5.0, 10.0, 0.0], [0.0, -10.0, 30.0]]),
        "aod_az_deg": torch.tensor([[90.0, -90.0, 0.0], [45.0, 30.0, -80.0]]),
        "aod_el_deg": torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]]),
        "direct_path_mask": torch.tensor([[True, False, False], [True, False, False]]),
    }


def test_angle_delay_forward_inverse_and_fft_normalization() -> None:
    system = SystemConfig(num_subcarriers=8, tx_array_shape=(2, 2))
    transform = AngleDelayTransform(system, scale=0.25)
    real = torch.randn(3, 1, 4, 8)
    imaginary = torch.randn(3, 1, 4, 8)
    csi = torch.complex(real, imaginary)
    representation = transform(csi)
    reconstructed = transform.inverse(representation)
    assert representation.shape == (3, 2, 2, 2, 8)
    assert torch.isfinite(representation).all()
    assert torch.allclose(reconstructed, csi, atol=2e-6, rtol=2e-6)
    original_energy = csi.abs().square().sum()
    represented_energy = (representation * 0.25).square().sum()
    assert torch.allclose(original_energy, represented_energy, atol=2e-5, rtol=2e-5)


def test_delay_and_circular_representations() -> None:
    codec = MultipathRepresentation(_statistics()["multipath"], max_paths=3)  # type: ignore[arg-type]
    batch = _path_batch()
    encoded, feature_mask = codec.encode(batch)
    decoded = codec.decode(encoded, batch["path_mask"])
    assert feature_mask.shape == encoded.shape
    assert torch.all(encoded[~feature_mask] == 0)
    assert torch.allclose(decoded["path_delays_ns"], batch["path_delays_ns"], atol=1e-5)
    assert torch.allclose(decoded["path_phases_rad"][:, :2], batch["path_phases_rad"][:, :2], atol=1e-5)
    assert torch.allclose(decoded["aoa_az_deg"][:, :2], batch["aoa_az_deg"][:, :2], atol=1e-4)


def test_structured_delay_representation_has_unit_weighted_rms() -> None:
    codec = MultipathRepresentation(
        _statistics()["multipath"],  # type: ignore[arg-type]
        max_paths=3,
        delay_mode="rms_normalized_shape",
    )
    batch = _path_batch()
    encoded, _ = codec.encode(batch)
    decoded = codec.decode(encoded, batch["path_mask"])
    shape = decoded["relative_delay_shape"]
    mask = batch["path_mask"]
    power = batch["path_powers_linear"] * mask
    fraction = power / power.sum(-1, keepdim=True)
    mean = (fraction * shape).sum(-1, keepdim=True)
    rms = (fraction * (shape - mean).square()).sum(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)
    assert torch.all(shape[:, 0] == 0)


def test_condition_mask_changes_embedding_and_null_branch() -> None:
    encoder = ConditionEncoder([-80.0, 50.0, 3.0, 0.5], [10.0, 20.0, 5.0, 0.5], 16, 32)
    values = torch.tensor([[-80.0, 50.0, 3.0, 1.0], [-80.0, 50.0, 999.0, 1.0]])
    validity = torch.tensor([[True, True, False, True], [True, True, False, True]])
    embedding = encoder(values, validity)
    assert torch.allclose(embedding[0], embedding[1])
    dropped = encoder(values, validity, drop_mask=torch.ones(2, dtype=torch.bool))
    assert torch.allclose(dropped, encoder.unconditional(2, values.device))

