from __future__ import annotations

import torch

from physics_csi.models.scene_aod_residual import SceneConditionedAoDResidual
from physics_csi.training.aod_residual import AoDDistributionLoss


def _model() -> SceneConditionedAoDResidual:
    return SceneConditionedAoDResidual(
        max_paths=3,
        map_id_to_index={"10": 0, "25": 1},
        condition_mean=[-80.0, 5.0, 0.0, 1.0],
        condition_std=[10.0, 5.0, 3.0, 1.0],
        hidden_dim=32,
        map_embedding_dim=8,
        condition_hidden_dim=8,
        layers=1,
        attention_heads=4,
        dropout=0.0,
        maximum_azimuth_residual_deg=20.0,
        maximum_elevation_residual_deg=10.0,
        initial_gate=0.1,
    )


def _payload(samples: int = 8) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    angle = torch.linspace(-175.0, 175.0, samples).unsqueeze(-1)
    payload = {
        "path_mask": torch.tensor([[True, True, False]]).repeat(samples, 1),
        "path_delays_ns": torch.tensor([[0.0, 10.0, 0.0]]).repeat(samples, 1),
        "path_powers_linear": torch.tensor([[5e-9, 5e-9, 0.0]]).repeat(samples, 1),
        "direct_path_mask": torch.tensor([[True, False, False]]).repeat(samples, 1),
        "path_phases_rad": torch.zeros(samples, 3),
        "aoa_az_deg": torch.zeros(samples, 3),
        "aoa_el_deg": torch.zeros(samples, 3),
        "aod_az_deg": torch.cat([angle, -angle, torch.zeros_like(angle)], dim=-1),
        "aod_el_deg": torch.zeros(samples, 3),
    }
    conditions = torch.tensor([[-80.0, 5.0, 0.0, 1.0]]).repeat(samples, 1)
    validity = torch.ones_like(conditions, dtype=torch.bool)
    return payload, conditions, validity


def test_aod_residual_starts_at_exact_dp_identity() -> None:
    model = _model().eval()
    payload, conditions, validity = _payload()
    output = model(payload, conditions, validity, map_id=torch.full((8,), 10))
    torch.testing.assert_close(output.aod_az_deg, payload["aod_az_deg"], atol=1e-5, rtol=0)
    torch.testing.assert_close(output.aod_el_deg, payload["aod_el_deg"])
    torch.testing.assert_close(output.azimuth_residual_deg, torch.zeros(8, 3))
    torch.testing.assert_close(output.elevation_residual_deg, torch.zeros(8, 3))


def test_aod_apply_preserves_every_non_aod_path_parameter() -> None:
    model = _model().eval()
    payload, conditions, validity = _payload()
    output = model(payload, conditions, validity, map_id=torch.full((8,), 10))
    corrected = model.apply_to_payload(payload, output)
    for key in (
        "path_mask",
        "path_delays_ns",
        "path_powers_linear",
        "path_phases_rad",
        "aoa_az_deg",
        "aoa_el_deg",
    ):
        torch.testing.assert_close(corrected[key], payload[key])


def test_periodic_aod_loss_has_finite_gradient() -> None:
    model = _model().train()
    payload, conditions, validity = _payload()
    reference = dict(payload)
    reference["aod_az_deg"] = torch.remainder(payload["aod_az_deg"] + 12.0 + 180.0, 360.0) - 180.0
    correct_output = model(payload, conditions, validity, map_id=torch.full((8,), 10))
    shuffled_output = model(payload, conditions, validity, map_id=torch.full((8,), 25))
    correct = model.apply_to_payload(payload, correct_output)
    shuffled = model.apply_to_payload(payload, shuffled_output)
    loss_function = AoDDistributionLoss(
        max_paths=3,
        power_db_mean=-3.0,
        power_db_std=4.0,
        projections=8,
        weighted_quantiles=16,
        maximum_azimuth_residual_deg=20.0,
        maximum_elevation_residual_deg=10.0,
    )
    metrics = loss_function(
        correct_payload=correct,
        shuffled_payload=shuffled,
        dp_payload=payload,
        reference_payload=reference,
        correct_output=correct_output,
    )
    metrics["loss"].backward()
    assert torch.isfinite(metrics["loss"])
    assert model.azimuth_head.weight.grad is not None
    assert torch.isfinite(model.azimuth_head.weight.grad).all()
    assert model.elevation_head.weight.grad is not None
    assert torch.isfinite(model.elevation_head.weight.grad).all()
