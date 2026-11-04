from __future__ import annotations

import torch
from torch import nn

from physics_csi.models.scene_relative_aod_residual import (
    SceneConditionedRelativeAoDResidual,
)


def _model() -> SceneConditionedRelativeAoDResidual:
    return SceneConditionedRelativeAoDResidual(
        max_paths=4,
        map_id_to_index={10: 0, 25: 1},
        condition_mean=[-80.0, 20.0, 0.0, 1.0],
        condition_std=[10.0, 10.0, 3.0, 1.0],
        hidden_dim=32,
        map_embedding_dim=8,
        condition_hidden_dim=8,
        dropout=0.0,
        maximum_log_scale=0.5,
        initial_gate=0.2,
    )


def _payload(samples: int = 3):
    mask = torch.tensor([[True, True, True, False]]).repeat(samples, 1)
    payload = {
        "path_mask": mask,
        "direct_path_mask": torch.tensor(
            [[True, False, False, False]]
        ).repeat(samples, 1),
        "path_delays_ns": torch.tensor(
            [[100.0, 120.0, 165.0, 0.0]]
        ).repeat(samples, 1),
        "path_powers_linear": torch.tensor(
            [[0.55, 0.30, 0.15, 0.0]]
        ).repeat(samples, 1),
        "path_phases_rad": torch.zeros(samples, 4),
        "aoa_az_deg": torch.zeros(samples, 4),
        "aoa_el_deg": torch.zeros(samples, 4),
        "aod_az_deg": torch.tensor(
            [[25.0, 42.0, 4.0, 0.0]]
        ).repeat(samples, 1),
        "aod_el_deg": torch.tensor(
            [[8.0, 15.0, -6.0, 0.0]]
        ).repeat(samples, 1),
    }
    conditions = torch.tensor([[-80.0, 20.0, 0.0, 1.0]]).repeat(samples, 1)
    validity = torch.ones_like(conditions, dtype=torch.bool)
    return payload, conditions, validity


def _direction(azimuth: torch.Tensor, elevation: torch.Tensor) -> torch.Tensor:
    azimuth = torch.deg2rad(azimuth)
    elevation = torch.deg2rad(elevation)
    return torch.stack(
        [
            elevation.cos() * azimuth.cos(),
            elevation.cos() * azimuth.sin(),
            elevation.sin(),
        ],
        dim=-1,
    )


def _center(payload: dict[str, torch.Tensor]) -> torch.Tensor:
    mask = payload["path_mask"]
    power = payload["path_powers_linear"] * mask
    fraction = power / power.sum(-1, keepdim=True)
    direction = _direction(payload["aod_az_deg"], payload["aod_el_deg"])
    value = (fraction.unsqueeze(-1) * direction).sum(1)
    return value / value.norm(dim=-1, keepdim=True)


def test_relative_cluster_starts_as_identity() -> None:
    model = _model().eval()
    payload, conditions, validity = _payload()
    output = model(payload, conditions, validity, map_id=torch.full((3,), 10))
    torch.testing.assert_close(output.aod_az_deg, payload["aod_az_deg"], atol=2e-4, rtol=0)
    torch.testing.assert_close(output.aod_el_deg, payload["aod_el_deg"], atol=2e-4, rtol=0)


def test_relative_cluster_changes_spread_but_preserves_global_center() -> None:
    model = _model().eval()
    final = model.output_network[-1]
    assert isinstance(final, nn.Linear)
    with torch.no_grad():
        final.bias.copy_(torch.tensor([0.9, -0.7]))
    payload, conditions, validity = _payload()
    output = model(payload, conditions, validity, map_id=torch.full((3,), 10))
    corrected = model.apply_to_payload(payload, output)
    center_angle = torch.rad2deg(
        torch.acos((_center(payload) * _center(corrected)).sum(-1).clamp(-1.0, 1.0))
    )
    assert float(center_angle.detach().max()) < 0.05

    before = _direction(payload["aod_az_deg"], payload["aod_el_deg"])
    after = _direction(corrected["aod_az_deg"], corrected["aod_el_deg"])
    before_separation = torch.acos((before[:, 0] * before[:, 1]).sum(-1).clamp(-1.0, 1.0))
    after_separation = torch.acos((after[:, 0] * after[:, 1]).sum(-1).clamp(-1.0, 1.0))
    assert float(
        (after_separation - before_separation).abs().mean().detach()
    ) > 1e-4


def test_relative_cluster_preserves_every_non_aod_parameter() -> None:
    model = _model().eval()
    payload, conditions, validity = _payload()
    output = model(payload, conditions, validity, map_id=torch.full((3,), 10))
    corrected = model.apply_to_payload(payload, output)
    for key in (
        "path_mask",
        "direct_path_mask",
        "path_delays_ns",
        "path_powers_linear",
        "path_phases_rad",
        "aoa_az_deg",
        "aoa_el_deg",
    ):
        torch.testing.assert_close(corrected[key], payload[key])

