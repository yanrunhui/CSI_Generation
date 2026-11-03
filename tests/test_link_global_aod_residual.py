from __future__ import annotations

import torch
from torch import nn

from physics_csi.models.scene_link_global_aod_residual import (
    SceneConditionedLinkGlobalAoDResidual,
)


def _model() -> SceneConditionedLinkGlobalAoDResidual:
    return SceneConditionedLinkGlobalAoDResidual(
        max_paths=3,
        map_id_to_index={"10": 0, "25": 1},
        condition_mean=[-80.0, 5.0, 0.0, 1.0],
        condition_std=[10.0, 5.0, 3.0, 1.0],
        hidden_dim=32,
        map_embedding_dim=8,
        condition_hidden_dim=8,
        dropout=0.0,
        maximum_azimuth_residual_deg=20.0,
        maximum_elevation_residual_deg=10.0,
        initial_gate=0.1,
    )


def _payload(samples: int = 4) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    payload = {
        "path_mask": torch.tensor([[True, True, False]]).repeat(samples, 1),
        "path_delays_ns": torch.tensor([[0.0, 10.0, 0.0]]).repeat(samples, 1),
        "path_powers_linear": torch.tensor([[7e-9, 3e-9, 0.0]]).repeat(samples, 1),
        "direct_path_mask": torch.tensor([[True, False, False]]).repeat(samples, 1),
        "path_phases_rad": torch.zeros(samples, 3),
        "aoa_az_deg": torch.zeros(samples, 3),
        "aoa_el_deg": torch.zeros(samples, 3),
        "aod_az_deg": torch.tensor([[-170.0, 35.0, 0.0]]).repeat(samples, 1),
        "aod_el_deg": torch.tensor([[3.0, -8.0, 0.0]]).repeat(samples, 1),
    }
    conditions = torch.tensor([[-80.0, 5.0, 0.0, 1.0]]).repeat(samples, 1)
    validity = torch.ones_like(conditions, dtype=torch.bool)
    return payload, conditions, validity


def test_link_global_starts_as_exact_identity() -> None:
    model = _model().eval()
    payload, conditions, validity = _payload()
    output = model(payload, conditions, validity, map_id=torch.full((4,), 10))
    torch.testing.assert_close(output.aod_az_deg, payload["aod_az_deg"], atol=1e-5, rtol=0)
    torch.testing.assert_close(output.aod_el_deg, payload["aod_el_deg"])
    torch.testing.assert_close(output.azimuth_residual_deg, torch.zeros(4, 3))
    torch.testing.assert_close(output.elevation_residual_deg, torch.zeros(4, 3))


def test_link_global_applies_one_shared_rotation_to_all_active_paths() -> None:
    model = _model().eval()
    final = model.output_network[-1]
    assert isinstance(final, nn.Linear)
    with torch.no_grad():
        final.bias.copy_(torch.tensor([0.7, -0.4]))
    payload, conditions, validity = _payload()
    output = model(payload, conditions, validity, map_id=torch.full((4,), 10))
    mask = payload["path_mask"]
    for sample in range(mask.shape[0]):
        active_azimuth = output.azimuth_residual_deg[sample][mask[sample]]
        active_elevation = output.elevation_residual_deg[sample][mask[sample]]
        torch.testing.assert_close(active_azimuth, active_azimuth[:1].expand_as(active_azimuth))
        torch.testing.assert_close(
            active_elevation, active_elevation[:1].expand_as(active_elevation)
        )


def test_link_global_preserves_every_non_aod_parameter() -> None:
    model = _model().eval()
    payload, conditions, validity = _payload()
    output = model(payload, conditions, validity, map_id=torch.full((4,), 10))
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
