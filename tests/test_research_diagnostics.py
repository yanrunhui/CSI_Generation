from __future__ import annotations

import torch

from physics_csi.evaluation.research_diagnostics import (
    aod_statistical_features,
    condition_inputs,
    direction_vectors,
    spatial_link_features,
)


def test_direction_vectors_are_continuous_across_azimuth_wrap() -> None:
    first = direction_vectors(torch.tensor([179.999]), torch.tensor([0.0]))
    second = direction_vectors(torch.tensor([-180.001]), torch.tensor([0.0]))
    torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)


def test_single_path_aod_features_have_unit_concentration_and_zero_spread() -> None:
    payload = {
        "path_mask": torch.tensor([[True, False]]),
        "path_powers_linear": torch.tensor([[2.0, 0.0]]),
        "aod_az_deg": torch.tensor([[30.0, 0.0]]),
        "aod_el_deg": torch.tensor([[10.0, 0.0]]),
    }
    features = aod_statistical_features(payload)
    torch.testing.assert_close(features[:, 6:8], torch.ones(1, 2))
    torch.testing.assert_close(features[:, 8], torch.zeros(1), atol=1e-2, rtol=0.0)


def test_spatial_link_features_preserve_complex_neighbor_correlation() -> None:
    magnitude = torch.ones(2, 1, 1, 4)
    common = torch.polar(magnitude, torch.full_like(magnitude, 0.3)).to(torch.complex64)
    csi = common.expand(2, 1, 4, 4).clone()
    features, names = spatial_link_features(csi, (2, 2), offsets=((0, 1), (1, 0)))
    assert names == (
        "tx_corr_drow_0_dcol_1_real",
        "tx_corr_drow_0_dcol_1_imag",
        "tx_corr_drow_1_dcol_0_real",
        "tx_corr_drow_1_dcol_0_imag",
    )
    torch.testing.assert_close(features[:, (0, 2)], torch.ones(2, 2))
    torch.testing.assert_close(features[:, (1, 3)], torch.zeros(2, 2), atol=1e-6, rtol=0.0)


def test_condition_inputs_zero_invalid_k_factor() -> None:
    conditions = torch.tensor([[-90.0, 100.0, 999.0, 0.0]])
    validity = torch.tensor([[True, True, False, True]])
    result = condition_inputs(
        conditions,
        validity,
        mean=torch.tensor([-80.0, 50.0, 5.0, 0.5]),
        std=torch.tensor([10.0, 25.0, 2.0, 0.5]),
    )
    assert result.shape == (1, 5)
    assert float(result[0, 2]) == 0.0
    assert float(result[0, 3]) == 0.0
    assert float(result[0, 4]) == 0.0
