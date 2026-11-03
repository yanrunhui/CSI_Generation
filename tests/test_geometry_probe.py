from __future__ import annotations

import math

import numpy as np
import torch

from physics_csi.data.geometry import (
    extract_map_geometry_features,
    geometry_feature_names,
)
from physics_csi.evaluation.geometry_probe import (
    GeometryProbeRegressor,
    ResidualGeometryProbe,
    geometry_variant_indices,
)
from physics_csi.evaluation.research_diagnostics import power_aod_joint_features


def test_geometry_features_for_axis_aligned_square() -> None:
    polygons = [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]
    features, summary = extract_map_geometry_features(
        polygons, map_size=20.0, orientation_bins=4, layout_grid_size=2
    )
    names = geometry_feature_names(orientation_bins=4, layout_grid_size=2)
    values = dict(zip(names, features.tolist()))
    assert features.shape == (len(names),)
    assert summary["valid_buildings"] == 1
    assert math.isclose(values["summed_footprint_fraction"], 0.25)
    assert math.isclose(values["building_compactness_mean"], math.pi / 4, rel_tol=1e-6)
    orientation = np.asarray(
        [values[f"edge_orientation_length_fraction_bin_{index}"] for index in range(4)]
    )
    np.testing.assert_allclose(orientation, [0.5, 0.0, 0.5, 0.0], atol=1e-7)
    layout = np.asarray(
        [
            values[f"area_weighted_centroid_layout_row_{row}_col_{column}"]
            for row in range(2)
            for column in range(2)
        ]
    )
    np.testing.assert_allclose(layout, [1.0, 0.0, 0.0, 0.0])


def test_power_aod_joint_features_single_path() -> None:
    payload = {
        "path_mask": torch.tensor([[True, False]]),
        "path_powers_linear": torch.tensor([[3.0, 0.0]]),
        "aod_az_deg": torch.tensor([[0.0, 0.0]]),
        "aod_el_deg": torch.tensor([[0.0, 0.0]]),
    }
    features = power_aod_joint_features(payload)
    expected = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]])
    torch.testing.assert_close(features, expected)


def test_geometry_probe_accepts_unseen_map_index() -> None:
    model = GeometryProbeRegressor(
        condition_dim=5,
        geometry_dim=8,
        target_dim=3,
        map_count=2,
        mode="geometry",
        hidden_dim=16,
        context_dim=4,
        geometry_hidden_dim=8,
    )
    result = model(torch.zeros(2, 5), torch.full((2,), -1), torch.zeros(2, 8))
    assert result.shape == (2, 3)


def test_residual_probe_starts_exactly_at_frozen_p0() -> None:
    base = GeometryProbeRegressor(
        condition_dim=5,
        geometry_dim=8,
        target_dim=3,
        map_count=2,
        mode="physics",
        hidden_dim=16,
        context_dim=4,
        geometry_hidden_dim=8,
    )
    residual = ResidualGeometryProbe(
        base,
        condition_dim=5,
        geometry_dim=6,
        target_dim=3,
        context_dim=4,
        geometry_hidden_dim=8,
        residual_hidden_dim=12,
        geometry_dropout=0.2,
        maximum_gate=0.5,
        initial_gate=0.05,
    ).eval()
    conditions = torch.randn(4, 5)
    map_index = torch.full((4,), -1)
    geometry = torch.randn(4, 6)
    expected = base(conditions, map_index, geometry)
    actual, correction = residual(conditions, map_index, geometry)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(correction, torch.zeros_like(correction))
    assert all(not parameter.requires_grad for parameter in residual.base_model.parameters())


def test_geometry_variants_are_nested() -> None:
    names = [
        "summed_footprint_fraction",
        "building_area_log1p_mean",
        "edge_orientation_length_fraction_bin_0",
        "area_weighted_centroid_layout_row_0_col_0",
    ]
    g1 = geometry_variant_indices(names, "G1")
    g2 = geometry_variant_indices(names, "G2")
    g3 = geometry_variant_indices(names, "G3")
    assert g1 == [0, 1]
    assert g2 == [0, 1, 2]
    assert g3 == [0, 1, 2, 3]
