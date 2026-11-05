from __future__ import annotations

import torch

from physics_csi.physics.aod_control import (
    AoDControlProjector,
    circular_difference_degrees,
    select_aod_candidates_within_tolerance,
    select_nearest_aod_candidates,
    select_payload_samples,
)


def _payload(angles: list[list[float]], powers: list[list[float]]) -> dict[str, torch.Tensor]:
    azimuth = torch.tensor(angles, dtype=torch.float32)
    power = torch.tensor(powers, dtype=torch.float32)
    return {
        "path_mask": power > 0,
        "path_powers_linear": power,
        "aod_az_deg": azimuth,
    }


def test_mean_control_hits_target_and_preserves_relative_angles() -> None:
    payload = _payload([[20.0, 40.0, 80.0]], [[0.6, 0.3, 0.1]])
    projector = AoDControlProjector(minimum_mean_concentration=0.0)
    output = projector(payload, 60.0, mode="mean")
    assert float(output.circular_error_deg.max()) < 1e-4
    before = circular_difference_degrees(
        payload["aod_az_deg"][:, :, None], payload["aod_az_deg"][:, None, :]
    )
    after = circular_difference_degrees(
        output.aod_az_deg[:, :, None], output.aod_az_deg[:, None, :]
    )
    assert torch.allclose(before, after, atol=1e-4, rtol=0.0)


def test_strongest_control_uses_first_argmax_and_reports_tie() -> None:
    payload = _payload([[20.0, 40.0, 80.0]], [[0.45, 0.45, 0.10]])
    output = AoDControlProjector(strongest_near_tie_margin=0.01)(
        payload, 60.0, mode="strongest"
    )
    assert int(output.strongest_path_index.item()) == 0
    assert bool(output.strongest_near_tie.item())
    assert torch.allclose(output.aod_az_deg, torch.tensor([[60.0, 80.0, 120.0]]))
    assert float(output.circular_error_deg.max()) < 1e-4


def test_mean_control_marks_symmetric_cluster_unstable() -> None:
    payload = _payload([[-90.0, 90.0]], [[0.5, 0.5]])
    output = AoDControlProjector(minimum_mean_concentration=0.1)(
        payload, 30.0, mode="mean"
    )
    assert not bool(output.control_valid.item())
    assert float(output.mean_resultant_concentration.item()) < 1e-5


def test_wrap_boundary_control() -> None:
    payload = _payload([[179.0, -170.0]], [[0.8, 0.2]])
    output = AoDControlProjector(minimum_mean_concentration=0.0)(
        payload, -179.0, mode="mean"
    )
    assert float(output.circular_error_deg.max()) < 1e-4
    assert bool(((output.aod_az_deg >= -180.0) & (output.aod_az_deg <= 180.0)).all())


def test_nearest_candidate_selection_is_unique_and_small_rotation() -> None:
    payload = _payload([[10.0], [65.0], [87.0], [-170.0]], [[1.0]] * 4)
    projector = AoDControlProjector(minimum_mean_concentration=0.0)
    selection = select_nearest_aod_candidates(
        payload,
        90.0,
        samples=2,
        mode="mean",
        projector=projector,
    )
    assert selection.indices.tolist() == [2, 1]
    assert torch.allclose(selection.angular_distance_deg, torch.tensor([3.0, 25.0]))
    assert int(torch.unique(selection.indices).numel()) == 2
    selected = select_payload_samples(payload, selection.indices)
    assert torch.allclose(selected["aod_az_deg"], torch.tensor([[87.0], [65.0]]))


def test_nearest_candidate_selection_can_exclude_strongest_ties() -> None:
    payload = _payload(
        [[89.0, 20.0], [88.0, -40.0], [87.0, 10.0]],
        [[0.5, 0.5], [0.7, 0.3], [0.8, 0.2]],
    )
    projector = AoDControlProjector(strongest_near_tie_margin=0.01)
    selection = select_nearest_aod_candidates(
        payload,
        90.0,
        samples=2,
        mode="strongest",
        projector=projector,
        exclude_strongest_near_ties=True,
    )
    assert selection.indices.tolist() == [1, 2]
    assert selection.eligible_samples == 2


def test_tolerance_selection_preserves_generation_order_without_projection() -> None:
    payload = _payload([[86.0], [91.0], [84.0], [94.0]], [[1.0]] * 4)
    projector = AoDControlProjector(minimum_mean_concentration=0.0)
    selection = select_aod_candidates_within_tolerance(
        payload,
        90.0,
        samples=3,
        tolerance_deg=5.0,
        mode="mean",
        projector=projector,
    )
    assert selection.indices.tolist() == [0, 1, 3]
    assert torch.all(selection.angular_distance_deg <= 5.0)
    assert selection.eligible_samples == 3
