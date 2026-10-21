import torch

from physics_csi.data.physical_labels import (
    compute_physical_conditions,
    rms_delay_spread_ns,
)


def test_physical_conditions_use_incoherent_power_and_weighted_delay_spread():
    powers = torch.tensor([[3.0, 1.0]])
    delays = torch.tensor([[0.0, 20.0]])
    mask = torch.tensor([[True, True]])
    direct = torch.tensor([[True, False]])
    conditions, validity = compute_physical_conditions(powers, delays, mask, direct)
    assert torch.allclose(conditions[:, 0], torch.tensor([10.0 * torch.log10(torch.tensor(4.0))]))
    assert torch.allclose(conditions[:, 1], torch.tensor([8.660254]), atol=1e-5)
    assert torch.allclose(conditions[:, 2], torch.tensor([10.0 * torch.log10(torch.tensor(3.0))]))
    assert conditions[0, 3] == 1
    assert validity.all()


def test_single_path_delay_spread_is_zero_with_finite_gradient():
    delays = torch.tensor([[7.0]], requires_grad=True)
    powers = torch.tensor([[1.0]], requires_grad=True)
    spread = rms_delay_spread_ns(delays, powers, torch.tensor([[True]]))

    assert torch.equal(spread, torch.zeros_like(spread))
    spread.sum().backward()
    assert torch.isfinite(delays.grad).all()
    assert torch.isfinite(powers.grad).all()


def test_nlos_k_factor_is_marked_invalid():
    powers = torch.tensor([[1.0, 0.5]])
    delays = torch.tensor([[0.0, 10.0]])
    mask = torch.tensor([[True, True]])
    direct = torch.tensor([[False, False]])
    conditions, validity = compute_physical_conditions(powers, delays, mask, direct)
    assert conditions[0, 3] == 0
    assert not validity[0, 2]

