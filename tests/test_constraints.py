import torch

from physics_csi.physics.constraints import PhysicsConstraintLayer, check_condition_feasibility


def test_power_delay_and_k_constraints_are_satisfied():
    layer = PhysicsConstraintLayer()
    delays = torch.tensor([[0.0, 10.0, 30.0]])
    weights = torch.tensor([[2.0, 1.0, 1.0]])
    mask = torch.tensor([[True, True, True]])
    target = torch.tensor([[0.0, 20.0, 3.0, 1.0]])
    validity = torch.ones_like(target, dtype=torch.bool)
    result = layer(delays, weights, mask, target, target_condition_validity=validity)
    assert torch.allclose(result.powers_linear.sum(-1), torch.ones(1), atol=1e-5)
    assert torch.allclose(result.actual_conditions[:, 1], target[:, 1], atol=1e-4)
    assert torch.allclose(result.actual_conditions[:, 2], target[:, 2], atol=1e-4)
    assert result.feasible.all()


def test_impossible_single_path_delay_spread_is_rejected():
    try:
        check_condition_feasibility(
            {
                "total_power_db": -80,
                "delay_spread_ns": 50,
                "k_factor_db": 3,
                "los_status": 1,
            },
            path_count=1,
        )
    except ValueError as error:
        assert "two active paths" in str(error)
    else:
        raise AssertionError("Expected an infeasible condition error")


def test_single_path_constraint_has_finite_gradient():
    layer = PhysicsConstraintLayer()
    delays = torch.tensor([[5.0]], requires_grad=True)
    weights = torch.tensor([[1.0]], requires_grad=True)
    mask = torch.tensor([[True]])
    target = torch.tensor([[-80.0, 0.0, 0.0, 1.0]])
    validity = torch.tensor([[True, True, False, True]])

    result = layer(
        delays,
        weights,
        mask,
        target,
        target_condition_validity=validity,
    )
    loss = (
        result.delays_ns.sum()
        + result.powers_linear.sum()
        + result.actual_conditions[:, :2].sum()
    )
    loss.backward()

    assert torch.isfinite(delays.grad).all()
    assert torch.isfinite(weights.grad).all()

