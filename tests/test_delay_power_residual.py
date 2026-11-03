from __future__ import annotations

import torch

from physics_csi.data.delay_power_residual import (
    ConditionNeighborhoodBatchSampler,
    DelayPowerResidualCacheDataset,
)
from physics_csi.models.scene_delay_power_residual import SceneConditionedDelayPowerResidual
from physics_csi.training.delay_power_residual import DelayPowerDistributionLoss


def _model() -> SceneConditionedDelayPowerResidual:
    return SceneConditionedDelayPowerResidual(
        max_paths=3,
        map_id_to_index={"10": 0, "25": 1},
        condition_mean=[-80.0, 5.0, 0.0, 1.0],
        condition_std=[10.0, 5.0, 3.0, 1.0],
        constraints={
            "enforce_total_power": True,
            "enforce_delay_spread": True,
            "enforce_k_factor": True,
            "enforce_los_direct_path": True,
            "epsilon": 1e-20,
        },
        hidden_dim=32,
        map_embedding_dim=8,
        condition_hidden_dim=8,
        layers=1,
        attention_heads=4,
        dropout=0.0,
    )


def _batch(samples: int = 4) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    conditions = torch.tensor([[-80.0, 5.0, 0.0, 1.0]]).repeat(samples, 1)
    validity = torch.ones_like(conditions, dtype=torch.bool)
    base = {
        "path_mask": torch.tensor([[True, True, False]]).repeat(samples, 1),
        "path_delays_ns": torch.tensor([[0.0, 10.0, 0.0]]).repeat(samples, 1),
        "path_powers_linear": torch.tensor([[5e-9, 5e-9, 0.0]]).repeat(samples, 1),
        "direct_path_mask": torch.tensor([[True, False, False]]).repeat(samples, 1),
        "path_phases_rad": torch.zeros(samples, 3),
        "aoa_az_deg": torch.zeros(samples, 3),
        "aoa_el_deg": torch.zeros(samples, 3),
        "aod_az_deg": torch.zeros(samples, 3),
        "aod_el_deg": torch.zeros(samples, 3),
    }
    return base, conditions, validity


def test_dp_residual_starts_at_projected_b3_identity() -> None:
    model = _model().eval()
    base, conditions, validity = _batch()
    output = model(base, conditions, validity, map_id=torch.full((4,), 10))
    torch.testing.assert_close(output.delay_log_gap_residual, torch.zeros(4, 3))
    torch.testing.assert_close(output.power_logit_residual, torch.zeros(4, 3))
    torch.testing.assert_close(output.constrained.delays_ns, base["path_delays_ns"])
    torch.testing.assert_close(output.constrained.powers_linear, base["path_powers_linear"])


def test_dp_distribution_loss_backpropagates_without_sample_pairing() -> None:
    model = _model().train()
    base, conditions, validity = _batch(8)
    # Reference order is intentionally permuted. Distribution losses must not
    # depend on a row-to-row correspondence.
    reference = {key: value.flip(0) for key, value in base.items()}
    correct_output = model(base, conditions, validity, map_id=torch.full((8,), 10))
    shuffled_output = model(base, conditions, validity, map_id=torch.full((8,), 25))
    correct = model.apply_to_payload(base, correct_output)
    shuffled = model.apply_to_payload(base, shuffled_output)
    loss_function = DelayPowerDistributionLoss(
        max_paths=3,
        power_db_mean=-3.0,
        power_db_std=4.0,
        projections=8,
        weighted_delay_bins=16,
    )
    metrics = loss_function(
        correct_payload=correct,
        shuffled_payload=shuffled,
        base_payload=base,
        reference_payload=reference,
        conditions=conditions,
        correct_output=correct_output,
    )
    metrics["loss"].backward()
    assert torch.isfinite(metrics["loss"])
    assert model.delay_head.weight.grad is not None
    assert torch.isfinite(model.delay_head.weight.grad).all()
    assert model.power_head.weight.grad is not None
    assert torch.isfinite(model.power_head.weight.grad).all()


def test_condition_neighborhood_batches_do_not_mix_maps_or_path_counts() -> None:
    samples = 16
    mask = torch.tensor([[True, True, False]]).repeat(samples, 1)
    reference = {
        "path_mask": mask,
        "path_delays_ns": torch.tensor([[0.0, 10.0, 0.0]]).repeat(samples, 1),
        "path_powers_linear": torch.tensor([[0.5, 0.5, 0.0]]).repeat(samples, 1),
        "conditions": torch.tensor([[-80.0, 5.0, 0.0, 1.0]]).repeat(samples, 1),
        "condition_validity": torch.ones(samples, 4, dtype=torch.bool),
        "map_id": torch.tensor([10] * 8 + [25] * 8),
    }
    base = {
        "path_mask": mask.clone(),
        "path_delays_ns": reference["path_delays_ns"].clone(),
        "path_powers_linear": reference["path_powers_linear"].clone(),
    }
    cache = {
        "format_version": 1,
        "kind": "b3_dp_residual_cache",
        "splits": {"train": {"reference": reference, "base": base}},
    }
    dataset = DelayPowerResidualCacheDataset(cache, "train")
    sampler = ConditionNeighborhoodBatchSampler(
        dataset,
        batch_size=6,
        batches_per_epoch=4,
        minimum_batch_size=4,
        condition_std=[10.0, 5.0, 3.0, 1.0],
        seed=7,
        shuffle=True,
    )
    for indices in sampler:
        assert torch.unique(dataset.map_ids[indices]).numel() == 1
        assert torch.unique(dataset.path_counts[indices]).numel() == 1
