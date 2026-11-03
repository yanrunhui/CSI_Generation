from __future__ import annotations

import torch
import pytest

from physics_csi.models.blocks import mask_from_path_count
from physics_csi.models.diffusion.multipath_diffusion import MultipathDiffusionModel
from physics_csi.models.diffusion.noise_schedule import GaussianDiffusion
from physics_csi.physics.constraints import PhysicsConstraintLayer
from physics_csi.physics.parameterization import PhysicsAwareParameterization
from physics_csi.physics.synthesizer import SystemConfig
from physics_csi.training.spatial_losses import paired_spatial_losses


def _stats() -> dict[str, object]:
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
        "path_count_histogram": [0, 10, 20, 70],
    }


def _config() -> dict[str, object]:
    return {
        "max_paths": 3,
        "hidden_dim": 32,
        "condition_embedding_dim": 16,
        "layers": 1,
        "attention_heads": 4,
        "dropout": 0.0,
        "diffusion_steps": 4,
        "condition_dropout": 0.1,
        "path_count_loss_weight": 1.0,
        "constraints": {
            "enforce_total_power": True,
            "enforce_delay_spread": True,
            "enforce_k_factor": True,
            "enforce_los_direct_path": True,
            "epsilon": 1e-20,
        },
    }


def _batch() -> dict[str, torch.Tensor]:
    mask = torch.tensor([[True, True, False], [True, True, True]])
    return {
        "conditions": torch.tensor([[-80.0, 10.0, 3.0, 1.0], [-75.0, 20.0, 2.0, 1.0]]),
        "condition_validity": torch.ones(2, 4, dtype=torch.bool),
        "path_count": torch.tensor([2, 3]),
        "path_mask": mask,
        "path_delays_ns": torch.tensor([[10.0, 30.0, 0.0], [4.0, 12.0, 45.0]]),
        "path_powers_linear": torch.tensor([[8e-9, 2e-9, 0.0], [6e-9, 3e-9, 1e-9]]),
        "path_phases_rad": torch.tensor([[0.2, -1.0, 0.0], [0.0, 1.0, -2.0]]),
        "aoa_az_deg": torch.tensor([[10.0, -20.0, 0.0], [20.0, -30.0, 80.0]]),
        "aoa_el_deg": torch.tensor([[5.0, 10.0, 0.0], [0.0, -10.0, 30.0]]),
        "aod_az_deg": torch.tensor([[90.0, -90.0, 0.0], [45.0, 30.0, -80.0]]),
        "aod_el_deg": torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]]),
        "direct_path_mask": torch.tensor([[True, False, False], [True, False, False]]),
    }


def test_path_mask_and_masked_diffusion_loss() -> None:
    counts = torch.tensor([1, 3])
    mask = mask_from_path_count(counts, 3)
    assert mask.tolist() == [[True, False, False], [True, True, True]]
    diffusion = GaussianDiffusion(timesteps=4)
    clean = torch.randn(2, 3, 2)
    element_mask = mask.unsqueeze(-1).expand_as(clean)

    def exact_noise(noisy: torch.Tensor, timestep: torch.Tensor, unconditional: bool) -> torch.Tensor:
        del noisy, timestep, unconditional
        return target_noise

    target_noise = torch.randn_like(clean) * element_mask
    original = diffusion.add_noise
    diffusion.add_noise = lambda value, timestep, noise=None: original(value, timestep, target_noise)  # type: ignore[method-assign]
    assert diffusion.training_loss(clean, exact_noise, element_mask=element_mask) == 0


def test_multipath_diffusion_forward_is_finite() -> None:
    model = MultipathDiffusionModel(_config(), SystemConfig(num_subcarriers=8, tx_array_shape=(2, 2)), _stats())
    metrics = model(_batch())
    assert set(metrics) >= {"loss", "diffusion_loss", "path_count_loss"}
    assert all(torch.isfinite(value) for value in metrics.values())
    metrics["loss"].backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_path_count_sampling_and_csi_generation_are_seeded() -> None:
    model = MultipathDiffusionModel(_config(), SystemConfig(num_subcarriers=8, tx_array_shape=(2, 2)), _stats()).eval()
    conditions = {"total_power_db": -80.0, "delay_spread_ns": 10.0, "k_factor_db": 3.0, "los_status": 1}
    first = model.generate(conditions, 2, sampling_steps=2, generator=torch.Generator().manual_seed(7))
    second = model.generate(conditions, 2, sampling_steps=2, generator=torch.Generator().manual_seed(7))
    assert first["csi"].shape == (2, 1, 4, 8)
    assert torch.all(first["path_count"] >= 2)
    assert torch.all(first["path_delays_ns"][:, 1] != first["path_delays_ns"][:, 0])
    assert torch.equal(first["path_count"], second["path_count"])
    assert torch.allclose(first["csi"], second["csi"])
    assert torch.isfinite(first["csi"].real).all()


def test_structured_parameterization_and_forced_path_count() -> None:
    config = _config()
    config["path_count"] = {
        "class_weight_power": 0.5,
        "max_weight_ratio": 4.0,
        "prior_correction": True,
    }
    config["parameterization"] = {
        "mode": "structured",
        "structured_delay": True,
        "structured_power": True,
        "structured_k_factor": True,
    }
    model = MultipathDiffusionModel(
        config,
        SystemConfig(num_subcarriers=8, tx_array_shape=(2, 2)),
        _stats(),
    ).eval()
    conditions = {
        "total_power_db": -80.0,
        "delay_spread_ns": 10.0,
        "k_factor_db": 3.0,
        "los_status": 1,
    }
    generated = model.generate(
        conditions,
        2,
        sampling_steps=2,
        forced_path_count=2,
        generator=torch.Generator().manual_seed(9),
    )
    assert generated["path_count"].tolist() == [2, 2]
    assert torch.all(generated["path_count_was_forced"])
    assert torch.allclose(
        generated["actual_conditions"][:, 1], torch.full((2,), 10.0), atol=1e-4
    )
    assert torch.all(generated["constraint_delay_change_ns"] < 1e-3)


def test_scene_id_conditions_count_and_denoising_and_receives_gradients() -> None:
    config = _config()
    config["condition_dropout"] = 0.0
    config["scene_conditioning"] = {
        "enabled": True,
        "embedding_dim": 8,
        "fusion_hidden_dim": 32,
        "map_id_to_index": {"0": 0, "7": 1},
    }
    model = MultipathDiffusionModel(
        config,
        SystemConfig(num_subcarriers=8, tx_array_shape=(2, 2)),
        _stats(),
    )
    batch = _batch()
    batch["map_id"] = torch.tensor([0, 7])
    same_conditions = batch["conditions"][:1].expand(2, -1)
    same_validity = batch["condition_validity"][:1].expand(2, -1)
    embeddings = model.condition_embedding(
        same_conditions, same_validity, map_id=torch.tensor([0, 7])
    )
    assert not torch.allclose(embeddings[0], embeddings[1])
    count_logits = model.path_count_model(embeddings)
    assert not torch.allclose(count_logits[0], count_logits[1])
    noisy = torch.zeros(2, 3, model.representation.feature_dim)
    timestep = torch.ones(2, dtype=torch.long)
    count = torch.tensor([2, 2])
    path_mask = mask_from_path_count(count, 3)
    denoised = model.denoiser(noisy, timestep, embeddings, count, path_mask)
    assert not torch.allclose(denoised[0], denoised[1])
    metrics = model(batch)
    metrics["loss"].backward()
    gradient_norm = model.scene_gradient_norm()
    assert gradient_norm is not None and float(gradient_norm) > 0.0


def test_scene_id_generation_rejects_unseen_map() -> None:
    config = _config()
    config["scene_conditioning"] = {
        "enabled": True,
        "embedding_dim": 8,
        "fusion_hidden_dim": 32,
        "map_id_to_index": {"0": 0, "7": 1},
    }
    model = MultipathDiffusionModel(
        config,
        SystemConfig(num_subcarriers=8, tx_array_shape=(2, 2)),
        _stats(),
    ).eval()
    conditions = {
        "total_power_db": -80.0,
        "delay_spread_ns": 10.0,
        "k_factor_db": 3.0,
        "los_status": 1,
    }
    with pytest.raises(ValueError, match="unseen Map ID"):
        model.generate(conditions, 2, map_id=99, sampling_steps=2)
    generated = model.generate(
        conditions,
        2,
        map_id=7,
        sampling_steps=2,
        generator=torch.Generator().manual_seed(11),
    )
    assert generated["map_id"].tolist() == [7, 7]
    assert generated["map_index"].tolist() == [1, 1]


def test_paired_spatial_losses_are_zero_for_identical_csi() -> None:
    generator = torch.Generator().manual_seed(17)
    real = torch.randn(3, 1, 4, 8, generator=generator)
    imaginary = torch.randn(3, 1, 4, 8, generator=generator)
    csi = torch.complex(real, imaginary)
    beamspace, spatial = paired_spatial_losses(csi, csi.clone(), (2, 2))
    assert float(beamspace) == pytest.approx(0.0, abs=1e-8)
    assert float(spatial) == pytest.approx(0.0, abs=1e-8)
    changed = csi.clone()
    changed[:, :, 0] = changed[:, :, 0] * torch.exp(torch.tensor(0.7j))
    _, changed_spatial = paired_spatial_losses(changed, csi, (2, 2))
    assert float(changed_spatial) > 0.0


def test_zero_csi_spatial_losses_have_finite_gradient() -> None:
    real = torch.zeros(2, 1, 4, 8, requires_grad=True)
    imaginary = torch.zeros_like(real)
    predicted = torch.complex(real, imaginary)
    target = torch.zeros_like(predicted)
    beamspace, spatial = paired_spatial_losses(predicted, target, (2, 2))
    (beamspace + spatial).backward()
    assert real.grad is not None
    assert torch.isfinite(real.grad).all()


def test_degenerate_structured_delay_has_finite_gradient() -> None:
    constraints = PhysicsConstraintLayer(
        enforce_total_power=True,
        enforce_delay_spread=True,
        enforce_k_factor=True,
        enforce_los_direct_path=True,
        epsilon=1e-20,
    )
    parameterization = PhysicsAwareParameterization(
        constraints,
        {
            "mode": "structured",
            "structured_delay": True,
            "structured_power": True,
            "structured_k_factor": True,
            "epsilon": 1e-20,
        },
    )
    delay_shape = torch.zeros(1, 3, requires_grad=True)
    power_weights = torch.ones(1, 3, requires_grad=True)
    raw = {
        "path_mask": torch.tensor([[True, True, False]]),
        "positive_power_weights": power_weights,
        "relative_delay_shape": delay_shape,
        "first_delay_ns": torch.tensor([5.0], requires_grad=True),
    }
    conditions = torch.tensor([[-80.0, 10.0, 3.0, 1.0]])
    validity = torch.ones(1, 4, dtype=torch.bool)
    result = parameterization(raw, conditions, validity)
    loss = (
        result.constrained.delays_ns.sum()
        + result.constrained.powers_linear.sum()
    )
    loss.backward()
    for tensor in (delay_shape, power_weights, raw["first_delay_ns"]):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_low_noise_spatial_supervision_is_finite_and_differentiable() -> None:
    config = _config()
    config["condition_dropout"] = 0.0
    config["scene_conditioning"] = {
        "enabled": True,
        "embedding_dim": 8,
        "fusion_hidden_dim": 32,
        "map_id_to_index": {"0": 0, "7": 1},
    }
    config["spatial_supervision"] = {
        "enabled": True,
        "beamspace_weight": 0.05,
        "spatial_weight": 0.02,
        "maximum_timestep_fraction": 1.0,
    }
    system = SystemConfig(num_subcarriers=8, tx_array_shape=(2, 2))
    model = MultipathDiffusionModel(config, system, _stats())
    batch = _batch()
    batch["map_id"] = torch.tensor([0, 7])
    batch["csi"] = model.synthesizer(
        batch["path_delays_ns"],
        batch["path_powers_linear"],
        batch["path_phases_rad"],
        batch["aoa_az_deg"],
        batch["aoa_el_deg"],
        batch["aod_az_deg"],
        batch["aod_el_deg"],
        batch["path_mask"],
    ).detach()
    metrics = model(batch)
    assert set(metrics) >= {
        "beamspace_loss",
        "spatial_loss",
        "spatial_auxiliary_weighted",
        "spatial_supervision_fraction",
    }
    assert all(torch.isfinite(value) for value in metrics.values())
    metrics["loss"].backward()
    gradient_norm = model.scene_gradient_norm()
    assert gradient_norm is not None and torch.isfinite(gradient_norm)


def test_scene_residual_adapter_starts_as_exact_frozen_b3() -> None:
    base_config = _config()
    base_config["condition_dropout"] = 0.0
    base = MultipathDiffusionModel(
        base_config,
        SystemConfig(num_subcarriers=8, tx_array_shape=(2, 2)),
        _stats(),
    ).eval()
    residual_config = _config()
    residual_config["condition_dropout"] = 0.0
    residual_config["scene_residual_adapter"] = {
        "enabled": True,
        "map_id_to_index": {"0": 0, "7": 1},
        "embedding_dim": 8,
        "hidden_dim": 16,
        "layers": 1,
        "attention_heads": 4,
        "dropout": 0.0,
        "freeze_backbone": True,
    }
    residual = MultipathDiffusionModel(
        residual_config,
        SystemConfig(num_subcarriers=8, tx_array_shape=(2, 2)),
        _stats(),
    ).eval()
    compatible = {
        key: value
        for key, value in base.state_dict().items()
        if key in residual.state_dict() and residual.state_dict()[key].shape == value.shape
    }
    residual.load_state_dict(compatible, strict=False)
    batch = _batch()
    features, _ = base.representation.encode(batch)
    timestep = torch.tensor([1, 2])
    base_condition = base.condition_embedding(
        batch["conditions"], batch["condition_validity"]
    )
    residual_condition = residual.condition_embedding(
        batch["conditions"], batch["condition_validity"]
    )
    base_prediction = base._predict_noise(
        features,
        timestep,
        base_condition,
        batch["path_count"],
        batch["path_mask"],
        map_index=None,
        drop_mask=None,
        unconditional=False,
    )
    map_index = residual.map_indices(
        torch.tensor([0, 7]), 2, features.device
    )
    residual_prediction = residual._predict_noise(
        features,
        timestep,
        residual_condition,
        batch["path_count"],
        batch["path_mask"],
        map_index=map_index,
        drop_mask=None,
        unconditional=False,
    )
    assert torch.allclose(base_prediction, residual_prediction, atol=1e-6, rtol=1e-6)
    assert not any(parameter.requires_grad for parameter in residual.denoiser.parameters())
    assert residual.scene_residual_adapter is not None
    assert all(
        parameter.requires_grad
        for parameter in residual.scene_residual_adapter.parameters()
    )
    residual.train()
    training_batch = _batch()
    training_batch["map_id"] = torch.tensor([0, 7])
    metrics = residual(training_batch)
    metrics["loss"].backward()
    gradient_norm = residual.scene_gradient_norm()
    assert gradient_norm is not None and float(gradient_norm) > 0.0
    assert all(parameter.grad is None for parameter in residual.denoiser.parameters())

