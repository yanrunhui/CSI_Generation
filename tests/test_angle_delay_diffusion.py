from __future__ import annotations

import torch

from physics_csi.models.diffusion.angle_delay_diffusion import AngleDelayDiffusionModel
from physics_csi.physics.synthesizer import SystemConfig


def _model() -> AngleDelayDiffusionModel:
    config = {
        "channels": 8,
        "layers": 2,
        "condition_embedding_dim": 16,
        "condition_hidden_dim": 32,
        "dropout": 0.0,
        "diffusion_steps": 4,
        "condition_dropout": 0.1,
    }
    statistics = {
        "condition_mean": [-80.0, 50.0, 3.0, 0.5],
        "condition_std": [10.0, 20.0, 5.0, 0.5],
        "multipath": {},
        "angle_delay_scale": 0.25,
    }
    return AngleDelayDiffusionModel(
        config, SystemConfig(num_subcarriers=8, tx_array_shape=(2, 2)), statistics
    )


def test_angle_delay_diffusion_forward_and_gradient() -> None:
    model = _model()
    batch = {
        "conditions": torch.tensor([[-80.0, 10.0, 3.0, 1.0], [-75.0, 20.0, 0.0, 0.0]]),
        "condition_validity": torch.tensor([[True, True, True, True], [True, True, False, True]]),
        "csi": torch.complex(torch.randn(2, 1, 4, 8), torch.randn(2, 1, 4, 8)),
    }
    metrics = model(batch)
    assert torch.isfinite(metrics["loss"])
    metrics["loss"].backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_angle_delay_reverse_shape_and_checkpoint_roundtrip(tmp_path) -> None:
    model = _model().eval()
    path = tmp_path / "state.pt"
    torch.save(model.state_dict(), path)
    restored = _model().eval()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    conditions = {"total_power_db": -80.0, "delay_spread_ns": 10.0, "k_factor_db": 3.0, "los_status": 1}
    generated = restored.generate(
        conditions, 2, sampling_steps=2, generator=torch.Generator().manual_seed(11)
    )
    assert generated["angle_delay"].shape == (2, 2, 2, 2, 8)
    assert generated["csi"].shape == (2, 1, 4, 8)
    assert torch.isfinite(generated["csi"].real).all()
