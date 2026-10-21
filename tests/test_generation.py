import torch

from physics_csi.models.conditional_vae import PhysicsConditionedVAE
from physics_csi.physics.synthesizer import SystemConfig


def model_config():
    return {
        "max_paths": 5,
        "latent_dim": 8,
        "hidden_dim": 32,
        "condition_embedding_dim": 16,
        "path_embedding_dim": 16,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "dropout": 0.0,
        "condition_normalization": {
            "mean": [-90.0, 100.0, 5.0, 0.5],
            "std": [20.0, 100.0, 10.0, 0.5],
        },
        "path_scales": {"delay_ns": 1000.0, "log_power_db": 20.0},
        "constraints": {
            "enforce_total_power": True,
            "enforce_delay_spread": True,
            "enforce_k_factor": True,
            "enforce_los_direct_path": True,
            "epsilon": 1e-20,
        },
    }


def test_generation_returns_multiple_complex_channels():
    model = PhysicsConditionedVAE(model_config(), SystemConfig(num_subcarriers=8))
    generated = model.generate(
        {
            "total_power_db": -80.0,
            "delay_spread_ns": 0.0,
            "k_factor_db": 3.0,
            "los_status": 1,
        },
        num_samples=4,
        sample_path_count=False,
    )
    assert generated["csi"].shape == (4, 1, 64, 8)
    assert generated["csi"].is_complex()
    assert generated["target_conditions"].shape == (4, 4)

