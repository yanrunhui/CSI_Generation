from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch


class LegacyPGCVAEAdapter:
    """Loads the sibling PGC-VAE for its original narrowband angular-grid baseline.

    The legacy output is deliberately not repeated across OFDM subcarriers: that
    would invent frequency selectivity absent from the baseline.
    """

    def __init__(
        self,
        reference_root: str | Path,
        checkpoint: str | Path,
        *,
        latent_dim: int = 64,
        resolution: int = 32,
        device: str | torch.device = "cpu",
    ) -> None:
        root = Path(reference_root).resolve()
        if not (root / "vae_model.py").exists():
            raise FileNotFoundError(f"PGC-VAE source not found at {root}")
        sys.path.insert(0, str(root))
        try:
            module = importlib.import_module("vae_model")
            self.model = module.WeightsVAE(
                in_channels=2,
                latent_dim=latent_dim,
                config={"resolution": resolution},
                use_conv=False,
            )
        finally:
            if sys.path[0] == str(root):
                sys.path.pop(0)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(device).eval()
        self.latent_dim = latent_dim
        self.device = torch.device(device)

    @torch.no_grad()
    def sample_angular_weights(self, num_samples: int) -> torch.Tensor:
        latent = torch.randn(num_samples, self.latent_dim, device=self.device)
        return self.model.decode(latent, force_relu=True)

    def generate_wideband(self, *_: Any, **__: Any) -> torch.Tensor:
        raise NotImplementedError(
            "The original PGC-VAE is narrowband and has no delay model. Use it only for the "
            "audited narrowband/angular-grid baseline; do not fabricate OFDM subcarriers."
        )

