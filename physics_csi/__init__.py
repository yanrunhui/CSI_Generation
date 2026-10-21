"""Physics-controllable MIMO-OFDM CSI generation."""

from physics_csi.models.conditional_vae import PhysicsConditionedVAE
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig

__all__ = ["MIMOOFDMSynthesizer", "PhysicsConditionedVAE", "SystemConfig"]
__version__ = "0.1.0"

