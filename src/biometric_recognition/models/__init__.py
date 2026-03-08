"""Models package for biometric recognition."""

from .multimodal_model import MultimodalBiometricModel
from .branches import FingerprintBranch, IrisBranch, FusionModule

__all__ = [
    "MultimodalBiometricModel",
    "FingerprintBranch",
    "IrisBranch",
    "FusionModule",
]
