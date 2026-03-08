"""Models package for biometric recognition."""

from .branches import FingerprintBranch, FusionModule, IrisBranch
from .multimodal_model import MultimodalBiometricModel

__all__ = [
    "MultimodalBiometricModel",
    "FingerprintBranch",
    "IrisBranch",
    "FusionModule",
]
