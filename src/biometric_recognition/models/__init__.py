"""Models package for biometric recognition."""

from biometric_recognition.models.branches import (
    FingerprintBranch,
    FusionModule,
    IrisBranch,
)
from biometric_recognition.models.multimodal_model import MultimodalBiometricModel

__all__ = [
    "MultimodalBiometricModel",
    "FingerprintBranch",
    "IrisBranch",
    "FusionModule",
]
