"""Multimodal biometric recognition model."""

import torch
import torch.nn as nn
from typing import Dict, Any

from .branches import FingerprintBranch, IrisBranch, FusionModule


class MultimodalBiometricModel(nn.Module):
    """Complete multimodal biometric recognition model."""

    def __init__(
        self,
        num_classes: int,
        fingerprint_backbone: str = "mobilenetv2_100",
        fingerprint_feature_dim: int = 1280,
        iris_feature_dim: int = 32,
        fusion_hidden_dim: int = 128,
        dropout: float = 0.5,
        freeze_fingerprint_backbone: bool = True,
    ):
        """Initialize the multimodal model.

        Args:
            num_classes: Number of people to classify
            fingerprint_backbone: Backbone model for fingerprint processing
            fingerprint_feature_dim: Feature dimension for fingerprint branch
            iris_feature_dim: Feature dimension for iris branches
            fusion_hidden_dim: Hidden dimension for fusion layer
            dropout: Dropout probability
            freeze_fingerprint_backbone: Whether to freeze fingerprint backbone
        """
        super().__init__()

        # Individual branches
        self.fingerprint_branch = FingerprintBranch(
            backbone_name=fingerprint_backbone,
            pretrained=True,
            freeze_backbone=freeze_fingerprint_backbone,
            feature_dim=fingerprint_feature_dim,
        )

        # Shared iris branch (same weights for left and right iris)
        self.iris_branch = IrisBranch(feature_dim=iris_feature_dim)

        # Fusion module
        self.fusion_module = FusionModule(
            fingerprint_dim=fingerprint_feature_dim,
            iris_dim=iris_feature_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=dropout,
        )

        # Classification head
        self.classifier = nn.Linear(fusion_hidden_dim, num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the complete model.

        Args:
            batch: Dictionary containing:
                - fingerprint: [batch_size, 3, height, width]
                - left_iris: [batch_size, 1, height, width]
                - right_iris: [batch_size, 1, height, width]

        Returns:
            Logits [batch_size, num_classes]
        """
        # Extract features from each modality
        fingerprint_features = self.fingerprint_branch(batch["fingerprint"])
        left_iris_features = self.iris_branch(batch["left_iris"])
        right_iris_features = self.iris_branch(batch["right_iris"])

        # Fuse features
        fused_features = self.fusion_module(
            fingerprint_features, left_iris_features, right_iris_features
        )

        # Classification
        logits = self.classifier(fused_features)

        return logits

    def get_features(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract features from all branches without classification.

        Args:
            batch: Input batch dictionary

        Returns:
            Dictionary of extracted features
        """
        with torch.no_grad():
            fingerprint_features = self.fingerprint_branch(batch["fingerprint"])
            left_iris_features = self.iris_branch(batch["left_iris"])
            right_iris_features = self.iris_branch(batch["right_iris"])
            fused_features = self.fusion_module(
                fingerprint_features, left_iris_features, right_iris_features
            )

        return {
            "fingerprint": fingerprint_features,
            "left_iris": left_iris_features,
            "right_iris": right_iris_features,
            "fused": fused_features,
        }
