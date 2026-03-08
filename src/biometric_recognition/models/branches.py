"""Neural network branches for different biometric modalities."""

import timm
import torch
import torch.nn as nn


class FingerprintBranch(nn.Module):
    """Branch for processing fingerprint images using a pretrained backbone."""

    def __init__(
        self,
        backbone_name: str = "mobilenetv2_100",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        feature_dim: int = 1280,
    ):
        """Initialize fingerprint processing branch.

        Args:
            backbone_name: Name of the timm model to use as backbone
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze the backbone weights
            feature_dim: Output feature dimension
        """
        super().__init__()

        # Load backbone from timm
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool="avg",
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # No projection layer - use MobileNetV2 features directly

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through fingerprint branch.

        Args:
            x: Input fingerprint images [batch_size, 3, height, width]

        Returns:
            Feature vector [batch_size, 1280] - MobileNetV2 output
        """
        # Return MobileNetV2 features directly
        return self.backbone(x)


class IrisBranch(nn.Module):
    """Branch for processing iris images using custom CNN."""

    def __init__(self, feature_dim: int = 64):
        """Initialize iris processing branch.

        Args:
            feature_dim: Output feature dimension
        """
        super().__init__()

        # 2 conv layers + global avg pooling
        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32
            # Second convolutional block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),  # 16x16 -> 1x1, outputs 32 features
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through iris branch.

        Args:
            x: Input iris images [batch_size, 1, height, width]

        Returns:
            Feature vector [batch_size, 32]
        """
        features = self.conv_layers(x)
        # After global avg pooling: [batch_size, 32, 1, 1] -> [batch_size, 32]
        return features.view(features.size(0), -1)


class FusionModule(nn.Module):
    """Module for fusing features from different modalities."""

    def __init__(
        self,
        fingerprint_dim: int = 1280,
        iris_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.5,
    ):
        """Initialize fusion module.

        Args:
            fingerprint_dim: Dimension of fingerprint features
            iris_dim: Dimension of iris features
            hidden_dim: Hidden dimension for fusion layer
            dropout: Dropout probability
        """
        super().__init__()

        # Total input dimension (fingerprint + 2 * iris)
        total_dim = fingerprint_dim + 2 * iris_dim

        self.fusion_layer = nn.Sequential(
            nn.Linear(total_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)
        )

    def forward(
        self,
        fingerprint_features: torch.Tensor,
        left_iris_features: torch.Tensor,
        right_iris_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse features from all modalities.

        Args:
            fingerprint_features: Features from fingerprint branch
            left_iris_features: Features from left iris branch
            right_iris_features: Features from right iris branch

        Returns:
            Fused features
        """
        # Concatenate all features
        fused = torch.cat(
            [fingerprint_features, left_iris_features, right_iris_features], dim=1
        )

        return self.fusion_layer(fused)
