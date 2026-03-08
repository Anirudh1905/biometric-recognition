"""Tests for model branches module."""

import torch

from biometric_recognition.models.branches import (
    FingerprintBranch,
    FusionModule,
    IrisBranch,
)


class TestFingerprintBranch:
    """Tests for FingerprintBranch class."""

    def test_initialization_default_params(self):
        """Test branch initialization with default parameters."""
        branch = FingerprintBranch()

        assert branch is not None
        assert hasattr(branch, "backbone")

    def test_initialization_custom_params(self):
        """Test branch initialization with custom parameters."""
        branch = FingerprintBranch(
            backbone_name="mobilenetv2_100",
            pretrained=False,
            freeze_backbone=False,
            feature_dim=1280,
        )

        assert branch is not None

    def test_forward_pass_shape(self, sample_fingerprint_tensor):
        """Test that forward pass produces correct output shape."""
        branch = FingerprintBranch(pretrained=False)

        output = branch(sample_fingerprint_tensor)

        # MobileNetV2 outputs 1280 features
        assert output.shape == (1, 1280)

    def test_forward_pass_batch(self):
        """Test forward pass with batch of images."""
        branch = FingerprintBranch(pretrained=False)
        batch_input = torch.randn(4, 3, 128, 128)

        output = branch(batch_input)

        assert output.shape == (4, 1280)

    def test_frozen_backbone_has_no_grad(self):
        """Test that frozen backbone parameters don't require gradients."""
        branch = FingerprintBranch(pretrained=False, freeze_backbone=True)

        for param in branch.backbone.parameters():
            assert not param.requires_grad

    def test_unfrozen_backbone_has_grad(self):
        """Test that unfrozen backbone parameters require gradients."""
        branch = FingerprintBranch(pretrained=False, freeze_backbone=False)

        for param in branch.backbone.parameters():
            assert param.requires_grad


class TestIrisBranch:
    """Tests for IrisBranch class."""

    def test_initialization(self):
        """Test branch initialization."""
        branch = IrisBranch()

        assert branch is not None
        assert hasattr(branch, "conv_layers")

    def test_forward_pass_shape(self, sample_iris_tensor):
        """Test that forward pass produces correct output shape."""
        branch = IrisBranch()

        output = branch(sample_iris_tensor)

        # Iris branch outputs 32 features (from conv2d output)
        assert output.shape == (1, 32)

    def test_forward_pass_batch(self):
        """Test forward pass with batch of images."""
        branch = IrisBranch()
        batch_input = torch.randn(4, 1, 64, 64)

        output = branch(batch_input)

        assert output.shape == (4, 32)

    def test_parameters_require_grad(self):
        """Test that iris branch parameters require gradients."""
        branch = IrisBranch()

        trainable_params = sum(
            p.numel() for p in branch.parameters() if p.requires_grad
        )
        assert trainable_params > 0


class TestFusionModule:
    """Tests for FusionModule class."""

    def test_initialization_default_params(self):
        """Test fusion module initialization with default parameters."""
        fusion = FusionModule()

        assert fusion is not None
        assert hasattr(fusion, "fusion_layer")

    def test_initialization_custom_params(self):
        """Test fusion module initialization with custom parameters."""
        fusion = FusionModule(
            fingerprint_dim=1280,
            iris_dim=32,
            hidden_dim=256,
            dropout=0.3,
        )

        assert fusion is not None

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        fusion = FusionModule(
            fingerprint_dim=1280,
            iris_dim=32,
            hidden_dim=128,
        )

        fingerprint_features = torch.randn(1, 1280)
        left_iris_features = torch.randn(1, 32)
        right_iris_features = torch.randn(1, 32)

        output = fusion(fingerprint_features, left_iris_features, right_iris_features)

        assert output.shape == (1, 128)

    def test_forward_pass_batch(self):
        """Test forward pass with batch of features."""
        fusion = FusionModule(
            fingerprint_dim=1280,
            iris_dim=32,
            hidden_dim=128,
        )
        batch_size = 4

        fingerprint_features = torch.randn(batch_size, 1280)
        left_iris_features = torch.randn(batch_size, 32)
        right_iris_features = torch.randn(batch_size, 32)

        output = fusion(fingerprint_features, left_iris_features, right_iris_features)

        assert output.shape == (batch_size, 128)

    def test_dropout_is_applied(self):
        """Test that dropout is applied during training."""
        fusion = FusionModule(dropout=0.5)
        fusion.train()

        fingerprint_features = torch.randn(100, 1280)
        left_iris_features = torch.randn(100, 32)
        right_iris_features = torch.randn(100, 32)

        # Run multiple times and check for variation (due to dropout)
        output1 = fusion(fingerprint_features, left_iris_features, right_iris_features)
        output2 = fusion(fingerprint_features, left_iris_features, right_iris_features)

        # Outputs should differ due to dropout in training mode
        assert not torch.allclose(output1, output2)
