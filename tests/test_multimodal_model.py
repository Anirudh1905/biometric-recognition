"""Tests for multimodal model module."""

import pytest
import torch

from biometric_recognition.models.multimodal_model import MultimodalBiometricModel


class TestMultimodalBiometricModel:
    """Tests for MultimodalBiometricModel class."""

    def test_initialization_default_params(self):
        """Test model initialization with default parameters."""
        model = MultimodalBiometricModel(num_classes=10)

        assert model is not None
        assert hasattr(model, "fingerprint_branch")
        assert hasattr(model, "iris_branch")
        assert hasattr(model, "fusion_module")
        assert hasattr(model, "classifier")

    def test_initialization_custom_params(self):
        """Test model initialization with custom parameters."""
        model = MultimodalBiometricModel(
            num_classes=45,
            fingerprint_backbone="mobilenetv2_100",
            fingerprint_feature_dim=1280,
            iris_feature_dim=32,
            fusion_hidden_dim=256,
            dropout=0.3,
            freeze_fingerprint_backbone=False,
        )

        assert model is not None

    def test_forward_pass_shape(self, sample_batch):
        """Test that forward pass produces correct output shape."""
        num_classes = 10
        model = MultimodalBiometricModel(num_classes=num_classes)
        model.eval()

        output = model(sample_batch)

        assert output.shape == (1, num_classes)

    def test_forward_pass_batch(self):
        """Test forward pass with batch of samples."""
        num_classes = 45
        batch_size = 4
        model = MultimodalBiometricModel(num_classes=num_classes)
        model.eval()

        batch = {
            "fingerprint": torch.randn(batch_size, 3, 128, 128),
            "left_iris": torch.randn(batch_size, 1, 64, 64),
            "right_iris": torch.randn(batch_size, 1, 64, 64),
        }

        output = model(batch)

        assert output.shape == (batch_size, num_classes)

    def test_get_features_returns_all_features(self, sample_batch):
        """Test that get_features returns features from all branches."""
        model = MultimodalBiometricModel(num_classes=10)
        model.eval()

        features = model.get_features(sample_batch)

        assert "fingerprint" in features
        assert "left_iris" in features
        assert "right_iris" in features
        assert "fused" in features

    def test_get_features_shapes(self, sample_batch):
        """Test that get_features returns correct shapes."""
        model = MultimodalBiometricModel(
            num_classes=10,
            fingerprint_feature_dim=1280,
            iris_feature_dim=32,
            fusion_hidden_dim=128,
        )
        model.eval()

        features = model.get_features(sample_batch)

        assert features["fingerprint"].shape == (1, 1280)
        assert features["left_iris"].shape == (1, 32)
        assert features["right_iris"].shape == (1, 32)
        assert features["fused"].shape == (1, 128)

    def test_model_in_training_mode(self, sample_batch):
        """Test that model can be set to training mode."""
        model = MultimodalBiometricModel(num_classes=10)
        model.train()

        # Should not raise
        output = model(sample_batch)

        assert output.shape == (1, 10)

    def test_model_in_eval_mode(self, sample_batch):
        """Test that model can be set to evaluation mode."""
        model = MultimodalBiometricModel(num_classes=10)
        model.eval()

        with torch.no_grad():
            output = model(sample_batch)

        assert output.shape == (1, 10)

    def test_classifier_output_matches_num_classes(self):
        """Test that classifier layer matches num_classes."""
        for num_classes in [5, 10, 45, 100]:
            model = MultimodalBiometricModel(num_classes=num_classes)

            assert model.classifier.out_features == num_classes

    def test_shared_iris_branch(self, sample_batch):
        """Test that left and right iris use the same branch."""
        model = MultimodalBiometricModel(num_classes=10)

        # The same IrisBranch instance should process both irises
        left_features = model.iris_branch(sample_batch["left_iris"])
        right_features = model.iris_branch(sample_batch["right_iris"])

        # Both should have the same shape
        assert left_features.shape == right_features.shape

    def test_model_parameter_count(self):
        """Test that model has expected number of parameters."""
        model = MultimodalBiometricModel(num_classes=10)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params > 0
        # With frozen backbone, trainable params should be less than total
        assert trainable_params < total_params

    def test_model_gradient_flow(self, sample_batch):
        """Test that gradients flow through trainable parts of the model."""
        model = MultimodalBiometricModel(num_classes=10)
        model.train()

        output = model(sample_batch)
        loss = output.sum()
        loss.backward()

        # Check that iris branch has gradients
        iris_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.iris_branch.parameters()
        )
        assert iris_has_grad

        # Check that fusion module has gradients
        fusion_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.fusion_module.parameters()
        )
        assert fusion_has_grad

        # Check that classifier has gradients
        classifier_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.classifier.parameters()
        )
        assert classifier_has_grad
