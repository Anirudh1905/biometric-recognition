"""Tests for model_utils module."""

import torch

from biometric_recognition.models import MultimodalBiometricModel
from biometric_recognition.utils.model_utils import (
    get_model_info,
    move_batch_to_device,
    save_checkpoint,
)


class TestMoveBatchToDevice:
    """Tests for move_batch_to_device function."""

    def test_moves_tensors_to_device(self, device):
        """Test that tensors are moved to the specified device."""
        batch = {
            "fingerprint": torch.randn(1, 3, 128, 128),
            "left_iris": torch.randn(1, 1, 64, 64),
            "right_iris": torch.randn(1, 1, 64, 64),
            "label": torch.tensor([0]),
        }

        result = move_batch_to_device(batch, device)

        for key in ["fingerprint", "left_iris", "right_iris", "label"]:
            assert result[key].device == device

    def test_preserves_non_tensor_values(self, device):
        """Test that non-tensor values are preserved."""
        batch = {
            "fingerprint": torch.randn(1, 3, 128, 128),
            "person_id": 42,
            "metadata": {"name": "test"},
        }

        result = move_batch_to_device(batch, device)

        assert result["person_id"] == 42
        assert result["metadata"] == {"name": "test"}

    def test_returns_new_dict(self, device):
        """Test that function returns a new dictionary."""
        batch = {"fingerprint": torch.randn(1, 3, 128, 128)}

        result = move_batch_to_device(batch, device)

        assert result is not batch


class TestGetModelInfo:
    """Tests for get_model_info function."""

    def test_returns_model_information(self):
        """Test that function returns expected model information."""
        model = MultimodalBiometricModel(num_classes=10)

        info = get_model_info(model, model_path="test/path/model.pth")

        assert "model_path" in info
        assert info["model_path"] == "test/path/model.pth"
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "model_size_mb" in info
        assert "architecture" in info

    def test_parameter_counts_are_positive(self):
        """Test that parameter counts are positive integers."""
        model = MultimodalBiometricModel(num_classes=10)

        info = get_model_info(model)

        assert info["total_parameters"] > 0
        assert info["trainable_parameters"] >= 0
        assert info["model_size_mb"] > 0

    def test_architecture_info_is_present(self):
        """Test that architecture details are included."""
        model = MultimodalBiometricModel(num_classes=10)

        info = get_model_info(model)

        arch = info["architecture"]
        assert "fingerprint_feature_dim" in arch
        assert "iris_feature_dim" in arch
        assert "fusion_hidden_dim" in arch


class TestSaveCheckpoint:
    """Tests for save_checkpoint function."""

    def test_saves_checkpoint_file(self, temp_dir):
        """Test that checkpoint file is created."""
        model = MultimodalBiometricModel(num_classes=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        checkpoint_path = temp_dir / "checkpoint.pth"

        save_checkpoint(
            path=checkpoint_path,
            epoch=5,
            model=model,
            optimizer=optimizer,
            val_accuracy=85.5,
        )

        assert checkpoint_path.exists()

    def test_checkpoint_contains_required_fields(self, temp_dir):
        """Test that checkpoint contains all required fields."""
        model = MultimodalBiometricModel(num_classes=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        checkpoint_path = temp_dir / "checkpoint.pth"

        save_checkpoint(
            path=checkpoint_path,
            epoch=5,
            model=model,
            optimizer=optimizer,
            val_accuracy=85.5,
        )

        checkpoint = torch.load(checkpoint_path, weights_only=False)

        assert "epoch" in checkpoint
        assert checkpoint["epoch"] == 5
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "val_accuracy" in checkpoint
        assert checkpoint["val_accuracy"] == 85.5

    def test_saves_optional_training_history(self, temp_dir):
        """Test that optional training history is saved."""
        model = MultimodalBiometricModel(num_classes=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        checkpoint_path = temp_dir / "checkpoint.pth"

        train_losses = [1.0, 0.8, 0.6]
        val_losses = [1.1, 0.9, 0.7]
        val_accuracies = [60.0, 70.0, 80.0]

        save_checkpoint(
            path=checkpoint_path,
            epoch=3,
            model=model,
            optimizer=optimizer,
            val_accuracy=80.0,
            train_losses=train_losses,
            val_losses=val_losses,
            val_accuracies=val_accuracies,
        )

        checkpoint = torch.load(checkpoint_path, weights_only=False)

        assert checkpoint["train_losses"] == train_losses
        assert checkpoint["val_losses"] == val_losses
        assert checkpoint["val_accuracies"] == val_accuracies

    def test_saves_config_as_dict(self, temp_dir):
        """Test that DictConfig is converted and saved."""
        from omegaconf import OmegaConf

        model = MultimodalBiometricModel(num_classes=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        checkpoint_path = temp_dir / "checkpoint.pth"

        cfg = OmegaConf.create({"model": {"dropout": 0.5}, "data": {"batch_size": 32}})

        save_checkpoint(
            path=checkpoint_path,
            epoch=1,
            model=model,
            optimizer=optimizer,
            val_accuracy=50.0,
            cfg=cfg,
        )

        checkpoint = torch.load(checkpoint_path, weights_only=False)

        assert "config" in checkpoint
        assert checkpoint["config"]["model"]["dropout"] == 0.5
