"""Tests for image_utils module."""

import base64
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from biometric_recognition.utils.image_utils import (
    image_file_to_base64,
    preprocess_image,
    prepare_batch_from_images,
)


class TestImageFileToBase64:
    """Tests for image_file_to_base64 function."""

    def test_converts_image_to_base64(self, temp_dir, sample_pil_image_rgb):
        """Test that image file is correctly converted to base64."""
        image_path = temp_dir / "test_image.png"
        sample_pil_image_rgb.save(image_path)

        result = image_file_to_base64(str(image_path))

        # Result should be a valid base64 string
        assert isinstance(result, str)
        # Should be decodable
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_raises_error_for_nonexistent_file(self):
        """Test that non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            image_file_to_base64("/nonexistent/path/image.png")


class TestPreprocessImage:
    """Tests for preprocess_image function."""

    def test_preprocess_rgb_image(self, sample_pil_image_rgb):
        """Test preprocessing an RGB image."""
        target_size = (128, 128)
        result = preprocess_image(
            sample_pil_image_rgb, target_size, grayscale=False, add_batch_dim=True
        )

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 3, 128, 128)  # batch, channels, height, width
        assert result.dtype == torch.float32
        # Values should be normalized to [0, 1]
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preprocess_grayscale_image(self, sample_pil_image_grayscale):
        """Test preprocessing a grayscale image."""
        target_size = (64, 64)
        result = preprocess_image(
            sample_pil_image_grayscale, target_size, grayscale=True, add_batch_dim=True
        )

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 1, 64, 64)  # batch, channels, height, width
        assert result.dtype == torch.float32

    def test_preprocess_without_batch_dim(self, sample_pil_image_rgb):
        """Test preprocessing without adding batch dimension."""
        target_size = (64, 64)
        result = preprocess_image(
            sample_pil_image_rgb, target_size, grayscale=False, add_batch_dim=False
        )

        assert result.shape == (3, 64, 64)  # channels, height, width

    def test_preprocess_converts_rgb_to_grayscale(self, sample_pil_image_rgb):
        """Test that RGB image is converted to grayscale when grayscale=True."""
        target_size = (64, 64)
        result = preprocess_image(
            sample_pil_image_rgb, target_size, grayscale=True, add_batch_dim=False
        )

        assert result.shape == (1, 64, 64)  # 1 channel for grayscale


class TestPrepareBatchFromImages:
    """Tests for prepare_batch_from_images function."""

    def test_prepare_batch_default_sizes(
        self, sample_pil_image_rgb, sample_pil_image_grayscale
    ):
        """Test preparing a batch with default sizes."""
        batch = prepare_batch_from_images(
            fingerprint_img=sample_pil_image_rgb,
            left_iris_img=sample_pil_image_grayscale,
            right_iris_img=sample_pil_image_grayscale,
        )

        assert "fingerprint" in batch
        assert "left_iris" in batch
        assert "right_iris" in batch

        # Default sizes: fingerprint (128, 128), iris (64, 64)
        assert batch["fingerprint"].shape == (1, 3, 128, 128)
        assert batch["left_iris"].shape == (1, 1, 64, 64)
        assert batch["right_iris"].shape == (1, 1, 64, 64)

    def test_prepare_batch_custom_sizes(
        self, sample_pil_image_rgb, sample_pil_image_grayscale
    ):
        """Test preparing a batch with custom sizes."""
        batch = prepare_batch_from_images(
            fingerprint_img=sample_pil_image_rgb,
            left_iris_img=sample_pil_image_grayscale,
            right_iris_img=sample_pil_image_grayscale,
            fingerprint_size=(64, 64),
            iris_size=(32, 32),
        )

        assert batch["fingerprint"].shape == (1, 3, 64, 64)
        assert batch["left_iris"].shape == (1, 1, 32, 32)
        assert batch["right_iris"].shape == (1, 1, 32, 32)

    def test_prepare_batch_with_device(
        self, sample_pil_image_rgb, sample_pil_image_grayscale, device
    ):
        """Test preparing a batch and moving to device."""
        batch = prepare_batch_from_images(
            fingerprint_img=sample_pil_image_rgb,
            left_iris_img=sample_pil_image_grayscale,
            right_iris_img=sample_pil_image_grayscale,
            device=device,
        )

        assert batch["fingerprint"].device == device
        assert batch["left_iris"].device == device
        assert batch["right_iris"].device == device
