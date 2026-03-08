"""Pytest configuration and shared fixtures."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def device():
    """Provide a CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def sample_fingerprint_tensor():
    """Create a sample fingerprint tensor (batch_size=1, channels=3, height=128, width=128)."""
    return torch.randn(1, 3, 128, 128)


@pytest.fixture
def sample_iris_tensor():
    """Create a sample iris tensor (batch_size=1, channels=1, height=64, width=64)."""
    return torch.randn(1, 1, 64, 64)


@pytest.fixture
def sample_batch(sample_fingerprint_tensor, sample_iris_tensor):
    """Create a sample batch dictionary for model input."""
    return {
        "fingerprint": sample_fingerprint_tensor,
        "left_iris": sample_iris_tensor,
        "right_iris": sample_iris_tensor.clone(),
        "label": torch.tensor([0]),
    }


@pytest.fixture
def sample_pil_image_rgb():
    """Create a sample RGB PIL image."""
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def sample_pil_image_grayscale():
    """Create a sample grayscale PIL image."""
    arr = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_s3_client():
    """Mock boto3 S3 client."""
    with patch("boto3.client") as mock_client:
        mock_s3 = MagicMock()
        mock_client.return_value = mock_s3
        yield mock_s3
