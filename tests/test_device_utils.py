"""Tests for device_utils module."""

import logging

import pytest
import torch

from biometric_recognition.utils.device_utils import get_device, print_device_info


class TestGetDevice:
    """Tests for get_device function."""

    def test_get_device_cpu_explicit(self):
        """Test getting CPU device when explicitly specified."""
        device = get_device("cpu")
        assert device == torch.device("cpu")

    def test_get_device_auto_returns_valid_device(self):
        """Test that auto device selection returns a valid device."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        # Should be one of cpu, cuda, or mps
        assert device.type in ("cpu", "cuda", "mps")

    def test_get_device_invalid_device(self):
        """Test that invalid device preference raises an error."""
        with pytest.raises(RuntimeError):
            get_device("invalid_device")


class TestPrintDeviceInfo:
    """Tests for print_device_info function."""

    def test_print_device_info_runs_without_error(self, caplog):
        """Test that print_device_info runs without raising exceptions."""
        with caplog.at_level(logging.INFO):
            print_device_info()
        # Should have logged something about device info
        assert "Device Information" in caplog.text or len(caplog.records) > 0
