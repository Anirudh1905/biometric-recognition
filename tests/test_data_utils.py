"""Tests for data_utils module."""

import json
from pathlib import Path

import pytest

from biometric_recognition.utils.data_utils import load_splits, save_splits


class TestSaveSplits:
    """Tests for save_splits function."""

    def test_save_splits_creates_file(self, temp_dir):
        """Test that save_splits creates a JSON file with splits."""
        splits = {
            "train": [0, 1, 2, 3, 4],
            "val": [5, 6],
            "test": [7, 8, 9],
        }

        result_path = save_splits(splits, temp_dir)

        assert Path(result_path).exists()
        assert result_path.endswith("data_splits.json")

    def test_save_splits_content_is_correct(self, temp_dir):
        """Test that saved splits can be read back correctly."""
        splits = {
            "train": [0, 1, 2, 3, 4],
            "val": [5, 6],
            "test": [7, 8, 9],
        }

        result_path = save_splits(splits, temp_dir)

        with open(result_path) as f:
            loaded = json.load(f)

        assert loaded == splits

    def test_save_splits_creates_parent_directories(self, temp_dir):
        """Test that save_splits creates parent directories if needed."""
        splits = {"train": [0, 1]}
        nested_path = temp_dir / "nested" / "deep" / "dir"

        result_path = save_splits(splits, nested_path)

        assert Path(result_path).exists()


class TestLoadSplits:
    """Tests for load_splits function."""

    def test_load_splits_reads_file(self, temp_dir):
        """Test that load_splits correctly reads a JSON file."""
        splits = {
            "train": [0, 1, 2],
            "val": [3, 4],
            "test": [5],
        }
        splits_path = temp_dir / "data_splits.json"
        with open(splits_path, "w") as f:
            json.dump(splits, f)

        loaded = load_splits(str(splits_path))

        assert loaded == splits

    def test_load_splits_raises_error_for_nonexistent_file(self):
        """Test that load_splits raises error for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_splits("/nonexistent/path/data_splits.json")
