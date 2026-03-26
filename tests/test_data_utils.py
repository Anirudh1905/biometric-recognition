"""Tests for data_utils module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from biometric_recognition.utils.data_utils import (
    _assign_to_splits,
    create_data_loader,
    create_data_loaders,
    create_dataset,
    create_stratified_splits,
    load_splits,
    save_splits,
)


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


class TestCreateStratifiedSplits:
    """Tests for create_stratified_splits function."""

    def test_no_image_leakage_across_splits(self):
        """Verify no image appears in multiple splits."""

        class MockDataset:
            samples = [
                {
                    "person_id": 0,
                    "fingerprint_path": "fp1",
                    "left_iris_path": "left1",
                    "right_iris_path": "right1",
                },
                {
                    "person_id": 0,
                    "fingerprint_path": "fp1",
                    "left_iris_path": "left1",
                    "right_iris_path": "right2",
                },
                {
                    "person_id": 0,
                    "fingerprint_path": "fp1",
                    "left_iris_path": "left2",
                    "right_iris_path": "right1",
                },
                {
                    "person_id": 0,
                    "fingerprint_path": "fp2",
                    "left_iris_path": "left1",
                    "right_iris_path": "right1",
                },
                {
                    "person_id": 0,
                    "fingerprint_path": "fp2",
                    "left_iris_path": "left2",
                    "right_iris_path": "right2",
                },
                {
                    "person_id": 1,
                    "fingerprint_path": "fp3",
                    "left_iris_path": "left3",
                    "right_iris_path": "right3",
                },
                {
                    "person_id": 1,
                    "fingerprint_path": "fp4",
                    "left_iris_path": "left4",
                    "right_iris_path": "right4",
                },
            ]

        dataset = MockDataset()
        train_idx, val_idx, test_idx = create_stratified_splits(dataset)

        def get_images(indices):
            images = set()
            for idx in indices:
                s = dataset.samples[idx]
                images.add(s["fingerprint_path"])
                images.add(s["left_iris_path"])
                images.add(s["right_iris_path"])
            return images

        train_images = get_images(train_idx)
        val_images = get_images(val_idx)
        test_images = get_images(test_idx)

        # Verify no overlap between any splits
        assert train_images.isdisjoint(val_images), "Train and val share images!"
        assert train_images.isdisjoint(test_images), "Train and test share images!"
        assert val_images.isdisjoint(test_images), "Val and test share images!"

    def test_all_samples_have_consistent_split_assignment(self):
        """Verify each sample's images are all in the same split."""

        class MockDataset:
            samples = [
                {
                    "person_id": 0,
                    "fingerprint_path": "fp1",
                    "left_iris_path": "left1",
                    "right_iris_path": "right1",
                },
                {
                    "person_id": 0,
                    "fingerprint_path": "fp2",
                    "left_iris_path": "left2",
                    "right_iris_path": "right2",
                },
                {
                    "person_id": 0,
                    "fingerprint_path": "fp3",
                    "left_iris_path": "left3",
                    "right_iris_path": "right3",
                },
                {
                    "person_id": 1,
                    "fingerprint_path": "fp4",
                    "left_iris_path": "left4",
                    "right_iris_path": "right4",
                },
                {
                    "person_id": 1,
                    "fingerprint_path": "fp5",
                    "left_iris_path": "left5",
                    "right_iris_path": "right5",
                },
                {
                    "person_id": 1,
                    "fingerprint_path": "fp6",
                    "left_iris_path": "left6",
                    "right_iris_path": "right6",
                },
            ]

        dataset = MockDataset()
        train_idx, val_idx, test_idx = create_stratified_splits(dataset)

        # Build image -> split mapping from results
        image_to_split = {}
        for idx in train_idx:
            s = dataset.samples[idx]
            for path in [
                s["fingerprint_path"],
                s["left_iris_path"],
                s["right_iris_path"],
            ]:
                image_to_split[path] = "train"
        for idx in val_idx:
            s = dataset.samples[idx]
            for path in [
                s["fingerprint_path"],
                s["left_iris_path"],
                s["right_iris_path"],
            ]:
                image_to_split[path] = "val"
        for idx in test_idx:
            s = dataset.samples[idx]
            for path in [
                s["fingerprint_path"],
                s["left_iris_path"],
                s["right_iris_path"],
            ]:
                image_to_split[path] = "test"

        # Verify each included sample has all images in same split
        all_included = train_idx + val_idx + test_idx
        for idx in all_included:
            s = dataset.samples[idx]
            splits = {
                image_to_split[s["fingerprint_path"]],
                image_to_split[s["left_iris_path"]],
                image_to_split[s["right_iris_path"]],
            }
            assert (
                len(splits) == 1
            ), f"Sample {idx} has images in different splits: {splits}"


class TestAssignToSplits:
    """Tests for _assign_to_splits helper function."""

    def test_single_image_goes_to_train(self):
        """Single image should always go to train."""
        import numpy as np

        rng = np.random.default_rng(42)
        result = _assign_to_splits(["img1"], 0.3, 0.5, rng)
        assert result == {"img1": "train"}

    def test_two_images_go_to_train_and_val(self):
        """Two images should go to train and val."""
        import numpy as np

        rng = np.random.default_rng(42)
        result = _assign_to_splits(["img1", "img2"], 0.3, 0.5, rng)
        assert result == {"img1": "train", "img2": "val"}

    def test_multiple_images_distributed_across_splits(self):
        """Multiple images should be distributed across all splits."""
        import numpy as np

        rng = np.random.default_rng(42)
        images = [f"img{i}" for i in range(10)]
        result = _assign_to_splits(images, 0.3, 0.5, rng)

        # Check all images are assigned
        assert set(result.keys()) == set(images)

        # Check all split types are present
        splits = set(result.values())
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits


class TestCreateDataset:
    """Tests for create_dataset function."""

    @patch("biometric_recognition.utils.data_utils.BiometricDataset")
    def test_create_dataset_with_config(self, mock_dataset_class):
        """Test create_dataset uses config values correctly."""
        cfg = OmegaConf.create(
            {
                "data": {
                    "num_people": 10,
                    "fingerprint_size": [128, 128],
                    "iris_size": [64, 64],
                }
            }
        )

        create_dataset(cfg, data_path="/data/path", preload=True)

        mock_dataset_class.assert_called_once_with(
            data_path="/data/path",
            num_people=10,
            fingerprint_size=(128, 128),
            iris_size=(64, 64),
            preload=True,
        )

    @patch("biometric_recognition.utils.data_utils.BiometricDataset")
    def test_create_dataset_with_preload_false(self, mock_dataset_class):
        """Test create_dataset with preload=False."""
        cfg = OmegaConf.create(
            {
                "data": {
                    "num_people": 5,
                    "fingerprint_size": [64, 64],
                    "iris_size": [32, 32],
                }
            }
        )

        create_dataset(cfg, data_path="/local/cache", preload=False)

        mock_dataset_class.assert_called_once_with(
            data_path="/local/cache",
            num_people=5,
            fingerprint_size=(64, 64),
            iris_size=(32, 32),
            preload=False,
        )


class TestCreateDataLoader:
    """Tests for create_data_loader function."""

    def test_create_data_loader_returns_dataloader(self):
        """Test that create_data_loader returns a DataLoader."""
        # Create a mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_dataset.__getitem__ = MagicMock(
            return_value={"data": torch.randn(3, 64, 64)}
        )

        loader = create_data_loader(
            dataset=mock_dataset,
            indices=[0, 1, 2, 3, 4],
            batch_size=2,
            shuffle=True,
            num_workers=0,
        )

        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 2

    def test_create_data_loader_uses_subset(self):
        """Test that create_data_loader creates a subset with given indices."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)

        indices = [10, 20, 30]
        loader = create_data_loader(
            dataset=mock_dataset,
            indices=indices,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        # The subset should have length equal to indices
        assert len(loader.dataset) == len(indices)


class TestCreateDataLoaders:
    """Tests for create_data_loaders function."""

    @patch("biometric_recognition.utils.data_utils.create_dataset")
    @patch("biometric_recognition.utils.data_utils.create_data_loader")
    def test_create_data_loaders_creates_train_and_val(
        self, mock_create_loader, mock_create_dataset
    ):
        """Test create_data_loaders creates train and val loaders."""
        cfg = OmegaConf.create(
            {
                "data": {
                    "path": "/data",
                    "num_people": 5,
                    "fingerprint_size": [64, 64],
                    "iris_size": [32, 32],
                    "batch_size": 8,
                    "num_workers": 0,
                }
            }
        )

        mock_dataset = MagicMock()
        mock_create_dataset.return_value = mock_dataset
        mock_create_loader.return_value = MagicMock(spec=DataLoader)

        train_loader, val_loader, test_loader = create_data_loaders(
            cfg,
            train_indices=[0, 1, 2],
            val_indices=[3, 4],
            data_path="/data",
            test_indices=None,
            preload=True,
        )

        # Should create dataset once
        mock_create_dataset.assert_called_once()

        # Should create train and val loaders (2 calls)
        assert mock_create_loader.call_count == 2

        # Test loader should be None
        assert test_loader is None

    @patch("biometric_recognition.utils.data_utils.create_dataset")
    @patch("biometric_recognition.utils.data_utils.create_data_loader")
    def test_create_data_loaders_creates_test_when_provided(
        self, mock_create_loader, mock_create_dataset
    ):
        """Test create_data_loaders creates test loader when indices provided."""
        cfg = OmegaConf.create(
            {
                "data": {
                    "path": "/data",
                    "num_people": 5,
                    "fingerprint_size": [64, 64],
                    "iris_size": [32, 32],
                    "batch_size": 8,
                    "num_workers": 0,
                }
            }
        )

        mock_dataset = MagicMock()
        mock_create_dataset.return_value = mock_dataset
        mock_create_loader.return_value = MagicMock(spec=DataLoader)

        train_loader, val_loader, test_loader = create_data_loaders(
            cfg,
            train_indices=[0, 1, 2],
            val_indices=[3, 4],
            data_path="/data",
            test_indices=[5, 6],
            preload=True,
        )

        # Should create 3 loaders (train, val, test)
        assert mock_create_loader.call_count == 3

        # Test loader should not be None
        assert test_loader is not None

    @patch("biometric_recognition.utils.data_utils.create_dataset")
    @patch("biometric_recognition.utils.data_utils.create_data_loader")
    def test_create_data_loaders_passes_data_path(
        self, mock_create_loader, mock_create_dataset
    ):
        """Test create_data_loaders passes data_path to create_dataset."""
        cfg = OmegaConf.create(
            {
                "data": {
                    "num_people": 5,
                    "fingerprint_size": [64, 64],
                    "iris_size": [32, 32],
                    "batch_size": 8,
                    "num_workers": 0,
                }
            }
        )

        mock_dataset = MagicMock()
        mock_create_dataset.return_value = mock_dataset
        mock_create_loader.return_value = MagicMock(spec=DataLoader)

        create_data_loaders(
            cfg,
            train_indices=[0, 1],
            val_indices=[2],
            data_path="/local/cache",
        )

        mock_create_dataset.assert_called_once_with(
            cfg, data_path="/local/cache", preload=True
        )
