"""Data loading utilities for training and pipeline tasks."""

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from biometric_recognition.data import BiometricDataset


def create_dataset(
    cfg: DictConfig, preload: bool = True, data_path_override: Optional[str] = None
) -> BiometricDataset:
    """Create a BiometricDataset from config.

    Args:
        cfg: Hydra configuration
        preload: Whether to preload images into memory
        data_path_override: Optional local path to use instead of cfg.data.path
            (useful when data has been pre-cached by data_prep stage)

    Returns:
        Configured BiometricDataset instance
    """
    # Use override path if provided (from data_prep cache), otherwise use config
    data_path = data_path_override or cfg.data.path

    # Convert DictConfig to dict for type compatibility
    config_dict: dict[str, Any] | None = None
    if data_path_override is None:
        from omegaconf import OmegaConf
        config_dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]

    return BiometricDataset(
        data_path=data_path,
        num_people=cfg.data.num_people,
        fingerprint_size=tuple(cfg.data.fingerprint_size),
        iris_size=tuple(cfg.data.iris_size),
        preload=preload,
        config=config_dict,
    )


def _assign_to_splits(
    images: list[str], test_size: float, val_ratio: float, rng: np.random.Generator
) -> dict[str, str]:
    """Assign images to train/val/test splits proportionally."""
    n = len(images)
    if n == 1:
        return {images[0]: "train"}
    if n == 2:
        return {images[0]: "train", images[1]: "val"}

    # Calculate split sizes
    n_val = max(1, int(n * test_size * (1 - val_ratio)))
    n_test = max(1, int(n * test_size * val_ratio))
    n_train = max(1, n - n_val - n_test)

    # Shuffle and assign
    shuffled = rng.permutation(images).tolist()
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    return dict(zip(shuffled, splits))


def create_stratified_splits(
    dataset: BiometricDataset,
    test_size: float = 0.3,
    val_ratio: float = 0.5,
    random_state: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Create stratified train/val/test splits ensuring no image leakage.

    For each person, all unique images are assigned to splits. A sample is only
    included if all its images (fingerprint, left/right iris) are in the same split.

    Args:
        dataset: The dataset to split
        test_size: Fraction for val+test combined (default 0.3 = 30%)
        val_ratio: Ratio of val within the test_size portion (default 0.5 = equal split)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """

    rng = np.random.default_rng(random_state)

    # Collect unique images per person per modality
    person_images: dict[int, dict[str, set[str]]] = {}
    for sample in dataset.samples:
        pid = sample["person_id"]
        if pid not in person_images:
            person_images[pid] = {"fp": set(), "left": set(), "right": set()}
        person_images[pid]["fp"].add(sample["fingerprint_path"])
        person_images[pid]["left"].add(sample["left_iris_path"])
        person_images[pid]["right"].add(sample["right_iris_path"])

    # Assign each image to a split
    image_to_split: dict[str, str] = {}
    for modalities in person_images.values():
        for images in modalities.values():
            assignments = _assign_to_splits(sorted(images), test_size, val_ratio, rng)
            image_to_split.update(assignments)

    # Only include samples where all images are in the same split
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    for idx, sample in enumerate(dataset.samples):
        splits = {
            image_to_split[sample["fingerprint_path"]],
            image_to_split[sample["left_iris_path"]],
            image_to_split[sample["right_iris_path"]],
        }
        if len(splits) == 1:  # All same split
            split = splits.pop()
            {"train": train_indices, "val": val_indices, "test": test_indices}[
                split
            ].append(idx)

    return train_indices, val_indices, test_indices


def create_data_loader(
    dataset: BiometricDataset,
    indices: list[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """Create a DataLoader for a subset of the dataset.

    Args:
        dataset: The full dataset
        indices: Indices to include in this loader
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes (auto-set to 0 in Airflow)

    Returns:
        Configured DataLoader
    """
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def create_data_loaders(
    cfg: DictConfig,
    train_indices: list[int],
    val_indices: list[int],
    test_indices: Optional[list[int]] = None,
    preload: bool = True,
    data_path_override: Optional[str] = None,
) -> tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create data loaders from pre-computed indices.

    Args:
        cfg: Hydra configuration
        train_indices: Training set indices
        val_indices: Validation set indices
        test_indices: Optional test set indices
        preload: Whether to preload images
        data_path_override: Optional local path to use instead of cfg.data.path
            (useful when data has been pre-cached by data_prep stage)

    Returns:
        Tuple of (train_loader, val_loader, test_loader or None)
    """
    dataset = create_dataset(
        cfg, preload=preload, data_path_override=data_path_override
    )

    train_loader = create_data_loader(
        dataset,
        train_indices,
        cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    val_loader = create_data_loader(
        dataset,
        val_indices,
        cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    test_loader = None
    if test_indices is not None:
        test_loader = create_data_loader(
            dataset,
            test_indices,
            cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
        )

    return train_loader, val_loader, test_loader


def save_splits(
    splits: dict[str, list[int]],
    output_path: Path,
) -> str:
    """Save data split indices to JSON file.

    Args:
        splits: Dictionary with train/val/test indices
        output_path: Directory to save the file

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    splits_path = output_path / "data_splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f)

    return str(splits_path)


def load_splits(splits_path: str) -> dict[str, list[int]]:
    """Load data split indices from JSON file.

    Args:
        splits_path: Path to data_splits.json

    Returns:
        Dictionary with train/val/test indices
    """
    with open(splits_path, "r") as f:
        result: dict[str, list[int]] = json.load(f)
        return result
