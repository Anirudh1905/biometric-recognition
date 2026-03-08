"""Data loading utilities for training and pipeline tasks."""

import json
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from biometric_recognition.data import BiometricDataset


def create_dataset(cfg: DictConfig, preload: bool = True) -> BiometricDataset:
    """Create a BiometricDataset from config.

    Args:
        cfg: Hydra configuration
        preload: Whether to preload images into memory

    Returns:
        Configured BiometricDataset instance
    """
    return BiometricDataset(
        data_path=cfg.data.path,
        num_people=cfg.data.num_people,
        fingerprint_size=tuple(cfg.data.fingerprint_size),
        iris_size=tuple(cfg.data.iris_size),
        preload=preload,
        config=cfg,
    )


def create_stratified_splits(
    dataset: BiometricDataset,
    test_size: float = 0.3,
    val_ratio: float = 0.5,
    random_state: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Create stratified train/val/test splits.

    Args:
        dataset: The dataset to split
        test_size: Fraction for val+test combined (default 0.3 = 30%)
        val_ratio: Ratio of val within the test_size portion (default 0.5 = equal split)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    person_ids = [sample["person_id"] for sample in dataset.samples]

    # First split: train vs (val+test)
    train_indices, temp_indices = train_test_split(
        range(len(dataset)),
        test_size=test_size,
        stratify=person_ids,
        random_state=random_state,
    )

    # Second split: val vs test
    temp_person_ids = [person_ids[i] for i in temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=val_ratio,
        stratify=temp_person_ids,
        random_state=random_state,
    )

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
) -> tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create data loaders from pre-computed indices.

    Args:
        cfg: Hydra configuration
        train_indices: Training set indices
        val_indices: Validation set indices
        test_indices: Optional test set indices
        preload: Whether to preload images

    Returns:
        Tuple of (train_loader, val_loader, test_loader or None)
    """
    dataset = create_dataset(cfg, preload=preload)

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
        return json.load(f)
