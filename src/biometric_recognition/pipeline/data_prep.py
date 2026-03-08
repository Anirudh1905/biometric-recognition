"""Data preparation task for the training pipeline."""

import json
import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from biometric_recognition.utils.data_utils import (
    create_dataset,
    create_stratified_splits,
    save_splits,
)
from biometric_recognition.utils.logging_utils import setup_logging


def prepare_data(cfg: DictConfig, output_dir: str) -> dict:
    """
    Prepare data splits and save indices for downstream tasks.

    Args:
        cfg: Hydra configuration
        output_dir: Directory to save split indices

    Returns:
        dict with paths to saved artifacts and dataset metadata
    """
    logging.info("Starting data preparation...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create dataset (no preload during prep, just indexing)
    dataset = create_dataset(cfg, preload=False)
    logging.info(f"Total dataset size: {len(dataset)} samples")

    # Create stratified splits
    train_indices, val_indices, test_indices = create_stratified_splits(
        dataset, test_size=0.3, val_ratio=0.5, random_state=cfg.seed
    )

    # Save split indices
    splits = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }
    splits_path = save_splits(splits, output_path)

    # Save config for downstream tasks
    config_path = output_path / "config.yaml"
    OmegaConf.save(cfg, config_path)

    metadata = {
        "total_samples": len(dataset),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "test_samples": len(test_indices),
        "num_classes": cfg.data.num_people,
        "splits_path": splits_path,
        "config_path": str(config_path),
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Data preparation complete. Splits saved to {splits_path}")
    logging.info(
        f"Train: {len(train_indices)}, Val: {len(val_indices)}, "
        f"Test: {len(test_indices)}"
    )

    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for splits"
    )
    args = parser.parse_args()

    setup_logging()
    cfg = OmegaConf.load(args.config)
    result = prepare_data(cfg, args.output_dir)
    print(json.dumps(result, indent=2))
