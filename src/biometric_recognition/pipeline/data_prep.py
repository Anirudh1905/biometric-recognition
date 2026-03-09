"""Data preparation task for the training pipeline."""

import json
import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from biometric_recognition.utils.aws_utils import get_data_path
from biometric_recognition.utils.data_utils import (
    create_dataset,
    create_stratified_splits,
    save_splits,
)
from biometric_recognition.utils.logging_utils import setup_logging


def prepare_data(cfg: DictConfig, output_dir: str) -> dict:
    """
    Download data (if S3), cache locally, create splits, and save indices.

    This stage handles all data downloading/caching so that downstream tasks
    (training, evaluation) can load directly from the local cache.

    Args:
        cfg: Hydra configuration
        output_dir: Directory to save split indices and metadata

    Returns:
        dict with paths to saved artifacts, dataset metadata, and cached data path
    """
    logging.info("Starting data preparation...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Download and cache data (handles S3 or validates local path)
    logging.info("=" * 40)
    logging.info("Step 1: Downloading and caching data...")
    logging.info("=" * 40)
    cached_data_path = get_data_path(
        path=cfg.data.path,
        cache_dir=cfg.data.get("cache_dir"),
        aws_region=cfg.get("aws", {}).get("region", "us-east-1"),
    )
    logging.info(f"Data available at: {cached_data_path}")

    # Step 2: Create dataset from cached path (no preload, just indexing)
    logging.info("=" * 40)
    logging.info("Step 2: Indexing dataset...")
    logging.info("=" * 40)
    dataset = create_dataset(cfg, preload=False, data_path_override=cached_data_path)
    logging.info(f"Total dataset size: {len(dataset)} samples")

    # Step 3: Create stratified splits
    logging.info("=" * 40)
    logging.info("Step 3: Creating stratified splits...")
    logging.info("=" * 40)
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
        "cached_data_path": cached_data_path,  # New: path to cached data
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Data preparation complete. Splits saved to {splits_path}")
    logging.info(f"Data cached at: {cached_data_path}")
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
