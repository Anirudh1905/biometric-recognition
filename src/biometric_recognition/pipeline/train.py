"""Training task for the training pipeline."""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from biometric_recognition.utils.data_utils import create_data_loaders, load_splits
from biometric_recognition.utils.device_utils import get_device
from biometric_recognition.utils.logging_utils import setup_logging
from biometric_recognition.utils.mlflow_utils import (
    log_artifact,
    log_metrics,
    log_model,
    mlflow_run,
)
from biometric_recognition.utils.model_utils import create_model
from biometric_recognition.utils.training_utils import train_loop


def train_model(
    config_path: str,
    splits_path: str,
    checkpoint_dir: str,
    resume_from: str | None = None,
    cached_data_path: str | None = None,
) -> dict:
    """
    Train the model using pre-computed data splits.

    Args:
        config_path: Path to saved config.yaml
        splits_path: Path to data_splits.json from data_prep task
        checkpoint_dir: Directory to save model checkpoints
        resume_from: Optional path to checkpoint to resume training from
        cached_data_path: Optional path to pre-cached data from data_prep stage

    Returns:
        dict with training results and paths to saved models
    """
    logging.info("Starting training...")

    cfg = OmegaConf.load(config_path)
    splits = load_splits(splits_path)

    # Log data source
    if cached_data_path:
        logging.info(f"Using pre-cached data from: {cached_data_path}")
    else:
        logging.info(f"Data will be loaded from: {cfg.data.path}")

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    device = get_device(cfg.training.device)
    logging.info(f"Using device: {device}")

    torch.manual_seed(cfg.seed)

    # Create data loaders (use cached path if provided)
    train_loader, val_loader, _ = create_data_loaders(
        cfg,
        splits["train_indices"],
        splits["val_indices"],
        preload=cfg.data.preload_images,
        data_path_override=cached_data_path,
    )

    # Create model
    model = create_model(cfg, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # Handle resume
    start_epoch = 0
    best_val_acc = 0.0

    if resume_from and Path(resume_from).exists():
        logging.info(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_acc = checkpoint.get("val_accuracy", 0.0)

    # Train
    history = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=cfg.training.epochs,
        checkpoint_dir=checkpoint_path,
        cfg=cfg,
        start_epoch=start_epoch,
        best_val_acc=best_val_acc,
    )

    # Save training history
    history["total_epochs"] = cfg.training.epochs
    history_path = checkpoint_path / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Log model to MLflow
    log_model(model, "model")

    result = {
        "best_model_path": str(checkpoint_path / "best_model.pth"),
        "final_model_path": str(checkpoint_path / "final_model.pth"),
        "history_path": str(history_path),
        "best_val_accuracy": history["best_val_accuracy"],
        "final_val_accuracy": history["val_accuracies"][-1],
    }

    logging.info(
        f"Training complete. Best val accuracy: {history['best_val_accuracy']:.2f}%"
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--splits", required=True, help="Path to data_splits.json")
    parser.add_argument("--checkpoint-dir", required=True, help="Checkpoint directory")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument(
        "--metadata", default=None, help="Path to metadata.json from data_prep"
    )
    parser.add_argument("--run-id", default=None, help="Run ID for experiment tracking")
    args = parser.parse_args()

    setup_logging()
    cfg = OmegaConf.load(args.config)

    # Read cached data path from metadata if provided
    data_path = None
    if args.metadata:
        with open(args.metadata) as f:
            data_path = json.load(f).get("cached_data_path")

    with mlflow_run(
        experiment_name=cfg.mlflow.experiment_name,
        run_name=f"train-{args.run_id}" if args.run_id else None,
        cfg=cfg,
        tags={"stage": "training", "run_id": args.run_id} if args.run_id else None,
    ):
        result = train_model(
            args.config, args.splits, args.checkpoint_dir, args.resume, data_path
        )
        log_metrics(
            {
                "best_val_accuracy": result["best_val_accuracy"],
                "final_val_accuracy": result["final_val_accuracy"],
            }
        )
        log_artifact(result["best_model_path"])
        log_artifact(result["history_path"])

    print(json.dumps(result, indent=2))
