"""Training task for the training pipeline."""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from biometric_recognition.utils import (
    create_data_loaders,
    create_model,
    get_device,
    load_splits,
    train_loop,
)


def train_model(
    config_path: str,
    splits_path: str,
    checkpoint_dir: str,
    resume_from: str | None = None,
) -> dict:
    """
    Train the model using pre-computed data splits.

    Args:
        config_path: Path to saved config.yaml
        splits_path: Path to data_splits.json from data_prep task
        checkpoint_dir: Directory to save model checkpoints
        resume_from: Optional path to checkpoint to resume training from

    Returns:
        dict with training results and paths to saved models
    """
    logging.info("Starting training...")

    cfg = OmegaConf.load(config_path)
    splits = load_splits(splits_path)

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    device = get_device(cfg.training.device)
    logging.info(f"Using device: {device}")

    torch.manual_seed(cfg.seed)

    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(
        cfg,
        splits["train_indices"],
        splits["val_indices"],
        preload=cfg.data.preload_images,
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
    args = parser.parse_args()

    result = train_model(args.config, args.splits, args.checkpoint_dir, args.resume)
    print(json.dumps(result, indent=2))
