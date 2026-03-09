"""Training loop utilities."""

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from biometric_recognition.utils.mlflow_utils import log_metrics
from biometric_recognition.utils.model_utils import (
    move_batch_to_device,
    save_checkpoint,
)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number (0-indexed)

    Returns:
        Tuple of (average_loss, accuracy_percentage)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    for batch in pbar:
        batch_device = move_batch_to_device(batch, device)

        optimizer.zero_grad()
        outputs = model(batch_device)
        loss = criterion(outputs, batch_device["label"])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_device["label"].size(0)
        correct += (predicted == batch_device["label"]).sum().item()

        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"}
        )

    return total_loss / len(train_loader), 100.0 * correct / total


def validate(
    model: nn.Module,
    data_loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    epoch: Optional[int] = None,
    collect_predictions: bool = False,
) -> tuple[float, float, list[int], list[int]]:
    """Validate or evaluate the model.

    Args:
        model: The model to validate/evaluate
        data_loader: Data loader (validation or test)
        criterion: Loss function
        device: Device to run on
        epoch: Current epoch number (0-indexed), used for progress bar label
        collect_predictions: If True, collect predictions and labels for metrics

    Returns:
        Tuple of (avg_loss, accuracy, predictions, labels).
        If collect_predictions=False, predictions and labels will be empty lists.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    desc = f"Epoch {epoch+1} [Val]" if epoch is not None else "Evaluating"

    with torch.no_grad():
        pbar = tqdm(data_loader, desc=desc)
        for batch in pbar:
            batch_device = move_batch_to_device(batch, device)

            outputs = model(batch_device)
            loss = criterion(outputs, batch_device["label"])

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_device["label"].size(0)
            correct += (predicted == batch_device["label"]).sum().item()

            if collect_predictions:
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(batch_device["label"].cpu().numpy().tolist())

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"}
            )

    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy, all_preds, all_labels


def train_loop(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    checkpoint_dir: Path,
    cfg: DictConfig | None = None,
    start_epoch: int = 0,
    best_val_acc: float = 0.0,
) -> dict[str, Any]:
    """Run the full training loop.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epochs: Total number of epochs
        checkpoint_dir: Directory to save checkpoints
        cfg: Optional config to save with checkpoints
        start_epoch: Starting epoch (for resume)
        best_val_acc: Best validation accuracy so far (for resume)

    Returns:
        Dictionary with training history and best accuracy
    """
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(start_epoch, epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        train_losses.append(train_loss)

        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device, epoch)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        logging.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Log to MLflow (no-op if no active run)
        log_metrics(
            {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            },
            step=epoch,
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                path=checkpoint_dir / "best_model.pth",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                val_accuracy=val_acc,
                cfg=cfg,
            )
            logging.info(f"New best model saved with val accuracy: {val_acc:.2f}%")

    # Save final model
    save_checkpoint(
        path=checkpoint_dir / "final_model.pth",
        epoch=epochs - 1,
        model=model,
        optimizer=optimizer,
        val_accuracy=val_accuracies[-1],
        cfg=cfg,
        train_losses=train_losses,
        val_losses=val_losses,
        val_accuracies=val_accuracies,
    )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "best_val_accuracy": best_val_acc,
    }
