"""Metrics and plotting utilities."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def plot_training_history(
    train_losses: list[float],
    val_losses: list[float],
    val_accuracies: list[float],
    save_path: str | Path,
) -> str:
    """Plot and save training history.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        val_accuracies: List of validation accuracies per epoch
        save_path: Path to save the plot

    Returns:
        Path to saved plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot losses
    ax1.plot(train_losses, label="Train Loss", color="blue")
    ax1.plot(val_losses, label="Validation Loss", color="red")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy
    ax2.plot(val_accuracies, label="Validation Accuracy", color="green")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logging.info(f"Training history saved to {save_path}")
    return str(save_path)


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    num_classes: int,
    save_path: str | Path,
) -> str:
    """Plot and save confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        save_path: Path to save the plot

    Returns:
        Path to saved plot
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations for smaller matrices
    if num_classes <= 20:
        thresh = cm.max() / 2.0
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logging.info(f"Confusion matrix saved to {save_path}")
    return str(save_path)


def get_classification_report(
    y_true: list[int],
    y_pred: list[int],
    output_dict: bool = True,
) -> dict | str:
    """Generate classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_dict: If True, return as dict; if False, return as string

    Returns:
        Classification report as dict or string
    """
    return classification_report(
        y_true, y_pred, output_dict=output_dict, zero_division=0
    )
