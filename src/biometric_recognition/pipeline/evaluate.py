"""Evaluation task for the training pipeline."""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from biometric_recognition.utils import (
    get_device,
    load_splits,
    create_data_loaders,
    create_model,
    validate,
    plot_training_history,
    plot_confusion_matrix,
    get_classification_report,
)


def evaluate_model(
    config_path: str,
    splits_path: str,
    model_path: str,
    history_path: str,
    output_dir: str,
) -> dict:
    """
    Evaluate the trained model on the test set.

    Args:
        config_path: Path to config.yaml
        splits_path: Path to data_splits.json
        model_path: Path to trained model checkpoint
        history_path: Path to training_history.json
        output_dir: Directory to save evaluation results

    Returns:
        dict with evaluation metrics and paths to artifacts
    """
    logging.info("Starting model evaluation...")

    cfg = OmegaConf.load(config_path)
    splits = load_splits(splits_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = get_device(cfg.training.device)
    logging.info(f"Using device: {device}")

    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = create_model(cfg, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create test loader
    _, _, test_loader = create_data_loaders(
        cfg,
        splits["train_indices"],  # Not used but required
        splits["val_indices"],    # Not used but required
        splits["test_indices"],
        preload=cfg.data.preload_images,
    )

    # Evaluate on test set using validate with collect_predictions=True
    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy, all_preds, all_labels = validate(
        model, test_loader, criterion, device, collect_predictions=True
    )

    logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Generate classification report
    class_report = get_classification_report(all_labels, all_preds, output_dict=True)

    # Plot training history
    with open(history_path, "r") as f:
        history = json.load(f)

    history_plot_path = plot_training_history(
        history["train_losses"],
        history["val_losses"],
        history["val_accuracies"],
        output_path / "training_history.png",
    )

    # Plot confusion matrix
    cm_plot_path = plot_confusion_matrix(
        all_labels, all_preds, cfg.data.num_people, output_path / "confusion_matrix.png"
    )

    # Compile results
    results = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_samples": len(all_labels),
        "classification_report": class_report,
        "training_history": history,
        "model_checkpoint_epoch": checkpoint.get("epoch", -1),
        "model_val_accuracy": checkpoint.get("val_accuracy", -1),
    }

    # Save results
    results_path = output_path / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    output = {
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "results_path": str(results_path),
        "history_plot_path": history_plot_path,
        "confusion_matrix_path": cm_plot_path,
        "best_val_accuracy": history.get("best_val_accuracy", -1),
    }

    logging.info(f"Evaluation complete. Results saved to {results_path}")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--splits", required=True, help="Path to data_splits.json")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--history", required=True, help="Path to training_history.json")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    result = evaluate_model(
        args.config, args.splits, args.model, args.history, args.output_dir
    )
    print(json.dumps(result, indent=2))
