"""Evaluation task for the training pipeline."""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from biometric_recognition.utils.data_utils import create_data_loaders, load_splits
from biometric_recognition.utils.device_utils import get_device
from biometric_recognition.utils.logging_utils import setup_logging
from biometric_recognition.utils.metrics_utils import (
    get_classification_report,
    plot_confusion_matrix,
    plot_training_history,
)
from biometric_recognition.utils.mlflow_utils import (
    log_artifact,
    log_metrics,
    mlflow_run,
)
from biometric_recognition.utils.model_utils import create_model
from biometric_recognition.utils.training_utils import validate


def evaluate_model(
    config_path: str,
    splits_path: str,
    model_path: str,
    history_path: str,
    output_dir: str,
    cached_data_path: str | None = None,
) -> dict:
    """
    Evaluate the trained model on the test set.

    Args:
        config_path: Path to config.yaml
        splits_path: Path to data_splits.json
        model_path: Path to trained model checkpoint
        history_path: Path to training_history.json
        output_dir: Directory to save evaluation results
        cached_data_path: Optional path to pre-cached data from data_prep stage

    Returns:
        dict with evaluation metrics and paths to artifacts
    """
    logging.info("Starting model evaluation...")

    cfg: DictConfig = OmegaConf.load(config_path)  # type: ignore[assignment]
    splits = load_splits(splits_path)

    # Resolve data path: use cached path from data_prep or fall back to config
    data_path = cached_data_path if cached_data_path else cfg.data.path
    logging.info(f"Using data from: {data_path}")

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
        splits["val_indices"],  # Not used but required
        data_path=data_path,
        test_indices=splits["test_indices"],
        preload=cfg.data.preload_images,
    )

    # Evaluate on test set using validate with collect_predictions=True
    criterion = nn.CrossEntropyLoss()
    if test_loader is None:
        raise ValueError("Test loader is None - test_indices must be provided")
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
    parser.add_argument(
        "--history", required=True, help="Path to training_history.json"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--metadata", default=None, help="Path to metadata.json from data_prep"
    )
    parser.add_argument("--run-id", default=None, help="Run ID for experiment tracking")
    args = parser.parse_args()

    setup_logging()
    cfg: DictConfig = OmegaConf.load(args.config)  # type: ignore[assignment]

    # Read cached data path from metadata if provided
    data_path = None
    if args.metadata:
        with open(args.metadata) as f:
            data_path = json.load(f).get("cached_data_path")

    with mlflow_run(
        experiment_name=cfg.mlflow.experiment_name,
        run_name=f"evaluate-{args.run_id}" if args.run_id else None,
        cfg=cfg,
        tags={"stage": "evaluation", "run_id": args.run_id} if args.run_id else None,
    ):
        result = evaluate_model(
            args.config,
            args.splits,
            args.model,
            args.history,
            args.output_dir,
            data_path,
        )
        log_metrics(
            {
                "test_accuracy": result["test_accuracy"],
                "test_loss": result["test_loss"],
                "best_val_accuracy": result["best_val_accuracy"],
            }
        )
        log_artifact(result["results_path"])
        log_artifact(result["history_plot_path"])
        log_artifact(result["confusion_matrix_path"])

    print(json.dumps(result, indent=2))
