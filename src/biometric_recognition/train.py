"""Training script for multimodal biometric recognition.

This script orchestrates the full training pipeline by reusing
the pipeline module functions: data_prep -> train -> evaluate -> upload
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from biometric_recognition.pipeline import (
    evaluate_model,
    prepare_data,
    train_model,
    upload_artifacts,
)
from biometric_recognition.utils.mlflow_utils import (
    log_artifact,
    log_metrics,
    mlflow_run,
)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function - orchestrates the full pipeline."""
    logging.info("Starting training pipeline...")

    # Use hydra output dir or create temp dir
    output_dir = Path(cfg.output_dir)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with mlflow_run(
        experiment_name="biometric-recognition",
        run_name=f"train-{cfg.data.num_people}p-{cfg.training.epochs}e",
        cfg=cfg,
    ):
        # Step 1: Data preparation
        logging.info("=" * 50)
        logging.info("Step 1: Data Preparation")
        logging.info("=" * 50)
        data_prep_result = prepare_data(cfg, str(output_dir / "data_prep"))

        # Step 2: Training
        logging.info("=" * 50)
        logging.info("Step 2: Training")
        logging.info("=" * 50)
        train_result = train_model(
            config_path=data_prep_result["config_path"],
            splits_path=data_prep_result["splits_path"],
            checkpoint_dir=str(checkpoint_dir),
        )

        # Step 3: Evaluation
        logging.info("=" * 50)
        logging.info("Step 3: Evaluation")
        logging.info("=" * 50)
        eval_result = evaluate_model(
            config_path=data_prep_result["config_path"],
            splits_path=data_prep_result["splits_path"],
            model_path=train_result["best_model_path"],
            history_path=train_result["history_path"],
            output_dir=str(output_dir / "evaluation"),
        )

        # Log final metrics and artifacts to MLflow
        log_metrics(
            {
                "best_val_accuracy": train_result["best_val_accuracy"],
                "test_accuracy": eval_result["test_accuracy"],
                "test_loss": eval_result["test_loss"],
            }
        )
        log_artifact(eval_result["results_path"])
        log_artifact(eval_result["confusion_matrix_path"])
        log_artifact(eval_result["history_plot_path"])

        # Step 4: Upload to S3 (if configured)
        logging.info("=" * 50)
        logging.info("Step 4: Upload to S3")
        logging.info("=" * 50)
        upload_result = upload_artifacts(
            config_path=data_prep_result["config_path"],
            model_path=train_result["best_model_path"],
            evaluation_results_path=eval_result["results_path"],
            plots_dir=str(output_dir / "evaluation"),
        )

        # Summary
        logging.info("=" * 50)
        logging.info("TRAINING COMPLETE!")
        logging.info(
            f"Best validation accuracy: {train_result['best_val_accuracy']:.2f}%"
        )
        logging.info(f"Test accuracy: {eval_result['test_accuracy']:.2f}%")
        logging.info(f"Best model: {train_result['best_model_path']}")
        if upload_result.get("status") == "success":
            logging.info(f"S3 location: {upload_result['base_uri']}")
        logging.info("=" * 50)


if __name__ == "__main__":
    main()
