"""Upload artifacts task for the training pipeline."""

import datetime
import json
import logging
from pathlib import Path

from omegaconf import OmegaConf

from biometric_recognition.utils.aws_utils import S3Utils
from biometric_recognition.utils.logging_utils import setup_logging
from biometric_recognition.utils.mlflow_utils import log_artifact, mlflow_run


def upload_artifacts(
    config_path: str,
    model_path: str,
    evaluation_results_path: str,
    plots_dir: str,
    run_id: str | None = None,
) -> dict:
    """
    Upload trained model and artifacts to S3.

    Args:
        config_path: Path to config.yaml
        model_path: Path to model checkpoint to upload
        evaluation_results_path: Path to evaluation_results.json
        plots_dir: Directory containing plots
        run_id: Optional run identifier (defaults to timestamp)

    Returns:
        dict with S3 URIs of uploaded artifacts
    """
    logging.info("Starting artifact upload to S3...")

    cfg = OmegaConf.load(config_path)

    s3_config = cfg.get("s3", {})
    if not s3_config.get("model_bucket"):
        logging.warning("S3 model storage not configured. Skipping upload.")
        return {"status": "skipped", "reason": "S3 not configured"}

    # Generate run ID if not provided
    if run_id is None:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        s3_utils = S3Utils(cfg.aws.region)
        bucket = s3_config["model_bucket"]
        prefix = s3_config.get("model_prefix", "models/").rstrip("/")

        uploaded = {}

        # Upload model
        model_s3_uri = f"s3://{bucket}/{prefix}/{run_id}/model.pth"
        s3_utils.upload_to_s3(model_path, model_s3_uri)
        uploaded["model"] = model_s3_uri
        logging.info(f"Uploaded model to {model_s3_uri}")

        # Upload evaluation results
        results_s3_uri = f"s3://{bucket}/{prefix}/{run_id}/evaluation_results.json"
        s3_utils.upload_to_s3(evaluation_results_path, results_s3_uri)
        uploaded["evaluation_results"] = results_s3_uri
        logging.info(f"Uploaded evaluation results to {results_s3_uri}")

        # Upload config
        config_s3_uri = f"s3://{bucket}/{prefix}/{run_id}/config.yaml"
        s3_utils.upload_to_s3(config_path, config_s3_uri)
        uploaded["config"] = config_s3_uri
        logging.info(f"Uploaded config to {config_s3_uri}")

        # Upload plots
        plots_path = Path(plots_dir)
        for plot_file in plots_path.glob("*.png"):
            plot_s3_uri = f"s3://{bucket}/{prefix}/{run_id}/{plot_file.name}"
            s3_utils.upload_to_s3(str(plot_file), plot_s3_uri)
            uploaded[plot_file.stem] = plot_s3_uri
            logging.info(f"Uploaded {plot_file.name} to {plot_s3_uri}")

        # Create and upload manifest
        manifest = {
            "run_id": run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "artifacts": uploaded,
        }

        manifest_path = plots_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        manifest_s3_uri = f"s3://{bucket}/{prefix}/{run_id}/manifest.json"
        s3_utils.upload_to_s3(str(manifest_path), manifest_s3_uri)
        uploaded["manifest"] = manifest_s3_uri

        # Log manifest to MLflow (contains all S3 URIs)
        log_artifact(str(manifest_path))

        logging.info("=" * 50)
        logging.info("ARTIFACTS UPLOADED TO S3!")
        logging.info(f"Run ID: {run_id}")
        logging.info(f"Base path: s3://{bucket}/{prefix}/{run_id}/")
        logging.info("=" * 50)

        return {
            "status": "success",
            "run_id": run_id,
            "artifacts": uploaded,
            "base_uri": f"s3://{bucket}/{prefix}/{run_id}/",
        }

    except Exception as e:
        logging.error(f"Failed to upload artifacts to S3: {e}")
        return {"status": "failed", "error": str(e)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--results", required=True, help="Path to evaluation_results.json"
    )
    parser.add_argument("--plots-dir", required=True, help="Directory with plots")
    parser.add_argument("--run-id", default=None, help="Optional run ID")
    args = parser.parse_args()

    setup_logging()
    cfg = OmegaConf.load(args.config)

    with mlflow_run(
        experiment_name=cfg.mlflow.experiment_name,
        run_name=f"upload-{args.run_id}" if args.run_id else None,
        tags={"stage": "upload", "run_id": args.run_id} if args.run_id else None,
    ):
        result = upload_artifacts(
            args.config, args.model, args.results, args.plots_dir, args.run_id
        )

    print(json.dumps(result, indent=2))
