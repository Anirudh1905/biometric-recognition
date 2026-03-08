"""MLflow tracking utilities."""

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import mlflow
from mlflow.models import infer_signature
from omegaconf import DictConfig

__all__ = [
    "get_tracking_uri",
    "flatten_config",
    "mlflow_run",
    "log_metrics",
    "log_artifact",
    "log_model",
    "infer_signature",
]


def get_tracking_uri() -> str:
    """Get MLflow tracking URI from environment or default."""
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def flatten_config(cfg: DictConfig, parent_key: str = "") -> dict[str, Any]:
    """Flatten nested OmegaConf config for MLflow params."""
    items: dict[str, Any] = {}
    for key, value in cfg.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, DictConfig):
            items.update(flatten_config(value, new_key))
        else:
            items[new_key] = value
    return items


@contextmanager
def mlflow_run(
    experiment_name: str,
    run_name: str | None = None,
    cfg: DictConfig | None = None,
    tags: dict[str, str] | None = None,
) -> Generator[mlflow.ActiveRun | None, None, None]:
    """Context manager for MLflow run with automatic param logging.

    Args:
        experiment_name: MLflow experiment name
        run_name: Optional run name
        cfg: Optional Hydra/OmegaConf config to log as params
        tags: Optional tags to add to the run

    Yields:
        Active MLflow run or None if MLflow is unavailable
    """
    try:
        mlflow.set_tracking_uri(get_tracking_uri())
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name) as run:
            if tags:
                mlflow.set_tags(tags)
            if cfg:
                params = flatten_config(cfg)
                mlflow.log_params(params)
            yield run
    except Exception as e:
        logging.warning(
            f"MLflow tracking unavailable: {e}. Continuing without tracking."
        )
        yield None


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics to MLflow if active run exists."""
    if mlflow.active_run():
        mlflow.log_metrics(metrics, step=step)


def log_artifact(path: str | Path) -> None:
    """Log artifact to MLflow if active run exists."""
    if mlflow.active_run():
        try:
            mlflow.log_artifact(str(path))
        except Exception as e:
            logging.warning(f"Failed to log artifact {path}: {e}")


def log_model(
    model: Any,
    artifact_path: str = "model",
    input_example: Any | None = None,
    signature: Any | None = None,
) -> None:
    """Log PyTorch model to MLflow if active run exists.

    Args:
        model: PyTorch model to log
        artifact_path: Path within the artifact store
        input_example: Optional example input for signature inference
        signature: Optional pre-computed model signature
    """
    if mlflow.active_run():
        try:
            mlflow.pytorch.log_model(
                model,
                artifact_path,
                signature=signature,
                input_example=input_example,
            )
        except Exception as e:
            logging.warning(f"Failed to log model: {e}")
