"""Training pipeline tasks for Airflow orchestration."""

from biometric_recognition.pipeline.data_prep import prepare_data
from biometric_recognition.pipeline.evaluate import evaluate_model
from biometric_recognition.pipeline.train import train_model
from biometric_recognition.pipeline.upload import upload_artifacts

__all__ = ["prepare_data", "train_model", "evaluate_model", "upload_artifacts"]
