"""
Airflow DAG for Biometric Recognition Model Training Pipeline.

This DAG orchestrates the training pipeline using BashOperator for full
multiprocessing support (num_workers > 0 in DataLoader).

Pipeline stages:
1. data_prep - Prepare data splits
2. train - Train the model
3. evaluate - Evaluate on test set
4. upload - Upload artifacts to S3
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.models import Variable

# Default arguments for the DAG
default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Configuration - can be overridden via Airflow Variables
CONFIG_PATH = Variable.get(
    "biometric_config_path", default_var="/opt/airflow/configs/config.yaml"
)
BASE_OUTPUT_DIR = Variable.get(
    "biometric_output_dir", default_var="/opt/airflow/outputs"
)

# Python path for the biometric_recognition package
PYTHON_CMD = "python -m"
WORKING_DIR = "/opt/airflow"


# Create the DAG
with DAG(
    dag_id="biometric_training_pipeline",
    default_args=default_args,
    description="Training pipeline for biometric recognition model",
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "training", "biometric"],
) as dag:

    # Use execution timestamp as run_id for unique directories
    run_id = "{{ ts_nodash }}"
    run_dir = f"{BASE_OUTPUT_DIR}/{run_id}"

    # Directory paths for each stage
    data_prep_dir = f"{run_dir}/data_prep"
    checkpoint_dir = f"{run_dir}/checkpoints"
    eval_dir = f"{run_dir}/evaluation"

    # Artifact paths (outputs from each stage)
    config_path = f"{data_prep_dir}/config.yaml"
    splits_path = f"{data_prep_dir}/data_splits.json"
    best_model_path = f"{checkpoint_dir}/best_model.pth"
    history_path = f"{checkpoint_dir}/training_history.json"
    eval_results_path = f"{eval_dir}/evaluation_results.json"

    # Task 1: Data Preparation
    data_prep = BashOperator(
        task_id="data_prep",
        bash_command=f"""
            cd {WORKING_DIR} && \\
            {PYTHON_CMD} biometric_recognition.pipeline.data_prep \\
                --config {CONFIG_PATH} \\
                --output-dir {data_prep_dir}
        """,
        env={"PYTHONPATH": f"{WORKING_DIR}/src"},
    )

    # Task 2: Model Training
    train = BashOperator(
        task_id="train",
        bash_command=f"""
            cd {WORKING_DIR} && \\
            {PYTHON_CMD} biometric_recognition.pipeline.train \\
                --config {config_path} \\
                --splits {splits_path} \\
                --checkpoint-dir {checkpoint_dir}
        """,
        env={"PYTHONPATH": f"{WORKING_DIR}/src"},
        execution_timeout=timedelta(hours=4),  # Training can take a while
    )

    # Task 3: Model Evaluation
    evaluate = BashOperator(
        task_id="evaluate",
        bash_command=f"""
            cd {WORKING_DIR} && \\
            {PYTHON_CMD} biometric_recognition.pipeline.evaluate \\
                --config {config_path} \\
                --splits {splits_path} \\
                --model {best_model_path} \\
                --history {history_path} \\
                --output-dir {eval_dir}
        """,
        env={"PYTHONPATH": f"{WORKING_DIR}/src"},
    )

    # Task 4: Upload Artifacts to S3
    upload = BashOperator(
        task_id="upload",
        bash_command=f"""
            cd {WORKING_DIR} && \\
            {PYTHON_CMD} biometric_recognition.pipeline.upload \\
                --config {config_path} \\
                --model {best_model_path} \\
                --results {eval_results_path} \\
                --plots-dir {eval_dir} \\
                --run-id {run_id}
        """,
        env={"PYTHONPATH": f"{WORKING_DIR}/src"},
    )

    # Define task dependencies
    data_prep >> train >> evaluate >> upload
