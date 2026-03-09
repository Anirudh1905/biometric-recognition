# Multimodal Biometric Recognition System

A PyTorch-based multimodal biometric recognition system that uses fingerprints and iris images for person identification, with MLflow experiment tracking, Airflow orchestration, and Kubernetes deployment.

## Features

- **Multimodal Approach**: Combines fingerprint and iris (left & right) biometric modalities
- **PyTorch Implementation**: Modern deep learning framework with GPU support
- **MLflow Integration**: Experiment tracking, model versioning, and artifact management
- **Airflow Orchestration**: Automated training pipelines with task dependencies
- **Kubernetes Ready**: Production deployment with health checks and auto-scaling
- **Terraform Infrastructure**: Infrastructure as Code for AWS resources (ECR, S3)
- **Hydra Configuration**: Flexible configuration management
- **S3 Integration**: Cloud storage for datasets and models

## System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Biometric Recognition System                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  Fingerprint │    │  Left Iris   │    │  Right Iris  │   Input Images    │
│  │   (128x128)  │    │   (64x64)    │    │   (64x64)    │                   │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                   │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ┌──────────────┐    ┌─────────────────────────────────┐                    │
│  │ MobileNetV2  │    │     Shared Iris CNN Branch      │   Feature          │
│  │  (pretrained)│    │   (2 Conv + GlobalAvgPool)      │   Extraction       │
│  │  → 1280 dim  │    │        → 32 dim each            │                    │
│  └──────┬───────┘    └──────┬───────────────┬──────────┘                    │
│         │                   │               │                                │
│         └───────────────────┼───────────────┘                               │
│                             ▼                                                │
│                   ┌─────────────────┐                                        │
│                   │  Fusion Module  │   Feature Fusion                       │
│                   │ (1344 → 128 dim)│   (Concatenate + Dense)                │
│                   └────────┬────────┘                                        │
│                            ▼                                                 │
│                   ┌─────────────────┐                                        │
│                   │   Classifier    │   Classification                       │
│                   │ (128 → N classes)│                                       │
│                   └────────┬────────┘                                        │
│                            ▼                                                 │
│                     Person Identity                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Model Components

| Component              | Architecture                        | Output Dimension    |
| ---------------------- | ----------------------------------- | ------------------- |
| **Fingerprint Branch** | MobileNetV2 (pretrained, frozen)    | 1280                |
| **Iris Branch**        | 2x Conv2D + MaxPool + GlobalAvgPool | 32 (shared for L/R) |
| **Fusion Module**      | Linear(1344→128) + ReLU + Dropout   | 128                 |
| **Classifier**         | Linear(128→num_classes)             | num_classes         |

## Project Structure

```text
biometric-recognition/
├── src/biometric_recognition/
│   ├── api/                    # FastAPI server
│   │   ├── serve.py           # API endpoints
│   │   └── schema.py          # Request/response models
│   ├── data/                   # Data loading
│   │   ├── dataset.py         # BiometricDataset class
│   │   └── dataset_ray.py     # Ray-based dataset loader
│   ├── models/                 # Model architectures
│   │   ├── branches.py        # Fingerprint, Iris, Fusion modules
│   │   └── multimodal_model.py # Main model class
│   ├── pipeline/               # Training pipeline tasks
│   │   ├── data_prep.py       # Data preparation
│   │   ├── train.py           # Model training
│   │   ├── evaluate.py        # Model evaluation
│   │   └── upload.py          # S3 upload
│   ├── utils/                  # Utilities
│   │   ├── aws_utils.py       # AWS/S3 utilities
│   │   ├── data_utils.py      # Data utilities
│   │   ├── device_utils.py    # Device detection (CPU/GPU/MPS)
│   │   ├── image_utils.py     # Image processing utilities
│   │   ├── logging_utils.py   # Logging configuration
│   │   ├── metrics_utils.py   # Evaluation metrics
│   │   ├── mlflow_utils.py    # MLflow tracking helpers
│   │   ├── model_utils.py     # Model utilities
│   │   └── training_utils.py  # Training loop
│   └── train.py               # Main training script
├── tests/                      # Unit tests
├── airflow/
│   ├── Dockerfile             # Airflow Docker image
│   └── dags/
│       └── biometric_training_dag.py  # Airflow DAG
├── k8s/
│   ├── deployment.yaml        # Kubernetes deployment
│   └── service.yaml           # Kubernetes service
├── configs/
│   └── config.yaml            # Hydra configuration
├── terraform/                  # Infrastructure as Code
│   ├── providers.tf           # AWS provider configuration
│   ├── variables.tf           # Input variables
│   ├── ecr.tf                 # ECR repository
│   ├── s3.tf                  # S3 bucket for artifacts
│   └── outputs.tf             # Output values
├── Dockerfile                  # API Docker image
└── docker-compose.yml         # Docker services (Airflow, MLflow, Postgres)
```

## Installation

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- AWS CLI (for S3 features)
- Terraform >= 1.0 (for infrastructure provisioning)
- kubectl (for Kubernetes deployment)

### Local Development

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Clone and install
git clone <repository-url>
cd biometric-recognition
poetry install

# Activate environment
poetry shell
```

## Training Pipeline

### Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌─────────┐│
│  │ Data Prep │───▶│   Train   │───▶│ Evaluate  │───▶│ Upload  ││
│  └───────────┘    └───────────┘    └───────────┘    └─────────┘│
│       │                │                │                │      │
│       ▼                ▼                ▼                ▼      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                     MLflow Tracking                          ││
│  │  • Parameters  • Metrics  • Artifacts  • Models              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline Stages

| Stage         | Description                             | Outputs                                      |
| ------------- | --------------------------------------- | -------------------------------------------- |
| **data_prep** | Create stratified train/val/test splits | `data_splits.json`, `config.yaml`            |
| **train**     | Train model with validation             | `best_model.pth`, `training_history.json`    |
| **evaluate**  | Evaluate on test set, generate plots    | `evaluation_results.json`, confusion matrix  |
| **upload**    | Upload artifacts to S3                  | S3 URIs                                      |

### Run Training Locally

```bash
# Using Poetry script (recommended)
poetry run train

# Or run directly
python src/biometric_recognition/train.py

# With custom parameters
python src/biometric_recognition/train.py training.epochs=20 data.batch_size=16

# Specify device
python src/biometric_recognition/train.py training.device=cuda
```

## MLflow Integration

MLflow provides experiment tracking, model versioning, and artifact storage.

### MLflow Architecture

```text
┌─────────────────────────────────────────────────────┐
│                   MLflow Server                      │
├─────────────────────────────────────────────────────┤
│  Tracking URI: <http://localhost:5000>              │
│                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ Experiments │  │    Runs     │  │  Artifacts  │ │
│  │             │  │             │  │             │ │
│  │ biometric-  │  │ • params    │  │ • models    │ │
│  │ recognition │  │ • metrics   │  │ • plots     │ │
│  │             │  │ • tags      │  │ • history   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
│                                                      │
│  Backend: SQLite    Artifacts: Local filesystem     │
└─────────────────────────────────────────────────────┘
```

### Tracked Metrics

- **Training**: `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy` (per epoch)
- **Final**: `best_val_accuracy`, `test_accuracy`, `test_loss`

### Logged Artifacts

- Model checkpoints (`best_model.pth`)
- Training history (`training_history.json`)
- Confusion matrix plot
- Training/validation curves

### Usage

```python
from biometric_recognition.utils.mlflow_utils import mlflow_run, log_metrics, log_artifact

with mlflow_run(experiment_name="biometric-recognition", run_name="my-run", cfg=cfg):
    # Training code...
    log_metrics({"accuracy": 0.95, "loss": 0.05})
    log_artifact("path/to/model.pth")
```

### Access MLflow UI

```bash
# Start services
docker-compose up -d mlflow

# Open browser
open http://localhost:5000
```

## Airflow Orchestration

Apache Airflow orchestrates the training pipeline as a DAG (Directed Acyclic Graph).

### DAG Structure

```text
┌─────────────────────────────────────────────────────────────────┐
│              biometric_training_pipeline DAG                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────┐                                                  │
│  │ data_prep │  Prepare data splits                             │
│  └─────┬─────┘                                                  │
│        │                                                         │
│        ▼                                                         │
│  ┌───────────┐                                                  │
│  │   train   │  Train model (4hr timeout)                       │
│  └─────┬─────┘                                                  │
│        │                                                         │
│        ▼                                                         │
│  ┌───────────┐                                                  │
│  │ evaluate  │  Evaluate on test set                            │
│  └─────┬─────┘                                                  │
│        │                                                         │
│        ▼                                                         │
│  ┌───────────┐                                                  │
│  │  upload   │  Upload to S3                                    │
│  └───────────┘                                                  │
│                                                                  │
│  Schedule: Manual trigger only                                   │
│  Tags: ml, training, biometric                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Start Airflow

```bash
# Start all services (Postgres, MLflow, Airflow)
docker-compose up -d

# Access Airflow UI
open http://localhost:8080
# Login: admin / admin
```

### Airflow Variables

| Variable                | Default                            | Description         |
| ----------------------- | ---------------------------------- | ------------------- |
| `biometric_config_path` | `/opt/airflow/configs/config.yaml` | Path to config file |
| `biometric_output_dir`  | `/opt/airflow/outputs`             | Output directory    |

### Trigger Training

```bash
# Via CLI
docker-compose exec airflow-webserver airflow dags trigger biometric_training_pipeline

# Or use the Airflow UI
```

## API Server

FastAPI-based REST API for real-time inference.

### Endpoints

| Endpoint      | Method | Description                   |
| ------------- | ------ | ----------------------------- |
| `/health`     | GET    | Health check, model status    |
| `/predict`    | POST   | Predict from uploaded files   |
| `/model/info` | GET    | Model metadata                |
| `/docs`       | GET    | Interactive API documentation |

### Run API Server

```bash
# Using Poetry script (recommended)
poetry run serve

# Or run directly
poetry run python -m biometric_recognition.api.serve

# Using Docker
docker-compose --profile prod up -d

# Environment variables
export MODEL_PATH="s3://biometric-recognition-artifacts/biometric_model/model.pth"
export NUM_CLASSES=45
```

### Example Request

```python
import requests

files = {
    "fingerprint": open("fingerprint.bmp", "rb"),
    "left_iris": open("left_iris.bmp", "rb"),
    "right_iris": open("right_iris.bmp", "rb"),
}

response = requests.post("http://localhost:8000/predict", files=files)
result = response.json()

print(f"Predicted class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.4f}")
```

### Response Format

```json
{
  "predicted_class": 12,
  "confidence": 0.9542,
  "top_k_predictions": [
    {"class": 12, "probability": 0.9542},
    {"class": 7, "probability": 0.0231},
    {"class": 3, "probability": 0.0089}
  ]
}
```

## Kubernetes Deployment

Production deployment on Kubernetes with health checks and resource management.

### Cluster Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Service (NodePort)                        ││
│  │                    Port: 30080 → 8000                        ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                      Deployment                              ││
│  │  ┌─────────────────────────────────────────────────────────┐││
│  │  │                    Pod (replica)                         │││
│  │  │  ┌───────────────────────────────────────────────────┐  │││
│  │  │  │              biometric-recognition                 │  │││
│  │  │  │  • Image: ECR repository                          │  │││
│  │  │  │  • Port: 8000                                     │  │││
│  │  │  │  • Resources: 512Mi-1Gi RAM, 100m-500m CPU        │  │││
│  │  │  │  • Liveness: /health (180s initial, 30s period)   │  │││
│  │  │  │  • Readiness: /health (120s initial, 10s period)  │  │││
│  │  │  └───────────────────────────────────────────────────┘  │││
│  │  └─────────────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Secrets: aws-credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_...)   │
└─────────────────────────────────────────────────────────────────┘
```

### Deploy to Kubernetes

```bash
# Create AWS credentials secret
kubectl create secret generic aws-credentials \
  --from-literal=AWS_ACCESS_KEY_ID=<your-key> \
  --from-literal=AWS_SECRET_ACCESS_KEY=<your-secret>

# Deploy
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check status
kubectl get pods -l app=biometric-recognition
kubectl get svc biometric-recognition

# View logs
kubectl logs -l app=biometric-recognition -f
```

### Resource Configuration

| Resource | Request | Limit |
| -------- | ------- | ----- |
| Memory   | 512Mi   | 1Gi   |
| CPU      | 100m    | 500m  |

### Health Probes

| Probe     | Path      | Initial Delay | Period |
| --------- | --------- | ------------- | ------ |
| Liveness  | `/health` | 180s          | 30s    |
| Readiness | `/health` | 120s          | 10s    |

### Scale Deployment

```bash
# Scale replicas
kubectl scale deployment biometric-recognition --replicas=3

# Check rollout status
kubectl rollout status deployment/biometric-recognition
```

## Infrastructure (Terraform)

Terraform is used to provision AWS infrastructure including ECR for container images and S3 for model artifacts.

### Resources Created

| Resource | Name | Description |
| -------- | ---- | ----------- |
| **ECR Repository** | `biometric-recognition` | Container image registry with scanning enabled |
| **S3 Bucket** | `biometric-recognition-artifacts` | Model and data storage with versioning and encryption |

### Infrastructure Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                    AWS Infrastructure                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    ECR Repository                            ││
│  │  • Name: biometric-recognition                               ││
│  │  • Image scanning on push                                    ││
│  │  • Lifecycle policy: Keep last 10 images                     ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    S3 Bucket                                 ││
│  │  • Name: biometric-recognition-artifacts                     ││
│  │  • Versioning: Enabled                                       ││
│  │  • Encryption: AES256 (SSE-S3)                               ││
│  │  • Public access: Blocked                                    ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Deploy Infrastructure

```bash
cd terraform

# Initialize Terraform
terraform init

# Preview changes
terraform plan

# Apply infrastructure
terraform apply

# Get outputs
terraform output
```

### Terraform Outputs

| Output | Description |
| ------ | ----------- |
| `ecr_repository_url` | ECR repository URL for Docker push |
| `ecr_repository_arn` | ECR repository ARN |
| `s3_bucket_name` | S3 bucket name (`biometric-recognition-artifacts`) |
| `s3_bucket_arn` | S3 bucket ARN |

## Docker Compose Services

```yaml
Services:
  postgres:      # Airflow metadata database
  mlflow:        # MLflow tracking server (port 5000)
  airflow-init:  # Database initialization
  airflow-webserver:  # Airflow UI (port 8080)
  airflow-scheduler:  # DAG scheduler
```

### Quick Start

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f mlflow
docker-compose logs -f airflow-webserver

# Stop services
docker-compose down
```

### Service URLs

| Service             | URL                       | Credentials   |
| ------------------- | ------------------------- | ------------- |
| Airflow UI          | <http://localhost:8080>   | admin / admin |
| MLflow UI           | <http://localhost:5000>   | -             |
| API (when deployed) | <http://localhost:8000>   | -             |

## Configuration

### Main Config (`configs/config.yaml`)

```yaml
data:
  path: "s3://biometric-recognition-artifacts/biometric_data/"  # S3 or local path
  num_people: 45
  fingerprint_size: [128, 128]
  iris_size: [64, 64]
  batch_size: 8
  num_workers: 8
  preload_images: true

model:
  fingerprint_feature_dim: 1280
  iris_feature_dim: 32
  fusion_hidden_dim: 128
  dropout: 0.5

training:
  epochs: 10
  learning_rate: 0.0001
  device: "auto"  # auto, cpu, cuda, mps

s3:
  model_bucket: "biometric-recognition-artifacts"
  model_prefix: "biometric_model/"
```

### Environment Variables

| Variable                | Description                         | Default                      |
| ----------------------- | ----------------------------------- | ---------------------------- |
| `MODEL_PATH`            | Model checkpoint path (local or S3) | `checkpoints/best_model.pth` |
| `NUM_CLASSES`           | Number of classes                   | 45                           |
| `MLFLOW_TRACKING_URI`   | MLflow server URL                   | `http://localhost:5000`      |
| `AWS_ACCESS_KEY_ID`     | AWS access key                      | -                            |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key                      | -                            |

## Dataset Format

```text
dataset/
├── 1/                    # Person ID
│   ├── Fingerprint/
│   │   └── image.bmp
│   ├── left/
│   │   └── iris_left.bmp
│   └── right/
│       └── iris_right.bmp
├── 2/
│   └── ...
└── N/
```

## Development

### Code Quality

```bash
# Format
poetry run black src/
poetry run isort src/

# Lint
poetry run flake8 src/

# Test
poetry run pytest
```

### Run Individual Pipeline Stages

```bash
# Data preparation
python -m biometric_recognition.pipeline.data_prep \
  --config configs/config.yaml \
  --output-dir outputs/data_prep

# Training
python -m biometric_recognition.pipeline.train \
  --config outputs/data_prep/config.yaml \
  --splits outputs/data_prep/data_splits.json \
  --checkpoint-dir outputs/checkpoints

# Evaluation
python -m biometric_recognition.pipeline.evaluate \
  --config outputs/data_prep/config.yaml \
  --splits outputs/data_prep/data_splits.json \
  --model outputs/checkpoints/best_model.pth \
  --history outputs/checkpoints/training_history.json \
  --output-dir outputs/evaluation
```

## Hardware Requirements

| Environment | CPU       | RAM       | GPU              |
| ----------- | --------- | --------- | ---------------- |
| Development | 4+ cores  | 8GB+      | Optional         |
| Training    | 8+ cores  | 16GB+     | NVIDIA 6GB+ VRAM |
| API Serving | 2+ cores  | 4GB+      | Optional         |
| Kubernetes  | 100m-500m | 512Mi-1Gi | -                |

## License

This project is licensed under the MIT License.
