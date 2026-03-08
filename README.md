# Multimodal Biometric Recognition System

A PyTorch-based multimodal biometric recognition system that uses fingerprints and iris images for person identification.

## Features

- **Multimodal Approach**: Combines fingerprint and iris (left & right) biometric modalities
- **PyTorch Implementation**: Modern deep learning framework with GPU support
- **Modular Architecture**: Clean, maintainable code structure
- **Hydra Configuration**: Flexible configuration management
- **Production Ready**: Logging, monitoring, and error handling

## Architecture

The system consists of three main branches:

1. **Fingerprint Branch**: Uses a pretrained MobileNetV2 backbone for feature extraction
2. **Iris Branches**: Shared CNN architecture for processing left and right iris images
3. **Fusion Module**: Combines features from all modalities for final classification

## Project Structure

```
biometric-recognition/
├── src/
│   └── biometric_recognition/
│       ├── data/                 # Data loading and preprocessing
│       ├── models/              # Model architectures
│       ├── utils/               # Utility functions
│       ├── train.py            # Training script
│       └── inference.py        # Inference script
├── configs/                     # Hydra configuration files
│   └── config.yaml             # Simple, unified configuration
├── outputs/                     # Training outputs (created during training)
├── checkpoints/                # Model checkpoints (created during training)
└── pyproject.toml              # Project dependencies and configuration
```

## Installation

### Prerequisites
- Python 3.12 or higher
- Docker (for containerized deployment)
- AWS CLI (if using S3 features)

### Method 1: Local Development Setup

1. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Clone and install the project**:
   ```bash
   git clone <repository-url>
   cd biometric-recognition
   poetry install
   ```

3. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

### Method 2: Docker Setup

1. **Build and run with Docker Compose**:
   ```bash
   # Development
   docker-compose --profile dev up

   # Production
   docker-compose --profile prod up -d

   # Training
   docker-compose --profile training up
   ```

2. **Or build manually**:
   ```bash
   # Build production image
   docker build --target production -t biometric-recognition:latest .

   # Run API server
   docker run -p 8000:8000 -v ./checkpoints:/home/app/checkpoints biometric-recognition:latest
   ```

## Dataset Format

The expected dataset structure is:
```
dataset/
├── 1/
│   ├── Fingerprint/
│   │   └── image.bmp
│   ├── left/
│   │   └── iris_left.bmp
│   └── right/
│       └── iris_right.bmp
├── 2/
│   ├── Fingerprint/
│   ├── left/
│   └── right/
└── ...
```

## Quick Start with S3

For a complete train → store → serve workflow with S3:

```bash
# 1. Configure AWS credentials
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export S3_MODEL_BUCKET="your-biometric-models-bucket"

# 2. Update dataset path in configs/config.yaml
# data:
#   path: "s3://your-bucket/datasets/"  # or local path

# 3. Train model (automatically uploads to S3)
python src/biometric_recognition/train.py

# 4. Serve from S3 (uses the uploaded model)
export MODEL_PATH="s3://your-bucket/models/best_model_YYYYMMDD_HHMMSS.pth"
python src/biometric_recognition/serve.py
```

## Usage

### Configuration

Update the main configuration file `configs/config.yaml`:

```yaml
# Dataset settings
data:
  path: "/path/to/your/dataset"  # Update this path
  num_people: 45                 # Number of people in your dataset
  batch_size: 8

# Model settings
model:
  fingerprint_feature_dim: 256
  iris_feature_dim: 64
  fusion_hidden_dim: 128
  dropout: 0.5

# Training settings
training:
  epochs: 50
  learning_rate: 0.0001
  device: "auto"  # auto, cpu, cuda, mps
```

### Training

```bash
# Basic training
python src/biometric_recognition/train.py

# Training with custom parameters
python src/biometric_recognition/train.py data.batch_size=16 training.epochs=100

# Training with specific device
python src/biometric_recognition/train.py training.device=cuda

# Training with automatic S3 model upload (set S3_MODEL_BUCKET environment variable)
export S3_MODEL_BUCKET="your-biometric-models-bucket"
python src/biometric_recognition/train.py
```

### Inference

```bash
# Run inference with best model
python src/biometric_recognition/inference.py model_path=checkpoints/best_model.pth

# Run inference and save predictions
python src/biometric_recognition/inference.py \
    model_path=checkpoints/best_model.pth \
    save_predictions=true \
    output_file=my_predictions.json
```

### API Server

Start the FastAPI server for real-time inference:

```bash
# Using Poetry
poetry run serve --model-path checkpoints/best_model.pth

# Using Python directly
python src/biometric_recognition/serve.py --model-path checkpoints/best_model.pth

# Using Docker
docker-compose --profile prod up -d
```

**API Endpoints:**
- `GET /health` - Health check
- `POST /predict` - Predict from base64 encoded images
- `POST /predict/files` - Predict from uploaded files
- `GET /model/info` - Get model information
- `GET /docs` - Interactive API documentation

**Example API Usage:**
```python
import requests
import base64

# Prepare images
with open("fingerprint.bmp", "rb") as f:
    fingerprint_b64 = base64.b64encode(f.read()).decode()

with open("left_iris.bmp", "rb") as f:
    left_iris_b64 = base64.b64encode(f.read()).decode()

with open("right_iris.bmp", "rb") as f:
    right_iris_b64 = base64.b64encode(f.read()).decode()

# Make prediction
response = requests.post("http://localhost:8000/predict", json={
    "fingerprint_image": fingerprint_b64,
    "left_iris_image": left_iris_b64,
    "right_iris_image": right_iris_b64
})

result = response.json()
print(f"Predicted class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.4f}")
```

## Configuration Options

### Data Configuration
- `data.path`: Path to the dataset directory
- `data.num_people`: Number of people in the dataset (default: 45)
- `data.fingerprint_size`: Size of fingerprint images [height, width] (default: [128, 128])
- `data.iris_size`: Size of iris images [height, width] (default: [64, 64])
- `data.batch_size`: Training batch size (default: 8)
- `data.num_workers`: Number of workers for data loading (default: 4)

### Model Configuration
- `model.fingerprint_feature_dim`: Feature dimension for fingerprint branch (default: 256)
- `model.iris_feature_dim`: Feature dimension for iris branches (default: 64)
- `model.fusion_hidden_dim`: Hidden dimension for fusion layer (default: 128)
- `model.dropout`: Dropout probability (default: 0.5)

### Training Configuration
- `training.epochs`: Number of training epochs (default: 50)
- `training.learning_rate`: Learning rate for optimizer (default: 0.0001)
- `training.device`: Training device: "auto", "cpu", "cuda", or "mps" (default: "auto")

## Model Performance

The model architecture achieves:
- Multi-modal feature fusion
- Shared weights for left/right iris processing
- Transfer learning with pretrained fingerprint backbone
- Dropout regularization for better generalization

## Production Features

### 🚀 **API Serving**
- **FastAPI** REST API for real-time inference
- Base64 and file upload support
- Automatic OpenAPI documentation
- Health checks and monitoring endpoints
- CORS support for web applications

### ☁️ **Cloud Integration**
- **S3 Support** for data and model storage
- Automatic data download and caching
- Model versioning and registry
- AWS credentials integration

### 🐳 **Docker Support**
- Multi-stage builds (development, production, training)
- Docker Compose for easy deployment
- Nginx reverse proxy configuration
- Health checks and auto-restart
- Non-root user security

### 🔄 **CI/CD Pipeline**
- **GitHub Actions** workflows
- Automated testing and linting
- Security scanning with Trivy
- Multi-architecture Docker builds
- Automated model training pipeline
- Slack notifications

### 📊 **Monitoring & Observability**
- Structured logging with timestamps
- Model performance tracking
- API metrics and health monitoring
- Training progress visualization

## Development

### Code Style
The project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

Run code quality checks:
```bash
# Format code
poetry run black src/
poetry run isort src/

# Lint
poetry run flake8 src/
poetry run mypy src/
```

### Testing
```bash
poetry run pytest
```

## Outputs

### Training
- Model checkpoints saved to `checkpoints/`
- Training history plots saved to `outputs/`
- Logs with training progress and metrics

### Inference
- Predictions saved as JSON with confidence scores
- Top-k predictions for each sample
- Accuracy metrics (if ground truth available)

## Hardware Requirements

- **Minimum**: CPU with 8GB RAM
- **Recommended**: NVIDIA GPU with 6GB+ VRAM
- **Apple Silicon**: MPS support included
- **Production**: 2+ CPU cores, 4GB+ RAM for API serving

## Production Deployment

### S3 Model Management Workflow

The system supports seamless S3 integration for model storage and serving:

1. **Train and Auto-Upload**:
   ```bash
   # Set S3 bucket for automatic upload after training
   export S3_MODEL_BUCKET="your-biometric-models-bucket"
   export AWS_ACCESS_KEY_ID="your-access-key"
   export AWS_SECRET_ACCESS_KEY="your-secret-key"

   # Train model (will automatically upload to S3)
   python src/biometric_recognition/train.py
   ```

2. **Serve from S3**:
   ```bash
   # Serve model directly from S3 URI
   export MODEL_PATH="s3://your-bucket/models/best_model_20241215_143022.pth"
   python src/biometric_recognition/serve.py
   ```

3. **Manual S3 Management**:
   ```bash
   # Run S3 management examples
   python examples/s3_model_management.py
   ```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Model configuration
MODEL_PATH=checkpoints/best_model.pth  # or S3 URI
NUM_CLASSES=45

# AWS configuration (required for S3 features)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# S3 model storage (optional - enables auto-upload)
S3_MODEL_BUCKET=your-biometric-models-bucket

# API configuration
HOST=0.0.0.0
PORT=8000
```

### Docker Deployment

1. **Single container**:
   ```bash
   docker run -d \
     --name biometric-api \
     -p 8000:8000 \
     -e MODEL_PATH=checkpoints/best_model.pth \
     -v ./checkpoints:/home/app/checkpoints:ro \
     biometric-recognition:latest
   ```

2. **With Docker Compose**:
   ```bash
   # Production deployment with Nginx
   docker-compose --profile prod up -d

   # Scale the API service
   docker-compose --profile prod up -d --scale biometric-api=3
   ```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: biometric-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: biometric-api
  template:
    metadata:
      labels:
        app: biometric-api
    spec:
      containers:
      - name: api
        image: biometric-recognition:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "s3://your-bucket/models/model.pth"
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
```

### GitHub Actions Secrets

Configure these secrets in your GitHub repository:

- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub password
- `AWS_ACCESS_KEY_ID` - AWS access key
- `AWS_SECRET_ACCESS_KEY` - AWS secret key
- `AWS_REGION` - AWS region
- `MODEL_BUCKET` - S3 bucket for models
- `SLACK_WEBHOOK` - Slack webhook URL (optional)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and quality checks
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{multimodal-biometric-recognition,
  title={Multimodal Biometric Recognition System},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/biometric-recognition}
}
```