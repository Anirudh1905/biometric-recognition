"""FastAPI server for biometric recognition inference."""

import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from PIL import Image

from biometric_recognition.api.schema import HealthResponse, PredictionResponse
from biometric_recognition.models import MultimodalBiometricModel
from biometric_recognition.utils.device_utils import get_device
from biometric_recognition.utils.image_utils import prepare_batch_from_images
from biometric_recognition.utils.logging_utils import setup_logging
from biometric_recognition.utils.model_utils import (
    get_model_info,
    load_model_from_checkpoint,
    predict_batch,
)

# Global model instance
model_instance = None
model_path = None
device = None

# Default model path - can be overridden by MODEL_PATH env var
# os.environ.setdefault(
#     "MODEL_PATH", "s3://biometric-recognition-artifacts/biometric_model/20260308_172116/model.pth"
# )


def get_model() -> MultimodalBiometricModel:
    """Dependency to get the loaded model."""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan with startup and shutdown events."""
    global model_instance, model_path, device

    # Startup
    # Configure logging
    setup_logging()

    # Model configuration from environment variables
    model_path = os.getenv("MODEL_PATH", "checkpoints/best_model.pth")
    num_classes = int(os.getenv("NUM_CLASSES", "45"))
    device_pref = os.getenv("DEVICE", "auto")

    try:
        # Set device
        device = get_device(device_pref)

        # Load model using shared utility
        model_instance = load_model_from_checkpoint(
            checkpoint_path=model_path, num_classes=num_classes, device=device
        )
        logging.info("Model loaded successfully on startup")
    except Exception as e:
        logging.error(f"Failed to load model on startup: {e}")
        # Don't fail startup, just log the error

    yield

    # Shutdown
    logging.info("Shutting down application")
    # Clean up resources if needed
    model_instance = None


# Create FastAPI app with lifespan
app = FastAPI(
    title="Biometric Recognition API",
    description="Multimodal biometric recognition using fingerprints and iris images",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_instance is not None else "model_not_loaded",
        model_loaded=model_instance is not None,
        device=str(device) if device is not None else "unknown",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    fingerprint: UploadFile = File(..., description="Fingerprint image file"),
    left_iris: UploadFile = File(..., description="Left iris image file"),
    right_iris: UploadFile = File(..., description="Right iris image file"),
    model: MultimodalBiometricModel = Depends(get_model),
):
    """Predict person identity from uploaded image files."""
    try:
        # Read images
        fingerprint_img = Image.open(io.BytesIO(await fingerprint.read()))
        left_iris_img = Image.open(io.BytesIO(await left_iris.read()))
        right_iris_img = Image.open(io.BytesIO(await right_iris.read()))

        # Prepare batch using shared utility
        batch = prepare_batch_from_images(
            fingerprint_img, left_iris_img, right_iris_img, device=device
        )

        # Inference using shared utility
        predictions = predict_batch(model, batch, device)

        # Extract results
        predicted_class = int(predictions["predicted_classes"][0])
        confidence = float(predictions["confidence_scores"][0])

        # Format top-k predictions
        top_k_predictions = [
            {
                "class": int(predictions["top_k_indices"][0][i]),
                "probability": float(predictions["top_k_probabilities"][0][i]),
            }
            for i in range(len(predictions["top_k_indices"][0]))
        ]

        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            top_k_predictions=top_k_predictions,
        )

    except Exception as e:
        logging.error(f"File prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model/info", response_model=Dict[str, Any])
async def model_info():
    """Get information about the loaded model."""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Use shared utility to get model info
    info = get_model_info(model_instance, model_path)
    info["device"] = str(device)

    return info


def main():
    """Run the FastAPI server."""
    uvicorn.run(
        "biometric_recognition.api.serve:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
