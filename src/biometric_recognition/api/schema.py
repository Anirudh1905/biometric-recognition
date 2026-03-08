# Pydantic models for API
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class BiometricData(BaseModel):
    """Input data for biometric recognition."""

    fingerprint_image: str = Field(..., description="Base64 encoded fingerprint image")
    left_iris_image: str = Field(..., description="Base64 encoded left iris image")
    right_iris_image: str = Field(..., description="Base64 encoded right iris image")


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    predicted_class: int = Field(..., description="Predicted person ID")
    confidence: float = Field(..., description="Prediction confidence")
    top_k_predictions: List[Dict[str, Any]] = Field(
        ..., description="Top-k predictions with probabilities"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    device: str
