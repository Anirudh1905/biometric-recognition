"""Pydantic models for API request/response schemas."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


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
