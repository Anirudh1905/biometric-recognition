"""Tests for API schema module."""

import pytest
from pydantic import ValidationError

from biometric_recognition.api.schema import HealthResponse, PredictionResponse


class TestPredictionResponse:
    """Tests for PredictionResponse schema."""

    def test_valid_response(self):
        """Test creating valid PredictionResponse."""
        response = PredictionResponse(
            predicted_class=5,
            confidence=0.95,
            top_k_predictions=[
                {"class": 5, "probability": 0.95},
                {"class": 3, "probability": 0.03},
                {"class": 1, "probability": 0.02},
            ],
        )

        assert response.predicted_class == 5
        assert response.confidence == 0.95
        assert len(response.top_k_predictions) == 3

    def test_missing_predicted_class_raises_error(self):
        """Test that missing predicted_class raises validation error."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                confidence=0.95,
                top_k_predictions=[],
            )

    def test_missing_confidence_raises_error(self):
        """Test that missing confidence raises validation error."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                predicted_class=5,
                top_k_predictions=[],
            )

    def test_missing_top_k_predictions_raises_error(self):
        """Test that missing top_k_predictions raises validation error."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                predicted_class=5,
                confidence=0.95,
            )

    def test_empty_top_k_predictions_is_valid(self):
        """Test that empty top_k_predictions list is valid."""
        response = PredictionResponse(
            predicted_class=0,
            confidence=1.0,
            top_k_predictions=[],
        )

        assert response.top_k_predictions == []


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_healthy_response(self):
        """Test creating healthy response."""
        response = HealthResponse(
            status="healthy",
            model_loaded=True,
            device="cuda:0",
        )

        assert response.status == "healthy"
        assert response.model_loaded is True
        assert response.device == "cuda:0"

    def test_unhealthy_response(self):
        """Test creating unhealthy response."""
        response = HealthResponse(
            status="model_not_loaded",
            model_loaded=False,
            device="unknown",
        )

        assert response.status == "model_not_loaded"
        assert response.model_loaded is False

    def test_missing_status_raises_error(self):
        """Test that missing status raises validation error."""
        with pytest.raises(ValidationError):
            HealthResponse(
                model_loaded=True,
                device="cpu",
            )

    def test_missing_model_loaded_raises_error(self):
        """Test that missing model_loaded raises validation error."""
        with pytest.raises(ValidationError):
            HealthResponse(
                status="healthy",
                device="cpu",
            )

    def test_missing_device_raises_error(self):
        """Test that missing device raises validation error."""
        with pytest.raises(ValidationError):
            HealthResponse(
                status="healthy",
                model_loaded=True,
            )
