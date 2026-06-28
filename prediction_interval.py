"""""
Prediction interval utilities for the XGBoost revenue forecasting pipeline.

This module provides:
- Conformal prediction intervals
- Forecast confidence estimation
- Model stability estimation
- Drift detection
- Forecast summary generation
"""

import numpy as np

# ==========================================================
# Configuration
# ==========================================================

CONFIDENCE_LEVEL: float = 0.95
DRIFT_THRESHOLD: int = 75

# ==========================================================
# Absolute Residuals
# ==========================================================

def calculate_residuals(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    """
    Compute absolute residuals between actual and predicted values.
    """

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) == 0:
        raise ValueError("Actual values cannot be empty.")

    if len(predicted) == 0:
        raise ValueError("Predicted values cannot be empty.")

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted arrays must have the same length."
        )

    return np.abs(actual - predicted)


# ==========================================================
# Conformal Quantile
# ==========================================================

def calculate_conformal_quantile(
    residuals: np.ndarray,
    confidence: float = CONFIDENCE_LEVEL,
) -> float:
    """
    Compute the conformal prediction quantile.
    """

    residuals = np.asarray(residuals, dtype=float)

    if len(residuals) == 0:
        raise ValueError("Residuals cannot be empty.")

    if not 0 < confidence < 1:
        raise ValueError(
            "Confidence level must be between 0 and 1."
        )

    alpha = 1 - confidence

    return float(
        np.quantile(
            residuals,
            1 - alpha,
            method="higher",
        )
    )

# ==========================================================
# Prediction Interval
# ==========================================================

def generate_prediction_interval(
    predictions: np.ndarray,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate prediction intervals from a conformal quantile.
    """

    predictions = np.asarray(predictions, dtype=float)

    if len(predictions) == 0:
        raise ValueError("Predictions cannot be empty.")

    if quantile < 0:
        raise ValueError("Quantile must be non-negative.")

    lower = predictions - quantile
    upper = predictions + quantile

    return lower, upper


# ==========================================================
# Forecast Confidence
# ==========================================================

def calculate_forecast_confidence(
    quantile: float,
    predictions: np.ndarray,
) -> float:
    """
    Estimate forecast confidence from prediction interval width.
    """

    predictions = np.asarray(predictions, dtype=float)

    if len(predictions) == 0:
        raise ValueError("Predictions cannot be empty.")

    average_prediction = max(float(np.mean(predictions)), 1.0)

    uncertainty_ratio = quantile / average_prediction

    confidence = np.clip(
        (1 - uncertainty_ratio) * 100,
        0,
        100,
    )

    return round(float(confidence), 2)

# ==========================================================
# Recent Model Stability
# ==========================================================

def calculate_recent_model_stability(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Estimate model stability using the coefficient of variation
    of prediction errors.
    """

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) == 0:
        raise ValueError("Actual values cannot be empty.")

    if len(predicted) == 0:
        raise ValueError("Predicted values cannot be empty.")

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted arrays must have the same length."
        )

    errors = np.abs(actual - predicted)

    mean_error = max(float(np.mean(errors)), 1.0)
    std_error = float(np.std(errors))

    coefficient_of_variation = std_error / mean_error

    stability = np.clip(
        100 / (1 + coefficient_of_variation),
        0,
        100,
    )

    return round(float(stability), 2)


# ==========================================================
# Drift Detection
# ==========================================================

def check_model_drift(
    stability: float,
    threshold: int = DRIFT_THRESHOLD,
) -> bool:
    """
    Determine whether model drift should be flagged.
    """

    return stability < threshold

# ==========================================================
# Forecast Summary
# ==========================================================

def generate_forecast_summary(
    forecast: float,
    lower: float,
    upper: float,
    confidence: float,
    stability: float,
) -> dict:
    """
    Generate a deployment-ready forecast summary.
    """

    return {
        "forecast": float(forecast),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "forecast_confidence": float(confidence),
        "model_stability": float(stability),
        "drift_detected": check_model_drift(stability),
    }


__all__ = [
    "calculate_residuals",
    "calculate_conformal_quantile",
    "generate_prediction_interval",
    "calculate_forecast_confidence",
    "calculate_recent_model_stability",
    "check_model_drift",
    "generate_forecast_summary",
]