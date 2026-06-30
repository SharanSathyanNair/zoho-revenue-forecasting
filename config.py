"""
Central configuration for the revenue forecasting pipeline.

This file is the single source of truth for paths, thresholds, and
constants that are currently scattered as hardcoded values across
generate_data.py, aggregate.py, feature_engineering.py,
xgboost_tuning.py, train_xgboost.py, walk_forward_validation.py,
forecast_engine.py, prediction_interval.py, arima_model.py, and
prophet_model.py.

Existing scripts are not yet wired to this file -- that happens as
part of the per-tenant pipeline refactor. For now this establishes
the settings that work needs, particularly the MINIMUM_DATA_WEEKS
tiering thresholds and DRIFT settings used to gate forecast behavior.
"""

from pathlib import Path

# ==========================================================
# Paths
# ==========================================================
# NOTE: these are still single-tenant paths matching the current
# pipeline. The per-tenant refactor will parameterize these by
# organization_id (e.g. data/{org_id}/weekly_business_metrics.csv)
# instead of a single fixed path.

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

CUSTOMERS_FILE = DATA_DIR / "customers.csv"
INVOICES_FILE = DATA_DIR / "invoices.csv"
PAYMENTS_FILE = DATA_DIR / "customerpayments.csv"
WEEKLY_METRICS_FILE = DATA_DIR / "weekly_business_metrics.csv"
MODEL_READY_FILE = DATA_DIR / "model_ready_data.csv"

XGBOOST_MODEL_FILE = MODEL_DIR / "xgboost_model.pkl"
XGBOOST_PARAMS_FILE = MODEL_DIR / "best_xgb_params.json"
ARIMA_MODEL_FILE = MODEL_DIR / "arima_model.pkl"
PROPHET_MODEL_FILE = MODEL_DIR / "prophet_model.pkl"

RESIDUALS_FILE = OUTPUT_DIR / "residuals.npy"
FORECAST_SUMMARY_FILE = OUTPUT_DIR / "forecast_summary.json"
FUTURE_FORECAST_FILE = OUTPUT_DIR / "future_forecast.csv"


# ==========================================================
# Random Seed
# ==========================================================

RANDOM_STATE = 42


# ==========================================================
# Feature Engineering
# ==========================================================

REVENUE_LAGS = (1, 2, 4, 8, 12, 26)
ROLLING_WINDOWS = (4, 8, 12)

# Longest lag determines how many weeks of history are consumed
# before a single usable row exists -- this directly drives
# MINIMUM_DATA_WEEKS below.
LONGEST_LAG_WEEKS = max(REVENUE_LAGS)


# ==========================================================
# Hyperparameter Tuning
# ==========================================================

TUNING_N_SPLITS = 5
TUNING_N_ITER = 100
TUNING_HOLDOUT_WEEKS = 12


# ==========================================================
# Training / Evaluation
# ==========================================================

DEFAULT_HOLDOUT_WEEKS = 12
WALK_FORWARD_INITIAL_TRAIN_SIZE = 104
WALK_FORWARD_FORECAST_HORIZON = 1
CALIBRATION_SIZE = 20


# ==========================================================
# Prediction Intervals
# ==========================================================

CONFIDENCE_LEVEL = 0.95


# ==========================================================
# Drift / Stability
# ==========================================================

# Below this stability score, the model is flagged as drifting.
DRIFT_STABILITY_THRESHOLD = 75

# Below this (lower, harder) threshold, the forecast should not be
# trusted at all and the pipeline should fall back to a simple
# non-ML method (seasonal naive / moving average) instead of
# showing an XGBoost forecast. This is intentionally lower than
# DRIFT_STABILITY_THRESHOLD: the first threshold means "warn the
# user", this one means "don't show the ML forecast at all."
DRIFT_FALLBACK_THRESHOLD = 55


# ==========================================================
# Minimum Data Tiering
# ==========================================================
# Real Zoho Books customers will have wildly different amounts of
# history -- from brand new accounts to years of data. These
# thresholds define what level of forecasting is responsible to
# offer at each stage, instead of assuming everyone has 208 weeks
# of clean history like the synthetic dataset does.
#
# Tiers (in weeks of usable invoice history):
#   < MINIMUM_DATA_WEEKS_NONE          -> no forecast offered at all
#   < MINIMUM_DATA_WEEKS_NAIVE_ONLY    -> naive/seasonal baseline only
#   < MINIMUM_DATA_WEEKS_REDUCED_ML    -> ML with a reduced (short-lag) feature set
#   >= MINIMUM_DATA_WEEKS_FULL_ML      -> full feature set, full ML pipeline

MINIMUM_DATA_WEEKS_NONE = 8
MINIMUM_DATA_WEEKS_NAIVE_ONLY = 12
MINIMUM_DATA_WEEKS_REDUCED_ML = LONGEST_LAG_WEEKS + 4   # = 30
MINIMUM_DATA_WEEKS_FULL_ML = WALK_FORWARD_INITIAL_TRAIN_SIZE  # = 104


__all__ = [
    "DATA_DIR",
    "MODEL_DIR",
    "OUTPUT_DIR",
    "CUSTOMERS_FILE",
    "INVOICES_FILE",
    "PAYMENTS_FILE",
    "WEEKLY_METRICS_FILE",
    "MODEL_READY_FILE",
    "XGBOOST_MODEL_FILE",
    "XGBOOST_PARAMS_FILE",
    "ARIMA_MODEL_FILE",
    "PROPHET_MODEL_FILE",
    "RESIDUALS_FILE",
    "FORECAST_SUMMARY_FILE",
    "FUTURE_FORECAST_FILE",
    "RANDOM_STATE",
    "REVENUE_LAGS",
    "ROLLING_WINDOWS",
    "LONGEST_LAG_WEEKS",
    "TUNING_N_SPLITS",
    "TUNING_N_ITER",
    "TUNING_HOLDOUT_WEEKS",
    "DEFAULT_HOLDOUT_WEEKS",
    "WALK_FORWARD_INITIAL_TRAIN_SIZE",
    "WALK_FORWARD_FORECAST_HORIZON",
    "CALIBRATION_SIZE",
    "CONFIDENCE_LEVEL",
    "DRIFT_STABILITY_THRESHOLD",
    "DRIFT_FALLBACK_THRESHOLD",
    "MINIMUM_DATA_WEEKS_NONE",
    "MINIMUM_DATA_WEEKS_NAIVE_ONLY",
    "MINIMUM_DATA_WEEKS_REDUCED_ML",
    "MINIMUM_DATA_WEEKS_FULL_ML",
]