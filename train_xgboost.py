import json
import numpy as np 
import warnings
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from prediction_interval import (
    calculate_residuals,
    calculate_conformal_quantile,
    generate_prediction_interval,
    calculate_forecast_confidence,
    calculate_recent_model_stability,
    check_model_drift,
    generate_forecast_summary,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

# ==========================================================
# Configuration
# ==========================================================

INPUT_FILE = "data/model_ready_data.csv"
PARAM_FILE = "models/best_xgb_params.json"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
FORECAST_SUMMARY_FILE = "forecast_summary.json"
MODEL_NAME = "xgboost_model.pkl"
PREDICTION_FILE = "xgboost_predictions.csv"
DEFAULT_FORECAST_WEEKS = 12
RANDOM_STATE = 42

# ==========================================================
# Directories
# ==========================================================

def create_directories():
    Path(MODEL_DIR).mkdir(exist_ok=True)
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

# ==========================================================
# Load Data
# ==========================================================

def load_data() -> pd.DataFrame:
    """Load the model-ready dataset."""
    path = Path(INPUT_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )
    try:
        df = pd.read_csv(
            path,
            parse_dates=["week"],
        )
    except Exception as e:
        raise RuntimeError(
            "Unable to load model-ready dataset."
        ) from e
    return (
        df.sort_values("week")
          .reset_index(drop=True)
    )

# ==========================================================
# Parameters
# ==========================================================


def load_parameters() -> dict:
    """Load tuned XGBoost parameters."""
    path = Path(PARAM_FILE)
    if not path.exists():
        raise FileNotFoundError(
            "Run xgboost_tuning.py first."
        )
    with open(path, "r") as f:
        return json.load(f)

from datetime import datetime


def calculate_forecast_horizon(
    last_week,
    forecast_type="weeks",
    weeks=12,
    target_year=None,
    target_month=None,
):
    """
    Calculate forecast horizon in weeks.
    """

    if forecast_type == "weeks":
        return max(1, weeks)

    target_date = pd.Timestamp(
        year=target_year,
        month=target_month,
        day=1,
    ) + pd.offsets.MonthEnd(0)

    horizon = (
        (target_date - last_week).days // 7
    )

    return max(1, horizon)

# ==========================================================
# Split
# ==========================================================

def split_data(df, forecast_horizon):
    train = df.iloc[:-forecast_horizon].copy()
    test = df.iloc[-forecast_horizon:].copy()

    return train, test

# ==========================================================
# Features
# ==========================================================

def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split features and target."""
    X = df.drop(
        columns=[
            "week",
            "weekly_revenue",
        ]
    )
    y = df["weekly_revenue"]
    return X, y

# ==========================================================
# Build Model
# ==========================================================

def build_model(
    params: dict,
) -> XGBRegressor:
    """Build the XGBoost model."""
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **params,
    )

# ==========================================================
# Train
# ==========================================================

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# ==========================================================
# Predictions
# ==========================================================

def predict(model, X):
    return model.predict(X)

# ==========================================================
# Evaluation
# ==========================================================

def evaluate(
    actual,
    predicted,
) -> tuple[float, float, float]:
    """Evaluate forecasting performance."""
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = mse ** 0.5
    mape = (
        mean_absolute_percentage_error(
            actual,
            predicted,
        )
        * 100
    )

    return mae, rmse, mape
# ==========================================================
# Save Model
# ==========================================================

def save_model(
    model: XGBRegressor,
) -> Path:
    """Save the trained model."""
    path = Path(MODEL_DIR) / MODEL_NAME
    joblib.dump(model, path)
    return path

# ==========================================================
# Save Predictions
# ==========================================================

def save_predictions(test, predictions, lower, upper, confidence, stability):
    df = pd.DataFrame(
        {
            "week": test["week"],
            "actual": test["weekly_revenue"],
            "predicted": predictions,
            "lower_bound": lower,
            "upper_bound": upper,
        }
    )

    df["prediction_error"] = df["actual"] - df["predicted"]
    df["forecast_confidence"] = confidence
    df["model_stability"] = stability

    path = Path(OUTPUT_DIR) / PREDICTION_FILE
    df.to_csv(path, index=False)

    return path

# ==========================================================
# Save Forecast Summary
# ==========================================================

def save_forecast_summary(summary):
    path = Path(OUTPUT_DIR) / FORECAST_SUMMARY_FILE

    with open(path, "w") as f:
        json.dump(summary, f, indent=4)

    return path

# ==========================================================
# Full History Plot
# ==========================================================

def plot_full_history(df, predictions):
    plt.figure(figsize=(15, 6))
    plt.plot(
        df["week"],
        df["weekly_revenue"],
        color="black",
        linewidth=2.5,
        label="Actual",
    )
    plt.plot(
        df["week"],
        predictions,
        color="blue",
        linewidth=2,
        label="Predicted",
    )
    plt.title("Actual vs Predicted Revenue (Entire Dataset)")

    plt.xlabel("Week")
    plt.ylabel("Revenue")

    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        Path(OUTPUT_DIR) / "xgboost_full_prediction.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

# ==========================================================
# Forecast Plot
# ==========================================================

def plot_forecast(test, predictions, lower, upper):
    plt.figure(figsize=(14, 6))
    plt.plot(
        test["week"],
        test["weekly_revenue"],
        color="black",
        linewidth=3,
        marker="o",
        markersize=6,
        label="Actual",
    )
    plt.plot(
        test["week"],
        predictions,
        color="blue",
        linewidth=3,
        marker="o",
        markersize=6,
        label="Forecast",
    )
    plt.fill_between(
        test["week"],
        lower,
        upper,
        color="dodgerblue",
        alpha=0.20,
        label="Prediction Interval",
    )
    plt.title("12-Week Revenue Forecast with Prediction Interval")

    plt.xlabel("Week")
    plt.ylabel("Revenue")

    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        Path(OUTPUT_DIR) / "xgboost_forecast.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

# ==========================================================
# Feature Importance
# ==========================================================

def plot_feature_importance(model, X):
    importance = (
        pd.Series(model.feature_importances_, index=X.columns).sort_values().tail(20)
    )

    plt.figure(figsize=(10, 8))

    importance.plot.barh(color="royalblue")

    plt.title("Top 20 Feature Importance")
    plt.tight_layout()

    plt.savefig(
        Path(OUTPUT_DIR) / "feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

# ==========================================================
# SHAP Summary
# ==========================================================

def plot_shap(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    shap.summary_plot(shap_values, X, show=False)

    plt.tight_layout()

    plt.savefig(
        Path(OUTPUT_DIR) / "shap_summary.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

# ==========================================================
# Summary
# ==========================================================

def print_summary(
    train_df,
    test_df,
    X_train,
    train_metrics,
    test_metrics,
    confidence,
    stability,
    drift_detected,
    model_path,
    prediction_path,
    summary_path,
):
    train_mae, train_rmse, train_mape = train_metrics
    test_mae, test_rmse, test_mape = test_metrics

    print("\n=====================================================")
    print("XGBoost Revenue Forecasting Results")
    print("=====================================================")

    print(f"Training Weeks          : {len(train_df)}")
    print(f"Testing Weeks           : {len(test_df)}")
    print(f"Features Used           : {X_train.shape[1]}")

    print("\nTraining Metrics")
    print("------------------------------")
    print(f"MAE                     : {train_mae:,.2f}")
    print(f"RMSE                    : {train_rmse:,.2f}")
    print(f"MAPE                    : {train_mape:.2f}%")

    print("\nTesting Metrics")
    print("------------------------------")
    print(f"MAE                     : {test_mae:,.2f}")
    print(f"RMSE                    : {test_rmse:,.2f}")
    print(f"MAPE                    : {test_mape:.2f}%")

    print("\nForecast Intelligence")
    print("------------------------------")
    print(f"Forecast Confidence     : {confidence:.2f}%")
    print(f"Model Stability         : {stability:.2f}%")
    print(f"Drift Detected          : {drift_detected}")

    print(f"\nModel Saved             : {model_path}")
    print(f"Predictions Saved       : {prediction_path}")
    print(f"Forecast Summary        : {summary_path}")

    print("=====================================================")

# ==========================================================    
# Main
# ==========================================================

def main() -> None:
    """Execute the training pipeline."""
    print("\nStarting XGBoost Training...\n")

    try:
        create_directories()
        df = load_data()
        params = load_parameters()
        forecast_horizon = DEFAULT_FORECAST_WEEKS
        train_df, test_df = split_data(
            df,
            forecast_horizon,
        )
        X_train, y_train = prepare_features(train_df)
        X_test, y_test = prepare_features(test_df)
        X_all, _ = prepare_features(df)
        model = train_model(
            build_model(params),
            X_train,
            y_train,
        )
        train_predictions = predict(model, X_train)
        test_predictions = predict(model, X_test)
        residuals = calculate_residuals(
            y_test,
            test_predictions,
        )
        # ==========================================================
        # Save Residuals
        # ==========================================================
        np.save(
            Path(OUTPUT_DIR) / "residuals.npy",
            residuals,
        )
        quantile = calculate_conformal_quantile(
            residuals,
        )
        lower, upper = generate_prediction_interval(
            test_predictions,
            quantile,
        )
        confidence = calculate_forecast_confidence(
            quantile,
            test_predictions,
        )
        stability = calculate_recent_model_stability(
            y_test,
            test_predictions,
        )
        drift_detected = check_model_drift(
            stability,
        )
        forecast_summary = generate_forecast_summary(
            forecast=test_predictions[-1],
            lower=lower[-1],
            upper=upper[-1],
            confidence=confidence,
            stability=stability,
        )
        full_predictions = predict(model, X_all)

        train_metrics = evaluate(
            y_train,
            train_predictions,
        )
        test_metrics = evaluate(
            y_test,
            test_predictions,
        )
        model_path = save_model(model)
        prediction_path = save_predictions(
            test_df,
            test_predictions,
            lower,
            upper,
            confidence,
            stability,
        )
        summary_path = save_forecast_summary(
            forecast_summary,
        )
        plot_full_history(
            df,
            full_predictions,
        )
        plot_forecast(
            test_df,
            test_predictions,
            lower,
            upper,
        )
        plot_feature_importance(
            model,
            X_train,
        )
        plot_shap(
            model,
            X_train,
        )
        print_summary(
            train_df,
            test_df,
            X_train,
            train_metrics,
            test_metrics,
            confidence,
            stability,
            drift_detected,
            model_path,
            prediction_path,
            summary_path,
        )
    except Exception as e:
        print("\n==============================")
        print("XGBoost Training Failed")
        print("==============================")
        print(e)
        print("==============================")
        raise

if __name__ == "__main__":
    main()