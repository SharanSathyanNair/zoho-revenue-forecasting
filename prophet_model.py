import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prophet import Prophet
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

warnings.filterwarnings("ignore")

# ==========================================================
# Configuration
# ==========================================================

INPUT_FILE = "data/model_ready_data.csv"

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

MODEL_NAME = "prophet_model.pkl"
PREDICTION_FILE = "prophet_predictions.csv"

FORECAST_HORIZON = 12


# ==========================================================
# Create Directories
# ==========================================================


def create_directories():
    Path(MODEL_DIR).mkdir(exist_ok=True)
    Path(OUTPUT_DIR).mkdir(exist_ok=True)


# ==========================================================
# Load Data
# ==========================================================


def load_data():
    df = pd.read_csv(INPUT_FILE, parse_dates=["week"])
    df = df.sort_values("week").reset_index(drop=True)
    return df


# ==========================================================
# Train Test Split
# ==========================================================


def train_test_split(df):
    train = df.iloc[:-FORECAST_HORIZON].copy()
    test = df.iloc[-FORECAST_HORIZON:].copy()
    return train, test


# ==========================================================
# Prepare Prophet Data
# ==========================================================


def prepare_prophet_data(df):
    prophet_df = df[["week", "weekly_revenue"]].copy()
    prophet_df.columns = ["ds", "y"]
    return prophet_df


# ==========================================================
# Train Model
# ==========================================================


def train_prophet(train_df):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
    )

    model.fit(train_df)

    return model


# ==========================================================
# Prediction
# ==========================================================


def make_predictions(model, test_df):
    future = test_df[["ds"]].copy()
    forecast = model.predict(future)
    predictions = forecast["yhat"].values

    return predictions, forecast


# ==========================================================
# Evaluation
# ==========================================================


def evaluate_model(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    return mae, rmse, mape


# ==========================================================
# Save Model
# ==========================================================


def save_model(model):
    model_path = Path(MODEL_DIR) / MODEL_NAME
    joblib.dump(model, model_path)
    return model_path


# ==========================================================
# Save Predictions
# ==========================================================


def save_predictions(test_df, predictions):
    prediction_df = pd.DataFrame(
        {
            "week": test_df["ds"],
            "actual": test_df["y"],
            "predicted": predictions,
        }
    )

    output_path = Path(OUTPUT_DIR) / PREDICTION_FILE
    prediction_df.to_csv(output_path, index=False)

    return prediction_df, output_path


# ==========================================================
# Plot Results
# ==========================================================


def plot_results(train_df, test_df, predictions):
    plt.figure(figsize=(14, 6))

    plt.plot(train_df["ds"], train_df["y"], label="Train")
    plt.plot(test_df["ds"], test_df["y"], label="Actual")
    plt.plot(test_df["ds"], predictions, label="Predicted")

    plt.title("Prophet Forecast")
    plt.xlabel("Week")
    plt.ylabel("Revenue")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = Path(OUTPUT_DIR) / "prophet_forecast.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return plot_path


# ==========================================================
# Summary
# ==========================================================


def print_summary(
    model, train_df, test_df, mae, rmse, mape, model_path, prediction_path
):
    print("\n==============================")
    print("Prophet Results")
    print("==============================")
    print(f"Train Rows      : {len(train_df)}")
    print(f"Test Rows       : {len(test_df)}")
    print(f"MAE             : {mae:,.2f}")
    print(f"RMSE            : {rmse:,.2f}")
    print(f"MAPE            : {mape:.2f}%")
    print(f"Model Saved     : {model_path}")
    print(f"Predictions CSV : {prediction_path}")
    print("==============================")


# ==========================================================
# Main
# ==========================================================


def main():
    print("\nStarting Prophet Training...\n")

    create_directories()

    df = load_data()
    train_df, test_df = train_test_split(df)

    train_prophet_df = prepare_prophet_data(train_df)
    test_prophet_df = prepare_prophet_data(test_df)

    model = train_prophet(train_prophet_df)
    predictions, forecast = make_predictions(model, test_prophet_df)

    mae, rmse, mape = evaluate_model(test_prophet_df["y"], predictions)

    model_path = save_model(model)
    prediction_df, prediction_path = save_predictions(test_prophet_df, predictions)

    plot_results(train_prophet_df, test_prophet_df, predictions)

    print_summary(
        model,
        train_prophet_df,
        test_prophet_df,
        mae,
        rmse,
        mape,
        model_path,
        prediction_path,
    )
if __name__ == "__main__":
    main()