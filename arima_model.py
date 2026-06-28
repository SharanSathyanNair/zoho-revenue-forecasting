import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pmdarima import auto_arima
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")


# ==========================================================
# Configuration
# ==========================================================

INPUT_FILE = "data/model_ready_data.csv"

MODEL_DIR = "models"

OUTPUT_DIR = "outputs"

MODEL_NAME = "arima_model.pkl"

PREDICTION_FILE = "arima_predictions.csv"

TEST_SIZE = 0.20


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
    FORECAST_HORIZON = 12
    train = df.iloc[:-FORECAST_HORIZON].copy()
    test = df.iloc[-FORECAST_HORIZON:].copy()
    return train, test


# ==========================================================
# Stationarity Test
# ==========================================================


def adf_test(series):
    result = adfuller(series)

    print("\nADF Test")
    print("-------------------------")
    print(f"ADF Statistic : {result[0]:.4f}")
    print(f"P-Value       : {result[1]:.4f}")
    print("-------------------------")

    return result


# ==========================================================
# Train Model
# ==========================================================


def train_arima(train):
    model = auto_arima(
        train,
        seasonal=False,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
        information_criterion="aic",
    )

    return model


# ==========================================================
# Prediction
# ==========================================================


def make_predictions(model, test):
    predictions = model.predict(n_periods=len(test))
    return predictions


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


def save_predictions(test, predictions):
    prediction_df = pd.DataFrame(
        {
            "week": test["week"],
            "actual": test["weekly_revenue"],
            "predicted": predictions,
        }
    )

    output_path = Path(OUTPUT_DIR) / PREDICTION_FILE
    prediction_df.to_csv(output_path, index=False)

    return prediction_df, output_path


# ==========================================================
# Plot Results
# ==========================================================


def plot_results(train, test, predictions):
    plt.figure(figsize=(14, 6))
    plt.plot(train["week"], train["weekly_revenue"], label="Train")
    plt.plot(test["week"], test["weekly_revenue"], label="Actual")
    plt.plot(test["week"], predictions, label="Predicted")
    plt.title("ARIMA Forecast")
    plt.xlabel("Week")
    plt.ylabel("Revenue")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = Path(OUTPUT_DIR) / "arima_forecast.png"
    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# ==========================================================
# Summary
# ==========================================================


def print_summary(model, train, test, mae, rmse, mape, model_path, prediction_path):
    print("\n==============================")
    print("ARIMA Results")
    print("==============================")
    print(f"Train Rows      : {len(train)}")
    print(f"Test Rows       : {len(test)}")
    print(f"Model Order     : {model.order}")
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
    print("\nStarting ARIMA Training...\n")

    create_directories()

    df = load_data()
    train_df, test_df = train_test_split(df)

    train_series = train_df["weekly_revenue"]
    test_series = test_df["weekly_revenue"]

    adf_test(train_series)

    model = train_arima(train_series)
    print(f"\nSelected ARIMA Order: {model.order}\n")
    predictions = make_predictions(model, test_df)

    mae, rmse, mape = evaluate_model(test_series, predictions)

    model_path = save_model(model)
    prediction_df, prediction_path = save_predictions(test_df, predictions)

    plot_results(train_df, test_df, predictions)

    print_summary(
        model,
        train_df,
        test_df,
        mae,
        rmse,
        mape,
        model_path,
        prediction_path,
    )


if __name__ == "__main__":
    main()