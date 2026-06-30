import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
OUTPUT_DIR = "outputs"
FORECAST_HORIZON = 1
INITIAL_TRAIN_SIZE = 104
RANDOM_STATE = 42


# ==========================================================
# Conformal Prediction
# ==========================================================

CALIBRATION_SIZE = 20
CONFIDENCE_LEVEL = 0.95

# ==========================================================
# Directories
# ==========================================================

def create_directories():
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
# Load Parameters
# ==========================================================

# ==========================================================
# Load Parameters
# ==========================================================

def load_parameters() -> dict:
    """Load tuned XGBoost parameters."""

    path = Path(PARAM_FILE)

    if not path.exists():
        raise FileNotFoundError(
            "Run xgboost_tuning.py before walk_forward_validation.py."
        )

    with open(path, "r") as f:
        return json.load(f)


# ==========================================================
# Feature Preparation
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
# Calibration Split
# ==========================================================

def split_calibration(train_df):

    model_train = train_df.iloc[:-CALIBRATION_SIZE]
    calibration = train_df.iloc[-CALIBRATION_SIZE:]
    return model_train, calibration

# ==========================================================
# Conformal Calibration
# ==========================================================

def calibrate_model(model, calibration):

    X_cal, y_cal = prepare_features(
        calibration
    )

    predictions = model.predict(
        X_cal
    )

    residuals = np.abs(
        y_cal - predictions
    )

    return residuals

# ==========================================================
# Build Model
# ==========================================================

def build_model(params):

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **params,
    )
    return model

# ==========================================================
# Walk Forward Validation
# ==========================================================

def walk_forward_validation(df, params):

    predictions = []
    actuals = []
    dates = []
    X_all, y_all = prepare_features(df)
    for i in range(
        INITIAL_TRAIN_SIZE,
        len(df),
    ):

        train = df.iloc[:i]
        test = df.iloc[i:i + FORECAST_HORIZON]
        if len(test) == 0:
            break

        X_train, y_train = prepare_features(train)
        X_test, y_test = prepare_features(test)
        model = build_model(params)
        model.fit(
            X_train,
            y_train,
        )

        prediction = model.predict(X_test)[0]
        predictions.append(
            prediction
        )

        actuals.append(
            y_test.iloc[0]
        )

        dates.append(
            test["week"].iloc[0]
        )

    results = pd.DataFrame({
        "week": dates,
        "actual": actuals,
        "predicted": predictions,
    })
    return results

# ==========================================================
# Evaluation
# ==========================================================

def evaluate(results):

    mae = mean_absolute_error(
        results["actual"],
        results["predicted"],
    )

    mse = mean_squared_error(
        results["actual"],
        results["predicted"],
    )

    rmse = np.sqrt(mse)
    mape = (
        mean_absolute_percentage_error(
            results["actual"],
            results["predicted"],
        )
        * 100
    )
    return mae, rmse, mape

# ==========================================================
# Save Results
# ==========================================================

def save_results(
    results: pd.DataFrame,
    lower,
    upper,
    confidence: float,
    stability: float,
):

    output = Path(OUTPUT_DIR) / "walk_forward_predictions.csv"
    results = results.copy()
    results["lower_bound"] = lower
    results["upper_bound"] = upper
    results["prediction_error"] = (
        results["actual"] -
        results["predicted"]
    )
    results["forecast_confidence"] = confidence
    results["model_stability"] = stability
    results.to_csv(
        output,
        index=False,
    )
    return output

# ==========================================================
# Walk Forward Plot
# ==========================================================

def plot_results(results,lower,upper):

    plt.figure(figsize=(16, 7))

    plt.plot(

        results["week"],

        results["actual"],

        color="black",

        linewidth=2.5,

        label="Actual",

    )

    plt.plot(

        results["week"],

        results["predicted"],

        color="blue",

        linewidth=2,

        label="Walk-Forward Prediction",

    )
    plt.fill_between(
        results["week"],
        lower,
        upper,
        color="dodgerblue",
        alpha=0.20,
        label="Prediction Interval",
    )

# ==========================================================
# Rolling RMSE Prediction Interval
# ==========================================================

    errors = (

      results["actual"] -

      results["predicted"]

    )

    rolling_rmse = (

        errors.shift(1).pow(2)

        .rolling(

            window=12,

            min_periods=4,

        )

        .mean()

        .pow(0.5)

    )

    # Handle the first few predictions where
    # there are not enough historical errors.

    rolling_rmse = rolling_rmse.bfill()

    upper = (

      results["predicted"]+ rolling_rmse

    )

    lower = (

        results["predicted"]

        - rolling_rmse

    )

    plt.fill_between(

        results["week"],

        lower,

        upper,

        color="deepskyblue",

        alpha=0.25,

        label="Rolling RMSE Interval",

    )

    plt.title(
        "Walk-Forward Validation with Rolling RMSE Prediction Interval"
    )

    plt.xlabel("Week")

    plt.ylabel("Revenue")

    plt.legend()

    plt.grid(alpha=0.30)

    plt.tight_layout()

    plt.savefig(

        Path(OUTPUT_DIR)
        / "walk_forward_plot.png",

        dpi=300,

        bbox_inches="tight",

    )

    plt.close()

# ==========================================================
# Summary
# ==========================================================

def print_summary(
    results,
    mae,
    rmse,
    mape,
    confidence,
    stability,
    drift_detected,
):
    print("\n=====================================================")
    print("Walk-Forward Validation Results")
    print("=====================================================")
    print(f"Predictions Made        : {len(results)}")
    print(f"Start Week              : {results['week'].min().date()}")
    print(f"End Week                : {results['week'].max().date()}")
    print("\nPerformance")
    print("------------------------------")
    print(f"MAE                     : {mae:,.2f}")
    print(f"RMSE                    : {rmse:,.2f}")
    print(f"MAPE                    : {mape:.2f}%")
    print("\nForecast Quality")
    print("------------------------------")
    print(f"Forecast Confidence     : {confidence:.2f}%")
    print(f"Model Stability         : {stability:.2f}%")
    print(f"Drift Detected          : {drift_detected}")
    print("\nArtifacts")
    print("------------------------------")
    print("Predictions             : outputs/walk_forward_predictions.csv")
    print("Plot                    : outputs/walk_forward_plot.png")
    print("Residuals (calibration) : outputs/residuals.npy")
    print("Forecast Summary        : outputs/forecast_summary.json")
    print("=====================================================")

# ==========================================================
# Main
# ==========================================================

def main() -> None:
    print("\nStarting Walk-Forward Validation...\n")
    try:
        create_directories()
        df = load_data()
        params = load_parameters()
        validation_results = walk_forward_validation(
            df,
            params,
        )

        mae, rmse, mape = evaluate(
            validation_results,
        )

        residuals = calculate_residuals(
            validation_results["actual"],
            validation_results["predicted"],
        )

        quantile = calculate_conformal_quantile(
            residuals,
        )

        lower, upper = generate_prediction_interval(
            validation_results["predicted"],
            quantile,
        )

        confidence = calculate_forecast_confidence(
            quantile,
            validation_results["predicted"],
        )

        stability = calculate_recent_model_stability(
            validation_results["actual"],
            validation_results["predicted"],
        )

        # ------------------------------------------------------
        # Overwrite outputs/residuals.npy with walk-forward
        # residuals (built from many rolling-origin windows,
        # not just one 12-week holdout). forecast_engine.py
        # reads this file to calibrate the live forecast
        # interval, so this makes the dashboard's interval and
        # drift signal reflect the more statistically robust
        # walk-forward evaluation.
        # ------------------------------------------------------

        np.save(
            Path(OUTPUT_DIR) / "residuals.npy",
            residuals,
        )

        drift_detected = check_model_drift(stability)

        forecast_summary = generate_forecast_summary(
            forecast=validation_results["predicted"].iloc[-1],
            lower=lower[-1],
            upper=upper[-1],
            confidence=confidence,
            stability=stability,
        )

        summary_path = Path(OUTPUT_DIR) / "forecast_summary.json"

        with open(summary_path, "w") as f:
            json.dump(forecast_summary, f, indent=4)

        save_results(
            validation_results,
            lower,
            upper,
            confidence,
            stability,
        )

        plot_results(
            validation_results,
            lower,
            upper,
        )

        print_summary(
            validation_results,
            mae,
            rmse,
            mape,
            confidence,
            stability,
            drift_detected,
        )

    except Exception as e:
        print("\n===================================")
        print("Walk Forward Validation Failed")
        print("===================================")
        print(e)
        print("===================================")
        raise

if __name__ == "__main__":
    main()