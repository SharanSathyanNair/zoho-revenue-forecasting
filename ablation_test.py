"""
Ablation Test: Customer / Churn-Derived Features

Question being tested:
Are the churn-derived features (active_customers_lag_1, customer_trend_4,
revenue_per_customer_lag_1) actually helping the model, or are they
injecting noise that hurts stability without meaningfully improving
accuracy?

Method:
Train two otherwise-identical XGBoost models (same tuned hyperparameters,
same train/test boundaries) -- one with the full feature set, one with
the three customer/churn-derived columns removed. Compare both on:

  1. The same 12-week holdout used by train_xgboost.py
  2. The same walk-forward windows used by walk_forward_validation.py

Both models are evaluated on identical splits, so any difference in
MAPE or stability is attributable to the dropped features, not to
different data or different hyperparameters.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from prediction_interval import (
    calculate_residuals,
    calculate_recent_model_stability,
)

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
PARAM_FILE = "models/best_xgb_params.json"
OUTPUT_DIR = "outputs"

HOLDOUT_WEEKS = 12
INITIAL_TRAIN_SIZE = 104
FORECAST_HORIZON = 1
RANDOM_STATE = 42

CHURN_DERIVED_FEATURES = [
    "active_customers_lag_1",
    "customer_trend_4",
    "revenue_per_customer_lag_1",
]


# ==========================================================
# Load Data / Params
# ==========================================================

def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE, parse_dates=["week"])
    return df.sort_values("week").reset_index(drop=True)


def load_parameters() -> dict:
    path = Path(PARAM_FILE)
    if not path.exists():
        raise FileNotFoundError("Run xgboost_tuning.py first.")
    with open(path, "r") as f:
        return json.load(f)


# ==========================================================
# Feature Preparation
# ==========================================================

def prepare_features(df: pd.DataFrame, drop_churn_features: bool):
    drop_columns = ["week", "weekly_revenue"]

    if drop_churn_features:
        drop_columns = drop_columns + CHURN_DERIVED_FEATURES

    X = df.drop(columns=drop_columns, errors="ignore")
    y = df["weekly_revenue"]
    return X, y


def build_model(params: dict) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **params,
    )


# ==========================================================
# Holdout Evaluation
# ==========================================================

def run_holdout(df, params, drop_churn_features):

    train = df.iloc[:-HOLDOUT_WEEKS].copy()
    test = df.iloc[-HOLDOUT_WEEKS:].copy()

    X_train, y_train = prepare_features(train, drop_churn_features)
    X_test, y_test = prepare_features(test, drop_churn_features)

    model = build_model(params)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = mean_absolute_percentage_error(y_test, predictions) * 100

    stability = calculate_recent_model_stability(y_test, predictions)

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "stability": stability,
        "n_features": X_train.shape[1],
    }


# ==========================================================
# Walk-Forward Evaluation
# ==========================================================

def run_walk_forward(df, params, drop_churn_features):

    predictions = []
    actuals = []
    dates = []

    for i in range(INITIAL_TRAIN_SIZE, len(df)):

        train = df.iloc[:i]
        test = df.iloc[i:i + FORECAST_HORIZON]

        if len(test) == 0:
            break

        X_train, y_train = prepare_features(train, drop_churn_features)
        X_test, y_test = prepare_features(test, drop_churn_features)

        model = build_model(params)
        model.fit(X_train, y_train)

        prediction = model.predict(X_test)[0]

        predictions.append(prediction)
        actuals.append(y_test.iloc[0])
        dates.append(test["week"].iloc[0])

    results = pd.DataFrame({
        "week": dates,
        "actual": actuals,
        "predicted": predictions,
    })

    mae = mean_absolute_error(results["actual"], results["predicted"])
    rmse = np.sqrt(mean_squared_error(results["actual"], results["predicted"]))
    mape = mean_absolute_percentage_error(
        results["actual"], results["predicted"]
    ) * 100

    stability = calculate_recent_model_stability(
        results["actual"], results["predicted"]
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "stability": stability,
        "n_predictions": len(results),
    }, results


# ==========================================================
# Comparison Printer
# ==========================================================

def print_comparison(title, full_result, ablated_result, extra_key, extra_label):

    print(f"\n{title}")
    print("=" * 60)
    print(f"{'Metric':<22}{'With Churn Features':>20}{'Without':>18}")
    print("-" * 60)
    print(f"{'MAE':<22}{full_result['mae']:>20,.2f}{ablated_result['mae']:>18,.2f}")
    print(f"{'RMSE':<22}{full_result['rmse']:>20,.2f}{ablated_result['rmse']:>18,.2f}")
    print(f"{'MAPE (%)':<22}{full_result['mape']:>20.2f}{ablated_result['mape']:>18.2f}")
    print(f"{'Stability (%)':<22}{full_result['stability']:>20.2f}{ablated_result['stability']:>18.2f}")
    print(f"{extra_label:<22}{full_result[extra_key]:>20}{ablated_result[extra_key]:>18}")
    print("=" * 60)

    mape_delta = ablated_result["mape"] - full_result["mape"]
    stability_delta = ablated_result["stability"] - full_result["stability"]

    print(f"\nMAPE change (without - with)      : {mape_delta:+.2f} points")
    print(f"Stability change (without - with) : {stability_delta:+.2f} points")

    if stability_delta > 2 and mape_delta < 2:
        print(
            "\n-> Removing churn-derived features improved stability "
            "without meaningfully hurting accuracy. These features "
            "appear to be injecting more noise than signal."
        )
    elif stability_delta < -2:
        print(
            "\n-> Removing churn-derived features made stability worse. "
            "These features appear to carry real, useful signal."
        )
    else:
        print(
            "\n-> No strong difference either way. The churn-derived "
            "features are not the main driver of instability."
        )


# ==========================================================
# Main
# ==========================================================

def main():

    print("\nStarting Ablation Test: Churn-Derived Features\n")

    df = load_data()
    params = load_parameters()

    # ------------------------------------------------------
    # Holdout comparison
    # ------------------------------------------------------

    print("Running 12-week holdout (with vs. without churn features)...")

    holdout_full = run_holdout(df, params, drop_churn_features=False)
    holdout_ablated = run_holdout(df, params, drop_churn_features=True)

    print_comparison(
        "12-Week Holdout Comparison",
        holdout_full,
        holdout_ablated,
        extra_key="n_features",
        extra_label="Features Used",
    )

    # ------------------------------------------------------
    # Walk-forward comparison
    # ------------------------------------------------------

    print("\n\nRunning walk-forward validation (with vs. without churn features)...")
    print("This retrains the model at every step for both variants -- may take a while.\n")

    wf_full, results_full = run_walk_forward(df, params, drop_churn_features=False)
    wf_ablated, results_ablated = run_walk_forward(df, params, drop_churn_features=True)

    print_comparison(
        "Walk-Forward Validation Comparison",
        wf_full,
        wf_ablated,
        extra_key="n_predictions",
        extra_label="Predictions Made",
    )

    # ------------------------------------------------------
    # Save results for inspection
    # ------------------------------------------------------

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    results_full.rename(columns={"predicted": "predicted_with_churn_features"}, inplace=True)
    results_ablated.rename(columns={"predicted": "predicted_without_churn_features"}, inplace=True)

    comparison = results_full.merge(
        results_ablated[["week", "predicted_without_churn_features"]],
        on="week",
    )

    output_path = Path(OUTPUT_DIR) / "ablation_walk_forward_comparison.csv"
    comparison.to_csv(output_path, index=False)

    print(f"\nPer-week comparison saved to: {output_path}")


if __name__ == "__main__":
    main()