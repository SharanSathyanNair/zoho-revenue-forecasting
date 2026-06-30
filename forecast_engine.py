"""
Recursive Forecast Engine

Loads the trained XGBoost model and generates
future weekly revenue forecasts until a user-
specified week or month.
"""

import joblib
import pandas as pd
import numpy as np

from pathlib import Path

from prediction_interval import (
    calculate_residuals,
    calculate_conformal_quantile,
    generate_horizon_scaled_interval,
)

# ==========================================================
# Configuration
# ==========================================================

MODEL_FILE = "models/xgboost_model.pkl"
DATA_FILE = "data/model_ready_data.csv"
OUTPUT_FILE = "outputs/future_forecast.csv"

TREND_WINDOW = 8

# ==========================================================
# Load Model
# ==========================================================

def load_model():

    model_path = Path(MODEL_FILE)

    if not model_path.exists():
        raise FileNotFoundError("Train the XGBoost model first.")

    return joblib.load(model_path)


# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset():

    data_path = Path(DATA_FILE)

    if not data_path.exists():
        raise FileNotFoundError("model_ready_data.csv not found.")

    df = pd.read_csv(
        data_path,
        parse_dates=["week"],
    )

    df = df.sort_values("week").reset_index(drop=True)

    return df

# ==========================================================
# Weighted Moving Average
# ==========================================================

def weighted_recent_average(series, window=TREND_WINDOW):

    recent = series.tail(window).to_numpy()

    if len(recent) == 0:
        return 0.0

    weights = np.arange(1, len(recent) + 1)
    weights = weights / weights.sum()

    return np.average(
        recent,
        weights=weights,
    )

# ==========================================================
# Safe Lag
# ==========================================================

def safe_lag(series, lag):

    if len(series) < lag:
        return series.iloc[0]

    return series.iloc[-lag]


# ==========================================================
# Forecast Horizon
# ==========================================================

def calculate_forecast_horizon(
    last_week,
    forecast_type,
    weeks=None,
    target_month=None,
    target_year=None,
):

    if forecast_type == "weeks":
        return weeks

    target_date = (
        pd.Timestamp(
            year=target_year,
            month=target_month,
            day=1,
        )
        + pd.offsets.MonthEnd(0)
    )

    horizon = (
        (target_date - last_week).days // 7
    )

    return max(1, horizon)
# ==========================================================
# Generate Next Week
# ==========================================================

def generate_next_week(history):

    latest = history.iloc[-1].copy()

    next_week = latest.copy()

    next_week["week"] = latest["week"] + pd.Timedelta(days=7)

    return next_week
# ==========================================================
# Update Future Features
# ==========================================================

def update_business_features(history, next_week):

    # ------------------------------------------------------
    # Customer Trend
    # ------------------------------------------------------

    next_week["active_customers_lag_1"] = weighted_recent_average(
            history["active_customers_lag_1"]   
    )

    next_week["customer_trend_4"] = weighted_recent_average(
            history["customer_trend_4"]
    )

    next_week["revenue_per_customer_lag_1"] = weighted_recent_average(
            history["revenue_per_customer_lag_1"]
    )

    next_week["new_customers_lag_1"] = weighted_recent_average(
            history["new_customers_lag_1"]
    )

    # ------------------------------------------------------
    # Invoice Trend
    # ------------------------------------------------------

    next_week["invoice_count_lag_1"] = weighted_recent_average(
        history["invoice_count_lag_1"]
    )

    next_week["average_invoice_lag_1"] = weighted_recent_average(
        history["average_invoice_lag_1"]
    )

    next_week["invoice_trend_4"] = weighted_recent_average(
        history["invoice_trend_4"]
    )

    next_week["invoice_growth"] = weighted_recent_average(
        history["invoice_growth"]
    )

    # ------------------------------------------------------
    # Payment Trend
    # ------------------------------------------------------

    next_week["payment_lag_1"] = weighted_recent_average(
        history["payment_lag_1"]
    )

    next_week["average_payment_lag_1"] = weighted_recent_average(
        history["average_payment_lag_1"]
    )

    next_week["payment_trend_4"] = weighted_recent_average(
        history["payment_trend_4"]
    )

    next_week["collection_rate_lag_1"] = weighted_recent_average(
        history["collection_rate_lag_1"]
    )

    next_week["outstanding_lag_1"] = weighted_recent_average(
        history["outstanding_lag_1"]
    )

    next_week["payment_count_lag_1"] = weighted_recent_average(
        history["payment_count_lag_1"]
    )

    # ------------------------------------------------------
    # Calendar Features
    # ------------------------------------------------------

    month = next_week["week"].month
    week = next_week["week"].isocalendar().week

    next_week["month_sin"] = np.sin(
        2 * np.pi * month / 12
    )

    next_week["month_cos"] = np.cos(
        2 * np.pi * month / 12
    )

    next_week["week_sin"] = np.sin(
        2 * np.pi * week / 52
    )

    next_week["week_cos"] = np.cos(
        2 * np.pi * week / 52
    )

    return next_week
# ==========================================================
# Update Revenue Features
# ==========================================================

def update_revenue_features(history, next_week):

    revenue = history["weekly_revenue"]

    next_week["lag_1"] = safe_lag(revenue, 1)
    next_week["lag_2"] = safe_lag(revenue, 2)
    next_week["lag_4"] = safe_lag(revenue, 4)
    next_week["lag_8"] = safe_lag(revenue, 8)
    next_week["lag_12"] = safe_lag(revenue, 12)
    next_week["lag_26"] = safe_lag(revenue, 26)

    next_week["rolling_mean_4"] = revenue.tail(4).mean()
    next_week["rolling_std_4"] = revenue.tail(4).std()

    next_week["rolling_mean_8"] = revenue.tail(8).mean()
    next_week["rolling_std_8"] = revenue.tail(8).std()

    next_week["rolling_mean_12"] = revenue.tail(12).mean()
    next_week["rolling_std_12"] = revenue.tail(12).std()

    next_week["ema_4"] = (
        revenue.ewm(span=4, adjust=False)
        .mean()
        .iloc[-1]
    )

    next_week["ema_12"] = (
        revenue.ewm(span=12, adjust=False)
        .mean()
        .iloc[-1]
    )

    next_week["revenue_vs_4w_avg"] = (
        next_week["lag_1"] /
        next_week["rolling_mean_4"]
    )

    next_week["revenue_vs_8w_avg"] = (
        next_week["lag_1"] /
        next_week["rolling_mean_8"]
    )

    next_week["revenue_growth_4w"] = (
        next_week["lag_1"] /
        next_week["lag_4"]
    ) - 1

    next_week["revenue_acceleration"] = (
        next_week["lag_1"] -
        next_week["lag_2"]
    )

    next_week["revenue_deviation"] = (
        next_week["lag_1"] -
        next_week["rolling_mean_12"]
    )

    next_week["rolling_cv_4"] = (
        next_week["rolling_std_4"] /
        next_week["rolling_mean_4"]
    )

    return next_week

# ==========================================================
# Recursive Forecast
# ==========================================================

def recursive_forecast(
    model,
    history,
    forecast_horizon,
):

    history = history.copy()

    forecasts = []

    for _ in range(forecast_horizon):

      # ------------------------------------------------------
      # Generate next week
      # ------------------------------------------------------

      next_week = generate_next_week(history)

      # ------------------------------------------------------
      # Update business trend features
      # ------------------------------------------------------

      next_week = update_business_features(
          history,
          next_week,
      )

      # ------------------------------------------------------
      # Update revenue features
      # ------------------------------------------------------

      next_week = update_revenue_features(
          history,
          next_week,
      )

      # ------------------------------------------------------
      # Prepare feature vector
      # ------------------------------------------------------

      X = (
          pd.DataFrame([next_week])
          .drop(
              columns=[
                  "week",
                  "weekly_revenue",
              ],
              errors="ignore",
          )
      )

      # ------------------------------------------------------
      # Predict revenue
      # ------------------------------------------------------

      prediction = float(
          model.predict(X)[0]
      )

      next_week["weekly_revenue"] = prediction

      # ------------------------------------------------------
      # Save prediction
      # ------------------------------------------------------

      forecasts.append(
          next_week.copy()
      )

      # ------------------------------------------------------
      # Append prediction to history
      # ------------------------------------------------------

      history = pd.concat(
          [
              history,
              pd.DataFrame([next_week]),
          ],
          ignore_index=True,
      )

    return pd.DataFrame(forecasts)

# ==========================================================
# Forecast with Prediction Interval
# ==========================================================

def forecast_with_confidence(
    model,
    history,
    forecast_horizon,
    residuals,
):

    forecast = recursive_forecast(
        model,
        history,
        forecast_horizon,
    )

    quantile = calculate_conformal_quantile(
        residuals
    )

    lower, upper = generate_horizon_scaled_interval(
        forecast["weekly_revenue"].values,
        quantile,
    )

    forecast["lower_bound"] = lower
    forecast["upper_bound"] = upper

    forecast["confidence"] = 95.0

    forecast["best_case"] = upper
    forecast["worst_case"] = lower

    return forecast
# ==========================================================
# Forecast Summary
# ==========================================================

def print_forecast_summary(forecast):

    print("\n==========================================")
    print("Forecast Summary")
    print("==========================================")

    print(f"Forecast Weeks : {len(forecast)}")
    print(f"Start Week     : {forecast.iloc[0]['week'].date()}")
    print(f"End Week       : {forecast.iloc[-1]['week'].date()}")

    print()

    print(
        f"Final Revenue  : "
        f"{forecast.iloc[-1]['weekly_revenue']:,.2f}"
    )

    print(
        f"Lower Bound    : "
        f"{forecast.iloc[-1]['lower_bound']:,.2f}"
    )

    print(
        f"Upper Bound    : "
        f"{forecast.iloc[-1]['upper_bound']:,.2f}"
    )

    print("Confidence     : 95%")

    print("==========================================")

# ==========================================================
# Save Forecast
# ==========================================================

def save_forecast(forecast_df):

    Path("outputs").mkdir(exist_ok=True)

    forecast_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    return forecast_df

# ==========================================================
# Run Forecast
# ==========================================================

def run_forecast(
    forecast_type="weeks",
    weeks=26,
    target_month=None,
    target_year=None,
):

    model = load_model()

    history = load_dataset()

    horizon = calculate_forecast_horizon(
        last_week=history.iloc[-1]["week"],
        forecast_type=forecast_type,
        weeks=weeks,
        target_month=target_month,
        target_year=target_year,
    )

    residuals = np.load(
        Path("outputs") / "residuals.npy"
    )

    forecast = forecast_with_confidence(
        model=model,
        history=history,
        forecast_horizon=horizon,
        residuals=residuals,
    )

    save_forecast(forecast)

    return forecast
# ==========================================================
# Main
# ==========================================================

def main():

    print("\nStarting Forecast Engine...\n")

    forecast = run_forecast(
        forecast_type="weeks",
        weeks=26,
    )

    print_forecast_summary(forecast)


if __name__ == "__main__":
    main()