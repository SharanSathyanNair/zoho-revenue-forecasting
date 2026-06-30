from pathlib import Path

import json

import pandas as pd

from forecast_engine import run_forecast


def load_history():

    path = Path("data/weekly_business_metrics.csv")

    history = pd.read_csv(
        path,
        parse_dates=["week"],
    )

    return history


def load_model_health():
    """
    Load the latest saved forecast summary, which contains
    model_stability and drift_detected (written by train_xgboost.py).

    Returns None if the summary hasn't been generated yet, so the
    dashboard can simply skip the warning instead of crashing.
    """

    path = Path("outputs/forecast_summary.json")

    if not path.exists():
        return None

    with open(path, "r") as f:
        return json.load(f)


def get_forecast(
    forecast_type,
    value,
):

    if forecast_type == "Weeks":

        return run_forecast(
            forecast_type="weeks",
            weeks=value,
        )

    else:

        history = load_history()

        last_date = history.iloc[-1]["week"]

        target_date = (
            last_date +
            pd.DateOffset(months=value)
        )

        return run_forecast(
            forecast_type="months",
            target_month=target_date.month,
            target_year=target_date.year,
        )


def calculate_growth(
    history,
    forecast,
):

    current = history.iloc[-1]["weekly_revenue"]

    future = forecast.iloc[-1]["weekly_revenue"]

    growth = (
        (future - current)
        / current
        * 100
    )

    return current, future, growth