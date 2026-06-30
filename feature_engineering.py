"""
Feature engineering pipeline for weekly revenue forecasting.

This module transforms raw weekly business metrics into a
model-ready dataset for XGBoost training and validation.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================================
# Configuration
# ==========================================================

INPUT_FILE: str = "data/weekly_business_metrics.csv"
OUTPUT_FILE: str = "data/model_ready_data.csv"


# ==========================================================
# Load Data
# ==========================================================

def load_data() -> pd.DataFrame:
    """
    Load the weekly business metrics dataset.
    """

    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    try:
        return pd.read_csv(
            INPUT_FILE,
            parse_dates=["week"],
        )

    except Exception as e:
        raise RuntimeError(
            "Unable to load input dataset."
        ) from e


# ==========================================================
# Input Validation
# ==========================================================

def validate_input(df: pd.DataFrame) -> None:
    """
    Validate the raw weekly dataset.
    """

    print("\nValidating weekly dataset...\n")

    if df.empty:
        raise ValueError("Input dataset is empty.")

    if not df["week"].is_unique:
        raise ValueError("Duplicate weeks detected.")

    if (df["weekly_revenue"] < 0).any():
        raise ValueError("Negative revenue detected.")

    if (df["active_customers"] < 0).any():
        raise ValueError("Negative customer count detected.")

    print("Validation passed.")


# ==========================================================
# Revenue Features
# ==========================================================

def create_revenue_features(df):

    revenue = df["weekly_revenue"]

    # Lag Features
    for lag in (1, 2, 4, 8, 12, 26):
        df[f"lag_{lag}"] = revenue.shift(lag)

    # Rolling Means
    shifted = revenue.shift(1)
    for window in (4, 8, 12):
        df[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        df[f"rolling_std_{window}"] = shifted.rolling(window).std()

    # Exponential Moving Averages
    df["ema_4"] = revenue.shift(1).ewm(span=4, adjust=False).mean()
    df["ema_12"] = revenue.shift(1).ewm(span=12, adjust=False).mean()

    # Revenue Momentum
    df["revenue_vs_4w_avg"] = (
        df["lag_1"] /
        df["rolling_mean_4"]
    )

    df["revenue_vs_8w_avg"] = (
        df["lag_1"] /
        df["rolling_mean_8"]
    )

    # Growth
    df["revenue_growth_4w"] = (
        df["lag_1"] /
        df["lag_4"]
    ) - 1

    # Acceleration
    df["revenue_acceleration"] = (
        df["lag_1"] -
        df["lag_2"]
    )

    # Deviation from Trend
    df["revenue_deviation"] = (
        df["lag_1"] -
        df["rolling_mean_12"]
    )

    # Volatility
    df["rolling_cv_4"] = (
        df["rolling_std_4"] /
        df["rolling_mean_4"]
    )

    return df
# ==========================================================
# Customer Features
# ==========================================================

def create_customer_features(df):
    df["active_customers_lag_1"] = df["active_customers"].shift(1)

    df["customer_trend_4"] = (
        df["active_customers_lag_1"]
        .rolling(4)
        .mean()
    )

    df["revenue_per_customer_lag_1"] = (
        df["weekly_revenue"].shift(1)
        /
        df["active_customers"].shift(1).replace(0, np.nan)
    )
    
    return df

# ==========================================================
# Invoice Features
# ==========================================================

def create_invoice_features(df):
    df["invoice_count_lag_1"] = (
        df["invoice_count"].shift(1)
    )

    df["average_invoice_lag_1"] = (
        df["average_invoice"].shift(1)
    )

    df["invoice_trend_4"] = (
        df["invoice_count_lag_1"]
        .rolling(4)
        .mean()
    )

    df["invoice_growth"] = (
        df["invoice_count_lag_1"] /
        df["invoice_count"].shift(4)
    ) - 1

    return df


# ==========================================================
# Payment Features
# ==========================================================

def create_payment_features(df):
    df["payment_lag_1"] = (
        df["weekly_payments"].shift(1)
    )

    df["average_payment_lag_1"] = (
        df["average_payment"].shift(1)
    )

    df["payment_trend_4"] = (
        df["payment_lag_1"]
        .rolling(4)
        .mean()
    )

    df["collection_rate_lag_1"] = (
        df["weekly_payments"].shift(1) /
        df["weekly_revenue"].shift(1).replace(0, np.nan)
    )

    df["outstanding_lag_1"] = (
        df["weekly_revenue"].shift(1) -
        df["weekly_payments"].shift(1)
    )
    
    return df 
# ==========================================================
# Calendar Features
# ==========================================================

def create_calendar_features(df):

    week = df["week"].dt.isocalendar().week.astype(int)

    df["month_sin"] = np.sin(
        2 * np.pi * df["week"].dt.month / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["week"].dt.month / 12
    )

    df["week_sin"] = np.sin(
        2 * np.pi * week / 52
    )

    df["week_cos"] = np.cos(
        2 * np.pi * week / 52
    )

    return df


# ==========================================================
# Clean Dataset
# ==========================================================

def clean_dataset(df):

    drop_columns = [
    "active_customers",
    "invoice_count",
    "average_invoice",
    "weekly_payments",
    "average_payment",
    ]

    df = df.drop(
        columns=drop_columns,
        errors="ignore",
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna()

    if df.empty:
        raise ValueError(
            "No rows remain after cleaning the dataset."
        )

    df = df.reset_index(
        drop=True
    )

    return df
# ==========================================================
# Export
# ==========================================================

def export_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Export the engineered dataset.
    """

    df = df.sort_values("week").reset_index(drop=True)

    try:
        df.to_csv(
            OUTPUT_FILE,
            index=False,
        )

    except Exception as e:
        raise RuntimeError(
            "Failed to export model-ready dataset."
        ) from e

    return df


# ==========================================================
# Output Validation
# ==========================================================

def validate_output(
    df: pd.DataFrame,
) -> None:
    """
    Validate the engineered dataset.
    """

    print("\nValidating model-ready dataset...\n")

    if df.empty:
        raise ValueError("Output dataset is empty.")

    if not df["week"].is_unique:
        raise ValueError("Duplicate weeks detected.")

    if df["weekly_revenue"].isna().any():
        raise ValueError("Revenue contains missing values.")

    if df.isnull().any().any():
        raise ValueError("Output dataset contains NaN values.")

    if not np.isfinite(
        df.select_dtypes(include=[np.number])
    ).all().all():
        raise ValueError(
            "Infinite values detected."
        )

    print("Validation passed.")


# ==========================================================
# Summary
# ==========================================================

def print_summary(df):

    print("\n==============================")
    print("Feature Engineering Summary")
    print("==============================")
    print(f"Rows                : {len(df)}")
    print(f"Columns             : {len(df.columns)}")
    print(f"Features            : {len(df.columns)-2}")
    print(f"Revenue Mean        : {df['weekly_revenue'].mean():,.2f}")
    print(f"Revenue Std         : {df['weekly_revenue'].std():,.2f}")
    print(f"Date Range          : {df['week'].min().date()} -> {df['week'].max().date()}")
    print("==============================")


# ==========================================================
# Main
# ==========================================================

def main() -> None:
    """
    Execute the complete feature engineering pipeline.
    """

    print("\nStarting feature engineering...\n")

    try:
        df = load_data()

        validate_input(df)

        df = create_revenue_features(df)
        print("Revenue:", type(df))

        df = create_customer_features(df)
        print("Customer:", type(df))

        df = create_invoice_features(df)
        print("Invoice:", type(df))

        df = create_payment_features(df)
        print("Payment:", type(df))

        df = create_calendar_features(df)
        print("Calendar:", type(df))

        df = clean_dataset(df)

        df = export_dataset(df)

        validate_output(df)

        print_summary(df)

        print(
            "\nmodel_ready_data.csv generated successfully."
        )

    except Exception as e:

        print("\n==============================")
        print("Feature Engineering Failed")
        print("==============================")
        print(e)
        print("==============================")
        raise

__all__ = [
    "load_data",
    "validate_input",
    "create_revenue_features",
    "create_customer_features",
    "create_invoice_features",
    "create_payment_features",
    "create_calendar_features",
    "clean_dataset",
    "export_dataset",
    "validate_output",
    "print_summary",
    "main",
]

if __name__ == "__main__":
    main()