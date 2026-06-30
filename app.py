import streamlit as st
import pandas as pd
import numpy as np

from components import (
    forecast_chart,
    metric_cards,
)

from utils import (
    load_history,
    get_forecast,
    calculate_growth,
)

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Zoho Revenue Forecasting",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Zoho Revenue Forecasting Dashboard")

st.markdown("---")

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.header("Forecast Settings")

forecast_type = st.sidebar.radio(
    "Forecast Type",
    [
        "Weeks",
        "Months",
    ],
)

if forecast_type == "Weeks":

    forecast_value = st.sidebar.slider(
        "Weeks Ahead",
        min_value=1,
        max_value=52,
        value=26,
    )

else:

    forecast_value = st.sidebar.slider(
        "Months Ahead",
        min_value=1,
        max_value=12,
        value=6,
    )

# ==========================================================
# Fixed Confidence
# ==========================================================

confidence = "95%"

# ==========================================================
# Load Data
# ==========================================================

history = load_history()

forecast = get_forecast(
    forecast_type=forecast_type,
    value=forecast_value,
)

current, future, growth = calculate_growth(
    history,
    forecast,
)

# ==========================================================
# KPI Cards
# ==========================================================

metric_cards(
    current=current,
    forecast=future,
    growth=growth,
    confidence=confidence,
)

st.markdown("---")

# ==========================================================
# Connect History to Forecast
# ==========================================================

last_history = history.iloc[[-1]].copy()

last_history["lower_bound"] = np.nan
last_history["upper_bound"] = np.nan

forecast_plot = pd.concat(
    [
        last_history,
        forecast,
    ],
    ignore_index=True,
)

# ==========================================================
# Forecast Graph
# ==========================================================

forecast_chart(
    history,
    forecast_plot,
    forecast_plot["lower_bound"],
    forecast_plot["upper_bound"],
)

# ==========================================================
# Forecast Table
# ==========================================================

st.subheader("Forecast")

table = forecast.copy()

table = table[
    [
        "week",
        "weekly_revenue",
    ]
]

table.columns = [
    "Week",
    "Predicted Revenue",
]

table["Week"] = (
    table["Week"]
    .dt.strftime("%d %b %Y")
)

table["Predicted Revenue"] = (
    "₹"
    + table["Predicted Revenue"]
    .round(0)
    .astype(int)
    .map("{:,}".format)
)

st.dataframe(
    table,
    use_container_width=True,
)

# ==========================================================
# Download
# ==========================================================

st.download_button(
    label="📥 Download Forecast CSV",
    data=forecast.to_csv(index=False),
    file_name="forecast.csv",
    mime="text/csv",
)