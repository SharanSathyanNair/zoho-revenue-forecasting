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
    load_model_health,
)


# ==========================================================
# Page
# ==========================================================

st.set_page_config(
    page_title="Zoho Revenue Forecasting",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Zoho Revenue Forecasting Dashboard")

# ==========================================================
# Model Health / Drift Warning
# ==========================================================

model_health = load_model_health()

if model_health is not None and model_health.get("drift_detected"):
    st.warning(
        f"⚠️ Model drift detected — recent prediction stability is "
        f"{model_health['model_stability']:.1f}% (below the 75% threshold). "
        f"Forecasts and confidence intervals below may be less reliable than "
        f"usual. Consider retraining the model on more recent data.",
        icon="⚠️",
    )

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
    ]
)

if forecast_type == "Weeks":
  forecast_value = st.sidebar.slider(
      "Weeks Ahead",
      1,
      52,
      26,
  )

else:
  forecast_value = st.sidebar.slider(
    "Months Ahead",
    1,
    12,
    6,
  )


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
confidence = "95%"
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

forecast = pd.concat(
    [last_history, forecast],
    ignore_index=True,
)

# ==========================================================
# Forecast Graph
# ==========================================================

forecast_chart(
    history,
    forecast,
    forecast["lower_bound"],
    forecast["upper_bound"],
    
)

# ==========================================================
# Forecast Table
# ==========================================================

table = forecast[
    [
        "week",
        "weekly_revenue",
    ]
].copy()

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
    "Download Forecast CSV",
    forecast.to_csv(index=False),
    "forecast.csv",
    "text/csv",
)