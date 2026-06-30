# 📈 AI-Powered Revenue Forecasting using Zoho Books and XGBoost

An AI-powered revenue forecasting system that predicts future weekly business revenue using historical accounting data from **Zoho Books**.

The project combines business intelligence, feature engineering, machine learning, hyperparameter tuning, recursive forecasting, and an interactive Streamlit dashboard to provide accurate revenue forecasts and business insights.


# 🚀 Features

- Weekly Revenue Forecasting up to **52 weeks (1 year)** ahead
- Feature Engineering using business KPIs
- XGBoost Machine Learning Model
- Hyperparameter Optimization using RandomizedSearchCV
- Walk-Forward Validation using TimeSeriesSplit
- Recursive Multi-Step Forecasting
- Conformal Prediction Intervals
- Interactive Streamlit Dashboard
- Forecast Download as CSV
- KPI Dashboard
- Business Trend Analysis


# 📊 Project Architecture

```
                    Zoho Books
                         │
                         ▼
          Business Transaction Data
                         │
                         ▼
             Weekly Business Metrics
                         │
                         ▼
              Feature Engineering
                         │
                         ▼
          Hyperparameter Tuning
                         │
                         ▼
               XGBoost Training
                         │
                         ▼
             Recursive Forecasting
                         │
                         ▼
             Prediction Intervals
                         │
                         ▼
             Streamlit Dashboard
```


# 📂 Project Structure

```
zoho-revenue-forecasting/
│
├── app.py
├── components.py
├── utils.py
│
├── feature_engineering.py
├── train_xgboost.py
├── tune_xgboost.py
├── forecast_engine.py
├── prediction_interval.py
│
├── data/
├── models/
├── outputs/
│
├── requirements.txt
└── README.md
```


# 📚 Data Source

The project uses **Zoho Books** as the primary data source.

Rather than training directly on raw accounting tables, the system aggregates business transactions into weekly business metrics.

## Zoho Books Modules Used

- Customers
- Invoices
- Payments Received
- Contacts


# 📈 Weekly Business Metrics

The following business metrics are generated from Zoho Books transactions.

| Metric | Description |
|---------|-------------|
| Weekly Revenue | Total invoice value generated during the week |
| Invoice Count | Number of invoices created |
| Average Invoice | Average invoice amount |
| Weekly Payments | Total payments received |
| Payment Count | Number of payments received |
| Average Payment | Average payment amount |
| Active Customers | Unique customers active during the week |
| New Customers | Customers added during the week |

These weekly metrics become the input for feature engineering.


# ⚙ Feature Engineering

The project creates **38 machine learning features** from weekly business metrics.

## Revenue Features

- Revenue Lag (1,2,4,8,12,26 weeks)
- Rolling Mean
- Rolling Standard Deviation
- Exponential Moving Average (EMA)
- Revenue Growth
- Revenue Acceleration
- Revenue Deviation
- Revenue Volatility
- Revenue vs Rolling Average


## Customer Features

- Active Customer Lag
- Customer Trend
- Revenue per Customer


## Invoice Features

- Invoice Count Lag
- Average Invoice Lag
- Invoice Trend
- Invoice Growth


## Payment Features

- Payment Lag
- Average Payment Lag
- Payment Trend
- Collection Rate
- Outstanding Revenue


## Calendar Features

Seasonality is captured using cyclical encoding.

- Month (Sin)
- Month (Cos)
- Week (Sin)
- Week (Cos)


# 🤖 Machine Learning Model

The forecasting engine is built using **XGBoost Regressor**.

## Why XGBoost?

- Excellent performance on structured business data
- Captures nonlinear business relationships
- Handles missing values effectively
- Built-in regularization reduces overfitting
- Fast training and prediction
- Highly interpretable feature importance


# 🔍 Hyperparameter Tuning

The model is optimized using:

- RandomizedSearchCV
- TimeSeriesSplit (Walk-Forward Validation)

Hyperparameters optimized include:

- Number of Trees
- Learning Rate
- Maximum Tree Depth
- Gamma
- Subsample
- Column Sampling
- Regularization
- Minimum Child Weight

The best parameter combination is automatically selected based on **lowest Mean Absolute Percentage Error (MAPE).**


# 📅 Walk-Forward Validation

Traditional random train-test splitting is not suitable for time-series forecasting.

Instead, this project uses **Walk-Forward Validation**.

```
Fold 1

Train ---------------- Test

Fold 2

Train ---------------------- Test

Fold 3

Train ---------------------------- Test
```

This ensures the model only learns from historical data and never sees future observations during training.

---

# 🔮 Forecasting Engine

After training, the model predicts future revenue recursively.

```
Predict Week 1

↓

Use Week 1 Prediction

↓

Predict Week 2

↓

Predict Week 3

↓

...

↓

Predict up to 52 Weeks
```

Each predicted value becomes part of the historical data used to forecast the next week.


# 📉 Prediction Intervals

The project generates prediction intervals using **Conformal Prediction**.

Instead of providing only a single revenue estimate, the model also calculates:

- Lower Bound
- Upper Bound
- Confidence Interval

This helps estimate the uncertainty associated with future forecasts.


# 📊 Dashboard Features

The Streamlit dashboard provides:

- Revenue Forecast Graph
- Historical vs Forecast Comparison
- KPI Cards
- Weekly Forecast Table
- Forecast Download
- Forecast Horizon Selection
- Confidence Metrics


# 📦 Technologies Used

- Python
- Pandas
- NumPy
- XGBoost
- Scikit-learn
- Streamlit
- Plotly
- Joblib


# 📈 Model Performance

Current XGBoost Performance

| Metric | Value |
|----------|--------|
| Training MAPE | ~2% |
| Testing MAPE | ~10% |
| Forecast Horizon | Up to 52 Weeks |
| Validation | Walk-Forward (TimeSeriesSplit) |

> Performance depends on the quality and quantity of historical business data available.

# ✅ Current Features

The project currently includes the following functionality:

### Data Processing
- Weekly business metric generation from Zoho Books transaction data
- Data validation and preprocessing
- Automated feature engineering pipeline
- Business KPI aggregation

### Machine Learning
- XGBoost Regressor for revenue forecasting
- Hyperparameter tuning using RandomizedSearchCV
- Walk-Forward Validation using TimeSeriesSplit
- Recursive multi-step forecasting (up to 52 weeks)
- Conformal Prediction Intervals
- Model evaluation using MAE, RMSE and MAPE

### Feature Engineering
- Revenue trend and momentum features
- Customer behaviour features
- Invoice trend features
- Payment and collection features
- Seasonality encoding using cyclical calendar features
- 38 engineered features generated from weekly business metrics

### Explainable AI
- SHAP feature importance analysis
- Global feature importance visualization
- Model interpretability

### Dashboard
- Interactive Streamlit dashboard
- Revenue forecast visualization
- Historical vs Forecast comparison
- KPI summary cards
- Weekly forecast table
- CSV export of forecast results
- Forecast horizon selection (up to one year)

### Business Intelligence
- Revenue trend analysis
- Customer trend analysis
- Invoice trend analysis
- Payment collection analysis
- Confidence interval estimation


# 🔄 Future Improvements

- Direct Zoho Books API Integration
- Automatic Model Retraining
- Multi-company Forecasting
- Cash Flow Forecasting
- Automated Scheduling
- Cloud deployment for organization-wide access
- Revenue driver recommendations based on feature importance
- Monthly, quarterly, and yearly forecasting options

# 👨‍💻 Author

**Sharan Sathyan Nair**

AI-powered Business Revenue Forecasting using Zoho Books and XGBoost.