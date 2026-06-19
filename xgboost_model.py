#______importing libraries__________
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import shap

#______loading featured weekly revenue data__________
df = pd.read_csv("data/featured_weekly_revenue.csv")
df["week_start"] = pd.to_datetime(df["week_start"])

#______defining feature columns and target__________
feature_cols = ["lag_1", "lag_2", "lag_4", "lag_8", "lag_12", "lag_52",
                 "rolling_mean_4", "rolling_mean_12", "rolling_mean_26",
                 "rolling_std_4", "rolling_std_12", "rolling_max_4",
                 "week_of_year", "month", "quarter", "is_month_end",
                 "is_quarter_end", "is_december", "is_january",
                 "weeks_elapsed", "wow_change", "revenue_vs_4w_avg",
                 "revenue_pct_change_4", "revenue_pct_change_52",
                 "active_customer_count", "total_open_balance",
                 "invoices_due_next_4_weeks"]

X = df[feature_cols]
y = df["revenue"]

#______time-aware train/test split__________
train_size = int(len(df) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]
y_test = y_test.reset_index(drop=True)

#______training XGBoost model with previously tuned parameters__________
model = xgb.XGBRegressor(
    n_estimators=575,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.9,
    random_state=42
)

model.fit(X_train, y_train)

#______generating predictions on test set__________
y_pred = model.predict(X_test)

#______calculating evaluation metrics__________
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

#______plotting actual vs predicted__________
test_dates = df["week_start"][train_size:].reset_index(drop=True)

plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test.values, label="Actual", color="black")
plt.plot(test_dates, y_pred, label="XGBoost Forecast (New Features)", color="blue")
plt.legend()
plt.title("XGBoost (New Features): Actual vs Predicted Revenue")
plt.xlabel("Week")
plt.ylabel("Revenue")
plt.savefig("outputs/xgboost_forecast_newfeatures.png", bbox_inches="tight")
plt.show()

#______SHAP explainability__________
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

#______summary plot — overall feature importance__________
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("outputs/shap_summary_newfeatures.png", bbox_inches="tight")
plt.show()