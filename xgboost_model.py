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
                "revenue_pct_change_4", "revenue_pct_change_52"]

X = df[feature_cols]
y = df["revenue"]

#______time-aware train/test split__________
train_size = int(len(df) * 0.8)

X_train = X[:train_size]
X_test  = X[train_size:]
y_train = y[:train_size]
y_test  = y[train_size:]
y_test  = y_test.reset_index(drop=True)

#______training XGBoost model with tuned parameters__________
model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=3,
    learning_rate=0.01,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

#______generating predictions on test set__________
y_pred = model.predict(X_test)

#______calculating evaluation metrics__________
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

#______plotting actual vs predicted__________
test_dates = df["week_start"][train_size:].reset_index(drop=True)

plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test.values, label="Actual",           color="black")
plt.plot(test_dates, y_pred,        label="XGBoost Forecast", color="blue")
plt.legend()
plt.title("XGBoost: Actual vs Predicted Revenue")
plt.xlabel("Week")
plt.ylabel("Revenue")
plt.savefig("outputs/xgboost_forecast.png", bbox_inches="tight")
plt.show()

#______SHAP explainability__________
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

#______summary plot — overall feature importance__________
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("outputs/shap_summary.png", bbox_inches="tight")
plt.show()

#______waterfall plot — breakdown for one specific week__________
# shap_values from TreeExplainer is a plain numpy array
# shap.waterfall_plot needs a shap.Explanation object — we build it manually

# pick which test week to explain — week index 0 = first test week
# change this number to explain any other week (0 to len(X_test)-1)
WEEK_TO_EXPLAIN = 0

# build the Explanation object manually
# shap_values[WEEK_TO_EXPLAIN] = array of 24 SHAP values for that week
# explainer.expected_value = the baseline (average prediction across training)
# X_test.iloc[WEEK_TO_EXPLAIN] = the actual feature values for that week
explanation = shap.Explanation(
    values=shap_values[WEEK_TO_EXPLAIN],
    base_values=explainer.expected_value,
    data=X_test.iloc[WEEK_TO_EXPLAIN].values,
    feature_names=feature_cols
)

# get the actual date of this week for the title
week_date = df["week_start"][train_size:].reset_index(drop=True)[WEEK_TO_EXPLAIN]
actual_rev = y_test[WEEK_TO_EXPLAIN]
predicted_rev = y_pred[WEEK_TO_EXPLAIN]

plt.figure()
shap.plots.waterfall(explanation, show=False)
plt.title(
    f"SHAP Waterfall — Week of {week_date.date()}\n"
    f"Actual: {actual_rev:,.0f}  |  Predicted: {predicted_rev:,.0f}",
    fontsize=10
)
plt.tight_layout()
plt.savefig("outputs/shap_waterfall.png", bbox_inches="tight")
plt.show()

print(f"\nWaterfall explained for: week of {week_date.date()}")
print(f"Baseline (avg prediction): {explainer.expected_value:,.0f}")
print(f"Final prediction:          {predicted_rev:,.0f}")
print(f"Actual revenue:            {actual_rev:,.0f}")
print(f"Difference (pred-actual):  {predicted_rev - actual_rev:,.0f}")

#______future forecasting — recursive walk-forward__________

#______load the full featured dataset__________
full_df = pd.read_csv("data/featured_weekly_revenue.csv")
full_df["week_start"] = pd.to_datetime(full_df["week_start"])

#______keep last 52 weeks of real data as the seed window__________
seed_window = full_df[["week_start", "revenue"]].tail(52).copy().reset_index(drop=True)

#______how many weeks ahead to forecast__________
WEEKS_AHEAD = 26  # 26 weeks = 6 months

#______historical volatility — used to scale noise__________
historical_std = full_df["revenue"].std()

#______storage for both pure and noisy predictions__________
pure_predictions  = []
noisy_predictions = []
future_dates      = []

np.random.seed(42)

#______recursive loop__________
for week_num in range(1, WEEKS_AHEAD + 1):

    rev = seed_window["revenue"].values

    #______lag features__________
    lag_1  = rev[-1]
    lag_2  = rev[-2]
    lag_4  = rev[-4]
    lag_8  = rev[-8]
    lag_12 = rev[-12]
    lag_52 = rev[-52]

    #______rolling statistics__________
    rolling_mean_4  = rev[-4:].mean()
    rolling_mean_12 = rev[-12:].mean()
    rolling_mean_26 = rev[-26:].mean()
    rolling_std_4   = rev[-4:].std()
    rolling_std_12  = rev[-12:].std()
    rolling_max_4   = rev[-4:].max()

    #______next week's date__________
    next_date = seed_window["week_start"].iloc[-1] + pd.Timedelta(weeks=1)

    #______calendar features — always known in advance__________
    week_of_year   = int(next_date.isocalendar().week)
    month          = next_date.month
    quarter        = (month - 1) // 3 + 1
    is_month_end   = int(next_date.is_month_end)
    is_december    = int(month == 12)
    is_january     = int(month == 1)
    weeks_elapsed  = len(full_df) + week_num

    #______quarter end detection__________
    prev_month     = seed_window["week_start"].iloc[-1].month
    prev_quarter   = (prev_month - 1) // 3 + 1
    is_quarter_end = int(quarter != prev_quarter)

    #______trend features__________
    wow_change            = (lag_1 - lag_2)  / lag_2  if lag_2  != 0 else 0
    revenue_vs_4w_avg     = lag_1 / rolling_mean_4    if rolling_mean_4 != 0 else 1
    revenue_pct_change_4  = (lag_1 - lag_4)  / lag_4  if lag_4  != 0 else 0
    revenue_pct_change_52 = (lag_1 - lag_52) / lag_52 if lag_52 != 0 else 0

    #______assemble feature row__________
    future_row = pd.DataFrame([{
        "lag_1":                  lag_1,
        "lag_2":                  lag_2,
        "lag_4":                  lag_4,
        "lag_8":                  lag_8,
        "lag_12":                 lag_12,
        "lag_52":                 lag_52,
        "rolling_mean_4":         rolling_mean_4,
        "rolling_mean_12":        rolling_mean_12,
        "rolling_mean_26":        rolling_mean_26,
        "rolling_std_4":          rolling_std_4,
        "rolling_std_12":         rolling_std_12,
        "rolling_max_4":          rolling_max_4,
        "week_of_year":           week_of_year,
        "month":                  month,
        "quarter":                quarter,
        "is_month_end":           is_month_end,
        "is_quarter_end":         is_quarter_end,
        "is_december":            is_december,
        "is_january":             is_january,
        "weeks_elapsed":          weeks_elapsed,
        "wow_change":             wow_change,
        "revenue_vs_4w_avg":      revenue_vs_4w_avg,
        "revenue_pct_change_4":   revenue_pct_change_4,
        "revenue_pct_change_52":  revenue_pct_change_52
    }])

    #______pure prediction — what the model thinks without noise__________
    predicted_revenue = model.predict(future_row)[0]
    predicted_revenue = max(predicted_revenue, 0)

    #______noisy prediction — adds realistic week-to-week variation__________
    noise = np.random.normal(loc=0, scale=historical_std * 0.15)
    predicted_revenue_noisy = max(predicted_revenue + noise, 0)

    #______store both versions__________
    pure_predictions.append(predicted_revenue)
    noisy_predictions.append(predicted_revenue_noisy)
    future_dates.append(next_date)

    #______feed noisy version back for next iteration__________
    new_row     = pd.DataFrame([{"week_start": next_date, "revenue": predicted_revenue_noisy}])
    seed_window = pd.concat([seed_window, new_row], ignore_index=True)

#______build results dataframe__________
future_df = pd.DataFrame({
    "week_start":       future_dates,
    "pure_forecast":    pure_predictions,
    "noisy_forecast":   noisy_predictions
})

print("\nWeekly future predictions:")
print(future_df.to_string(index=False))

#______aggregate to monthly using noisy forecast__________
future_df["month"]       = future_df["week_start"].dt.to_period("M")
monthly_forecast         = future_df.groupby("month")["noisy_forecast"].sum().reset_index()
monthly_forecast.columns = ["month", "predicted_monthly_revenue"]
monthly_forecast["predicted_monthly_revenue"] = monthly_forecast["predicted_monthly_revenue"].round(2)

print("\nMonthly forecast (next 6 months):")
print(monthly_forecast.to_string(index=False))

#______plot 1 — weekly forecast showing both pure trend and noisy line__________
plt.figure(figsize=(14, 5))

real_tail = full_df[["week_start", "revenue"]].tail(52)
plt.plot(real_tail["week_start"], real_tail["revenue"],
         color="black", label="Real (last 52 weeks)")

plt.plot(future_df["week_start"], future_df["noisy_forecast"],
         color="cornflowerblue", linestyle="-",
         alpha=0.7, label="Simulated weekly forecast")

plt.plot(future_df["week_start"], future_df["pure_forecast"],
         color="blue", linestyle="--",
         linewidth=2, label="Trend forecast")

split_date = full_df["week_start"].iloc[-1]
plt.axvline(x=split_date, color="red", linestyle=":",
            linewidth=1.5, label="Forecast start")

plt.title("XGBoost — 6 Month Weekly Revenue Forecast")
plt.xlabel("Week")
plt.ylabel("Revenue")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/future_weekly_forecast.png", bbox_inches="tight")
plt.show()

#______plot 2 — monthly bar chart__________
plt.figure(figsize=(10, 5))
plt.bar(monthly_forecast["month"].astype(str),
        monthly_forecast["predicted_monthly_revenue"],
        color="steelblue", edgecolor="white")

plt.title("XGBoost — Monthly Revenue Forecast (Next 6 Months)")
plt.xlabel("Month")
plt.ylabel("Predicted Revenue")
plt.tight_layout()
plt.savefig("outputs/future_monthly_forecast.png", bbox_inches="tight")
plt.show()