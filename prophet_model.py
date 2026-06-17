#______importing libraries__________
import pandas as pd
import numpy as np
from prophet import Prophet  # Prophet's forecasting model class
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

#______loading weekly revenue data__________
# Prophet does NOT use engineered features (lags, rolling stats, calendar flags)
# it only needs the raw date and revenue columns, so we load weekly_revenue.csv, not the featured one
df = pd.read_csv("data/weekly_revenue.csv")
df["week_start"] = pd.to_datetime(df["week_start"])

#______preparing data in Prophet's required format__________
# Prophet requires exactly two columns named "ds" (date) and "y" (the value to predict)
# this is a strict naming rule, the library will not work with any other column names
prophet_df = df.rename(columns={"week_start": "ds", "revenue": "y"})

#______time-aware train/test split__________
train_size = int(len(prophet_df) * 0.8)  # 80% of weeks for training
train = prophet_df[:train_size]           # first 80% chronologically
test = prophet_df[train_size:]            # last 20% chronologically, never shuffled

#______training Prophet model__________
model = Prophet(
    yearly_seasonality=True,   # let Prophet model annual seasonal patterns
    weekly_seasonality=False,  # turn off, since our data is already weekly aggregated, no daily pattern exists within a week
    changepoint_prior_scale=0.05,  # default flexibility for trend changepoints, higher = more flexible trend
    seasonality_mode="additive"
)
model.fit(train)  # train Prophet on the training portion only

#______generating forecast for test period__________
future = model.make_future_dataframe(periods=len(test), freq="W-MON")  # creates future dates to predict, matching test set length and weekly frequency
forecast = model.predict(future)  # generates yhat, yhat_lower, yhat_upper for every date including training dates

#______extracting only the test period predictions__________
forecast_test = forecast.tail(len(test))  # the last N rows of forecast correspond to our test period

#______calculating evaluation metrics__________
y_true = test["y"].values
y_pred = forecast_test["yhat"].values

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

#______plotting actual vs predicted with confidence interval__________
plt.figure(figsize=(12, 6))
plt.plot(test["ds"], y_true, label="Actual", color="black")
plt.plot(test["ds"], y_pred, label="Prophet Forecast", color="green")
plt.fill_between(test["ds"], forecast_test["yhat_lower"], forecast_test["yhat_upper"],
                  color="green", alpha=0.2, label="Confidence Interval")  # shaded band showing uncertainty range
plt.legend()
plt.title("Prophet: Actual vs Predicted Revenue")
plt.xlabel("Week")
plt.ylabel("Revenue")
plt.savefig("outputs/prophet_forecast.png", bbox_inches="tight")
plt.show()

#______plotting Prophet's own decomposition__________
fig = model.plot_components(forecast)  # built-in Prophet plot showing trend and yearly seasonality separately
plt.savefig("outputs/prophet_components.png", bbox_inches="tight")
plt.show()