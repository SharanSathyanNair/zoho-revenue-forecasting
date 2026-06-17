#______importing libraries__________
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
 #______loading feature weekly revenue data__________
df=pd.read_csv("data/featured_weekly_revenue.csv")
df["week_start"]=pd.to_datetime(df["week_start"])

#______settting week start as index for time series__________
df=df.set_index("week_start")

#______time-aware train/test split___________
train_size=int(len(df)*0.8)
train=df["revenue"][:train_size]
test=df["revenue"][train_size:]

#______fitting ARIMA model______
model=ARIMA(train, order=(0,1,1))
model_fit=model.fit()

print(model_fit.summary())

#_______generating forecast for test period___________
forecast=model_fit.forecast(steps=len(test))

#______calculating evaluation metrics__________
mae=np.mean(np.abs(test.values-forecast.values))
rmse=np.sqrt(np.mean((test.values-forecast.values)**2))
mape=np.mean(np.abs((test.values-forecast.values)/test.values))*100

print(f"MAE: {mae:2f}")
print(f"RMSE: {rmse:2f}")
print(f"MAPE: {mape:2f}")

#______plotting actual vs predicted________
plt.figure(figsize=(12, 6))
plt.plot(test.index, test.values, label="Actual", color="black")
plt.plot(test.index, forecast.values, label="ARIMA Forecast", color="red")
plt.legend()
plt.title("ARIMA: Actual vs Predicted Revenue")
plt.xlabel("Week")
plt.ylabel("Revenue")
plt.savefig("outputs/arima_forecast.png")
plt.show()