#______importing libraries__________
import pandas as pd
import numpy as np 
import xgboost as xgb
from sklearn.metrics import mean_absolute_error,mean_squared_error
import  matplotlib.pyplot as plt
import shap 

#______loading featured weekly revenue data__________
df=pd.read_csv("data/featured_weekly_revenue.csv")
df["week_start"]=pd.to_datetime(df["week_start"])

#______defining feature columns and target________
feature_cols = ["lag_1", "lag_2", "lag_4", "lag_8", "lag_12", "lag_52",
                 "rolling_mean_4", "rolling_mean_12", "rolling_mean_26",
                 "rolling_std_4", "rolling_std_12", "rolling_max_4",
                 "week_of_year", "month", "quarter", "is_month_end",
                 "is_quarter_end", "is_december", "is_january",
                 "weeks_elapsed", "wow_change", "revenue_vs_4w_avg",
                 "revenue_pct_change_4", "revenue_pct_change_52"]

X = df[feature_cols]
y = df["revenue"]

#______time-awware train/test split__________
train_size=int(len(df)*0.8)

X_train=X[:train_size]
X_test=X[train_size:]
y_test=y[train_size:]
y_train=y[:train_size]
y_test=y_test.reset_index(drop=True)

#______training xgboost model__________
model=xgb.XGBRegressor(
  n_estimators=200,#the number of trees the model builds. Each tree corrects the errors of the previous ones. 200 is a reasonable starting point, not too few to underfit, not so many it overfits or trains slowly.
  max_depth=4,
  learning_rate=0.05,
  subsample=0.8,#each tree is trained on a random 80% of the training rows instead of all of them. This adds randomness that helps prevent overfitting
  colsample_bytree=0.8,#each tree only considers 80% of your 24 features when deciding splits, rather than all features every time. Also reduces overfitting and makes the model more robust.
  random_state=42
)

model.fit(X_train,y_train)

#______generating predictions on test set__________
y_pred=model.predict(X_test)#runs the trained model on your test features and generates predicted revenue values for each of the 31 test weeks.

#______calculating evaluation metrics__________
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
mape=np.mean(np.abs((y_test.values-y_pred)/y_test.values))*100 #calculates the average percentage error. .values converts y_test from a pandas series to a plain numpy array so it aligns correctly with y_pred for the subtraction.

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

#_______-debugging black line issue problem__________
#print(y_test[:5])
#print(test_date.values[:5])

#______plotting actual vs predicted________
test_dates = df["week_start"][train_size:].reset_index(drop=True)#reset the index on test_dates before plotting

plt.figure(figsize=(12, 6))#plots the true revenue values as a black line
plt.plot(test_dates, y_test.values, label="Actual", color="black")
plt.plot(test_dates, y_pred, label="XGBoost Forecast", color="blue")#plots your XGBoost predictions as a blue line, this time using a different colour from the ARIMA red so you can keep them visually distinct when comparing later.
plt.legend()
plt.title("XGBoost: Actual vs Predicted Revenue")
plt.xlabel("Week")
plt.ylabel("Revenue")
plt.savefig("outputs/xgboost_forecast.png")
plt.show()

#______SHAP explainablitly__________
explainer=shap.TreeExplainer(model)# creates a SHAP explainer specifically built for tree-based models like XGBoost. It knows how to efficiently calculate exactly how each tree split contributed to every prediction.
shap_values=explainer.shap_values(X_test)#calculates the SHAP value for every feature, for every row in your test set. This is a big matrix — 31 test weeks by 24 features, each cell telling you how much that feature pushed that specific week's prediction up or down.

#______summary plot-overall feature importance__________
shap.summary_plot(shap_values,X_test,show=False)# generates the most important SHAP visual, a chart that ranks all 24 features by their overall impact across every prediction. Features at the top matter most. show=False stops it from immediately popping up so we can save it first
plt.savefig("outputs/shap_summary.png", bbox_inches="tight")# trims excess whitespace around SHAP plots which tend to have wide margins by default.
plt.show()

#______waterfall plot for one specific prediction__________
sample_index=5
shap.plots.waterfall(shap.Explanation(
  values=shap_values[sample_index],
  base_values=explainer.expected_value,
  data=X_test.iloc[sample_index],
  feature_names=feature_cols
))
plt.savefig("outputs/shap_waterfall.png", bbox_inches="tight")
plt.show()


#_______results from waterfall model_________________
# E[f(X)] = 5,685,144.5, is the model's baseline expected prediction averaged across all training data. Each bar then shows how much each feature pushed the prediction away from that baseline, ending at the final predicted value of roughly ₹4,697,268 at the top.
#revenue_vs_4w_avg pulled the prediction down by ₹4,41,048 — by far the dominant factor. This means that week's revenue was running noticeably below its recent 4-week average, and the model leaned heavily on that signal to predict a lower number.
#rolling_mean_4 pushed it back up by ₹1,05,046, partially offsetting that — the recent 4-week average itself was still reasonably healthy.
#revenue_pct_change_52, wow_change, and lag_4 all pulled slightly down, reinforcing the "this week is underperforming its recent pattern" story.
#lag_2, lag_1, and lag_52 pulled slightly up, adding small counterbalancing signals from recent and year-ago revenue.