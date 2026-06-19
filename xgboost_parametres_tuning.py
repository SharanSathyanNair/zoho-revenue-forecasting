#______importing libraries__________
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, mean_absolute_percentage_error

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

#______STEP 1: grid search for structural parameters__________
param_grid = {
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [3, 4, 5],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9]
}

tscv = TimeSeriesSplit(n_splits=5)
mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

base_model = xgb.XGBRegressor(
    n_estimators=1000,
    random_state=42,
    eval_metric="mae"
)

print("="*60)
print("STEP 1: Searching for best structural parameters...")
print("="*60)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=tscv,
    scoring=mape_scorer,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X, y)

best_params = grid_search.best_params_

print("\nBest structural parameters found:")
print(best_params)
print(f"Best cross-validated MAPE: {-grid_search.best_score_*100:.2f}%")

#______showing top 5 combinations for transparency__________
results_df = pd.DataFrame(grid_search.cv_results_)
results_df["mean_test_mape"] = -results_df["mean_test_score"] * 100
top5 = results_df.sort_values("mean_test_mape").head(5)
print("\nTop 5 parameter combinations (for reference):")
print(top5[["params", "mean_test_mape"]].to_string(index=False))

#______STEP 2: early stopping to find optimal n_estimators__________
print("\n" + "="*60)
print("STEP 2: Finding optimal n_estimators with early stopping...")
print("="*60)

split_point = int(len(X) * 0.85)
X_train_es = X[:split_point]
X_val_es = X[split_point:]
y_train_es = y[:split_point]
y_val_es = y[split_point:]

final_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=best_params["learning_rate"],
    max_depth=best_params["max_depth"],
    subsample=best_params["subsample"],
    colsample_bytree=best_params["colsample_bytree"],
    random_state=42,
    eval_metric="mae",
    early_stopping_rounds=20
)

final_model.fit(
    X_train_es, y_train_es,
    eval_set=[(X_val_es, y_val_es)],
    verbose=False
)

optimal_n_estimators = final_model.best_iteration

print(f"\nOptimal number of trees: {optimal_n_estimators}")
print(f"Validation MAE at that point: {final_model.best_score:.2f}")

#______FINAL RECOMMENDED CONFIGURATION__________
print("\n" + "="*60)
print("FINAL RECOMMENDED XGBOOST CONFIGURATION")
print("="*60)
print(f"""
model = xgb.XGBRegressor(
    n_estimators={optimal_n_estimators},
    learning_rate={best_params['learning_rate']},
    max_depth={best_params['max_depth']},
    subsample={best_params['subsample']},
    colsample_bytree={best_params['colsample_bytree']},
    random_state=42
)
""")
print("Copy this block into xgboost_model.py whenever you rerun this tuning script.")
print("="*60)