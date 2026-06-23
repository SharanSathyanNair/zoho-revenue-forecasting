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
                 "revenue_pct_change_4", "revenue_pct_change_52"]

X = df[feature_cols]
y = df["revenue"]

#______STEP 0: carve out the final test set FIRST — nothing below this line
#______touches X_test/y_test until the very last evaluation__________
test_size = int(len(df) * 0.20)
train_pool_size = len(df) - test_size

X_pool, X_test = X[:train_pool_size], X[train_pool_size:]
y_pool, y_test = y[:train_pool_size], y[train_pool_size:]
y_test = y_test.reset_index(drop=True)

print(f"Train+val pool: {len(X_pool)} rows  |  Held-out test: {len(X_test)} rows (never touched until the end)")

#______STEP 1: grid search for structural parameters, using ONLY the pool__________
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

print("\n" + "="*60)
print("STEP 1: Searching for best structural parameters (on pool only)...")
print("="*60)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=tscv,
    scoring=mape_scorer,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_pool, y_pool)

best_params = grid_search.best_params_

print("\nBest structural parameters found:")
print(best_params)
print(f"Best cross-validated MAPE (pool only): {-grid_search.best_score_*100:.2f}%")

#______STEP 2: early stopping, using a slice of the POOL — still never touches X_test__________
print("\n" + "="*60)
print("STEP 2: Finding optimal n_estimators with early stopping (within pool)...")
print("="*60)

es_split_point = int(len(X_pool) * 0.85)
X_train_es = X_pool[:es_split_point]
X_val_es = X_pool[es_split_point:]
y_train_es = y_pool[:es_split_point]
y_val_es = y_pool[es_split_point:]

print(f"Early-stopping train: {len(X_train_es)} rows | val: {len(X_val_es)} rows (both inside the pool, test untouched)")

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

#______FIX: best_iteration is 0-indexed, so the tree count is +1__________
optimal_n_estimators = final_model.best_iteration + 1

print(f"\nOptimal number of trees: {optimal_n_estimators} (best_iteration={final_model.best_iteration} + 1)")
print(f"Validation MAE at that point: {final_model.best_score:.2f}")

#______STEP 3: refit on the FULL pool with final params, then score ONCE on the
#______untouched test set — this is the only honest, unbiased number__________
print("\n" + "="*60)
print("STEP 3: Refitting on full pool, scoring once on held-out test...")
print("="*60)

production_model = xgb.XGBRegressor(
    n_estimators=optimal_n_estimators,
    learning_rate=best_params["learning_rate"],
    max_depth=best_params["max_depth"],
    subsample=best_params["subsample"],
    colsample_bytree=best_params["colsample_bytree"],
    random_state=42
)
production_model.fit(X_pool, y_pool)

y_pred = production_model.predict(X_test)
final_mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100
print(f"\nFinal, never-touched-before MAPE on the true test set: {final_mape:.2f}%")

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
print("="*60)