import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit,
)
from xgboost import XGBRegressor

# ==========================================================
# Configuration
# ==========================================================

INPUT_FILE: str = "data/model_ready_data.csv"
OUTPUT_DIR: str = "models"
OUTPUT_FILE: str = "best_xgb_params.json"
N_SPLITS: int = 5
N_ITER: int = 100
RANDOM_STATE: int = 42

# ==========================================================
# Create Directory
# ==========================================================

def create_directory() -> None:
    """Create the output directory if it does not exist."""

    Path(OUTPUT_DIR).mkdir(exist_ok=True)


# ==========================================================
# Load Data
# ==========================================================

def load_data() -> pd.DataFrame:
    """Load the model-ready dataset."""

    path = Path(INPUT_FILE)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    try:

        df = pd.read_csv(
            path,
            parse_dates=["week"],
        )

    except Exception as e:
        raise RuntimeError(
            "Unable to load model-ready dataset."
        ) from e

    return (
        df.sort_values("week")
          .reset_index(drop=True)
    )


# ==========================================================
# Prepare Features
# ==========================================================

def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataset into features and target."""

    X = df.drop(
        columns=[
            "week",
            "weekly_revenue",
        ]
    )

    y = df["weekly_revenue"]
    return X, y

# ==========================================================
# Time Series Cross Validation
# ==========================================================

def build_cv() -> TimeSeriesSplit:
    """Create the walk-forward cross-validation object."""

    return TimeSeriesSplit(
        n_splits=N_SPLITS
    )


# ==========================================================
# Parameter Space
# ==========================================================

def build_parameter_space() -> dict:
    """Define the XGBoost hyperparameter search space."""

    return {
        "n_estimators": randint(250, 900),
        "max_depth": randint(2, 8),
        "learning_rate": uniform(0.01, 0.19),
        "subsample": uniform(0.70, 0.30),
        "colsample_bytree": uniform(0.70, 0.30),
        "gamma": uniform(0, 1),
        "min_child_weight": randint(1, 8),
        "reg_alpha": uniform(0, 2),
        "reg_lambda": uniform(1, 4),
    }


# ==========================================================
# Build Model
# ==========================================================

def build_model() -> XGBRegressor:
    """Build the XGBoost regressor."""

    return XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

# ==========================================================
# Hyperparameter Search
# ==========================================================

def tune_model(
    model: XGBRegressor,
    X: pd.DataFrame,
    y: pd.Series,
) -> RandomizedSearchCV:
    """Perform randomized hyperparameter tuning."""

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=build_parameter_space(),
        n_iter=N_ITER,
        scoring="neg_mean_absolute_percentage_error",
        cv=build_cv(),
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search


# ==========================================================
# Save Parameters
# ==========================================================

def save_parameters(
    search: RandomizedSearchCV,
) -> Path:
    """Save the best hyperparameters."""
    params = {}
    for key, value in search.best_params_.items():
        if isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            value = float(value)
        params[key] = value
    output_path = Path(OUTPUT_DIR) / OUTPUT_FILE

    try:
        with open(output_path, "w") as f:
            json.dump(
                params,
                f,
                indent=4,
            )
    except Exception as e:
        raise RuntimeError(
            "Unable to save tuned parameters."
        ) from e
    return output_path


# ==========================================================
# Summary
# ==========================================================

def print_summary(search, output_path):

    print("\n==============================")
    print("XGBoost Hyperparameter Search")
    print("==============================")
    print(
        f"Best CV MAPE : {-search.best_score_ * 100:.2f}%"
    )
    clean_params = {}

    for key, value in search.best_params_.items():
        if isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            value = round(float(value), 6)

        clean_params[key] = value

    print(f"Best Params  : {clean_params}")
    print(
        f"Saved To     : {output_path}"
    )
    print("==============================")


# ==========================================================
# Main
# ==========================================================

def main() -> None:
    """Execute the hyperparameter tuning pipeline."""
    print(
        "\nStarting XGBoost Hyperparameter Tuning...\n"
    )
    try:
        create_directory()
        df = load_data()
        X, y = prepare_features(df)
        model = build_model()
        search = tune_model(
            model,
            X,
            y,
        )
        output_path = save_parameters(
            search,
        )
        print_summary(
            search,
            output_path,
        )
    except Exception as e:

        print("\n==============================")
        print("Hyperparameter Tuning Failed")
        print("==============================")
        print(e)
        print("==============================")
        raise

if __name__ == "__main__":
    main()