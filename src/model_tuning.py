"""Step 6: hyperparameter tuning with XGBoost and LightGBM, compared against the baselines."""
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

NUM = ["rider_age", "rider_rating", "distance_km", "order_hour", "vehicle_condition",
       "multiple_deliveries", "prep_time_min"]
CAT = ["weather", "traffic", "order_type", "vehicle_type", "city", "festival"]
TARGET = "delivery_time_min"
N_ITER = 15   # number of random parameter combinations to try per model
CV = 3        # cross-validation folds

df = pd.read_csv("data/processed/clean.csv")
X, y = df[NUM + CAT], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def make_pipe(model, scale=False):
    num = StandardScaler() if scale else "passthrough"
    pre = ColumnTransformer([("num", num, NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return Pipeline([("prep", pre), ("model", model)])


# model, search space (None = no tuning)
experiments = {
    "Linear Regression": (make_pipe(LinearRegression(), scale=True), None),
    "Random Forest": (
        make_pipe(RandomForestRegressor(random_state=42, n_jobs=-1)),
        {"model__n_estimators": [100, 200, 300],
         "model__max_depth": [None, 10, 20],
         "model__min_samples_leaf": [1, 2, 5]},
    ),
    "XGBoost": (
        make_pipe(XGBRegressor(tree_method="hist", random_state=42, n_jobs=-1)),
        {"model__n_estimators": [200, 400, 600],
         "model__max_depth": [3, 5, 7],
         "model__learning_rate": [0.03, 0.05, 0.1],
         "model__subsample": [0.7, 0.85, 1.0],
         "model__colsample_bytree": [0.7, 0.85, 1.0]},
    ),
    "LightGBM": (
        make_pipe(LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)),
        {"model__n_estimators": [200, 400, 600],
         "model__num_leaves": [15, 31, 63],
         "model__learning_rate": [0.03, 0.05, 0.1],
         "model__subsample": [0.7, 0.85, 1.0],
         "model__colsample_bytree": [0.7, 0.85, 1.0]},
    ),
}

results, best_models = {}, {}
for name, (pipe, grid) in experiments.items():
    start = time.time()
    if grid:
        search = RandomizedSearchCV(pipe, grid, n_iter=N_ITER, cv=CV,
                                    scoring="neg_mean_absolute_error",
                                    random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        model, params = search.best_estimator_, search.best_params_
    else:
        pipe.fit(X_train, y_train)
        model, params = pipe, {}

    pred = model.predict(X_test)
    results[name] = {
        "MAE": round(mean_absolute_error(y_test, pred), 3),
        "RMSE": round(root_mean_squared_error(y_test, pred), 3),
        "R2": round(r2_score(y_test, pred), 4),
        "seconds": round(time.time() - start, 1),
        "best_params": {k.replace("model__", ""): v for k, v in params.items()},
    }
    best_models[name] = model
    print(f"{name:18s} MAE={results[name]['MAE']}  RMSE={results[name]['RMSE']}  R2={results[name]['R2']}")

# save results, best model and comparison chart
Path("reports").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)
Path("images").mkdir(exist_ok=True)
Path("reports/tuning_results.json").write_text(json.dumps(results, indent=2, default=str))

best_name = min(results, key=lambda k: results[k]["MAE"])
joblib.dump(best_models[best_name], "models/best_model.joblib")
print(f"\nBest model: {best_name} (MAE {results[best_name]['MAE']} min). Saved to models/best_model.joblib")

mae = pd.Series({k: v["MAE"] for k, v in results.items()}).sort_values()
plt.figure(figsize=(7, 4))
mae.plot.barh(color="#E8590C")
plt.xlabel("MAE (minutes, lower is better)")
plt.title("Model Comparison after Tuning")
plt.tight_layout()
plt.savefig("images/11_model_comparison.png", dpi=130)