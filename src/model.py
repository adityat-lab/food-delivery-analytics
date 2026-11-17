"""Step 4: predict delivery time (Linear Regression vs Random Forest)."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM = ["rider_age", "rider_rating", "distance_km", "order_hour", "vehicle_condition",
       "multiple_deliveries", "prep_time_min"]
CAT = ["weather", "traffic", "order_type", "vehicle_type", "city", "festival"]
TARGET = "delivery_time_min"

df = pd.read_csv("data/processed/clean.csv")
X, y = df[NUM + CAT], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def make_pipe(model, scale):
    num = StandardScaler() if scale else "passthrough"
    pre = ColumnTransformer([("num", num, NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return Pipeline([("prep", pre), ("model", model)])


models = {
    "Linear Regression": make_pipe(LinearRegression(), True),
    "Random Forest": make_pipe(RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42), False),
}

metrics = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics[name] = {"MAE": round(mean_absolute_error(y_test, pred), 2),
                     "R2": round(r2_score(y_test, pred), 3)}
    print(name, metrics[name])

Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)
Path("images").mkdir(exist_ok=True)
joblib.dump(models["Random Forest"], "models/rf_model.joblib")
Path("reports/model_metrics.json").write_text(json.dumps(metrics, indent=2))

# feature importance
rf = models["Random Forest"]
names = rf.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names).sort_values().tail(12)
plt.figure(figsize=(7, 5)); imp.plot.barh(color="#E8590C")
plt.title("Top Feature Importances (Random Forest)")
plt.tight_layout(); plt.savefig("images/10_feature_importance.png", dpi=130)
print("Saved model, metrics and feature importance chart.")