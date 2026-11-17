"""Step 1: clean the raw Kaggle food delivery data and engineer features."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
OUT = Path("data/processed/clean.csv")
CANDIDATES = ["train.csv", "food_delivery.csv", "sample_orders.csv"]

RENAME = {
    "id": "order_id",
    "deliverypersonid": "rider_id",
    "deliverypersonage": "rider_age",
    "deliverypersonratings": "rider_rating",
    "restaurantlatitude": "rest_lat",
    "restaurantlongitude": "rest_lon",
    "deliverylocationlatitude": "del_lat",
    "deliverylocationlongitude": "del_lon",
    "orderdate": "order_date",
    "timeorderd": "order_time",
    "timeordered": "order_time",
    "timeorderpicked": "pickup_time",
    "weatherconditions": "weather",
    "roadtrafficdensity": "traffic",
    "vehiclecondition": "vehicle_condition",
    "typeoforder": "order_type",
    "typeofvehicle": "vehicle_type",
    "multipledeliveries": "multiple_deliveries",
    "festival": "festival",
    "city": "city",
    "timetakenmin": "delivery_time_min",
}


def norm(col: str) -> str:
    return re.sub(r"[^a-z0-9]", "", col.lower())


def haversine(lat1, lon1, lat2, lon2):
    r = 6371
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def find_raw_file() -> Path:
    for name in CANDIDATES:
        if (RAW_DIR / name).exists():
            return RAW_DIR / name
    raise SystemExit(
        "No raw file found in data/raw/. Download 'train.csv' from Kaggle "
        "(see README) or run: python src/generate_sample_data.py"
    )


def main():
    path = find_raw_file()
    print(f"Reading {path}")
    df = pd.read_csv(path)
    df.columns = [RENAME.get(norm(c), norm(c)) for c in df.columns]
    n0 = len(df)

    # 1. strip whitespace, turn "NaN" strings into real missing values
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].astype("string").str.strip()
            s = s.where(~s.str.lower().isin(["nan", ""]))
            df[c] = s.astype(object).where(s.notna(), np.nan)

    df = df.drop_duplicates()

    # 2. fix messy text columns
    df["delivery_time_min"] = (
        df["delivery_time_min"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    )
    df["weather"] = df["weather"].astype(str).str.replace(r"^conditions\s*", "", regex=True)
    df["weather"] = df["weather"].replace("nan", np.nan)
    for c in ["traffic", "order_type", "vehicle_type", "city", "festival"]:
        df[c] = df[c].astype(str).str.strip().str.title().replace("Nan", np.nan)

    # 3. numeric columns
    for c in ["rider_age", "rider_rating", "vehicle_condition", "multiple_deliveries",
              "rest_lat", "rest_lon", "del_lat", "del_lon"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 4. dates and times
    df["order_date"] = pd.to_datetime(df["order_date"], dayfirst=True, errors="coerce")
    ordered = pd.to_datetime(df["order_time"], format="%H:%M:%S", errors="coerce")
    picked = pd.to_datetime(df["pickup_time"], format="%H:%M:%S", errors="coerce")
    df["order_hour"] = ordered.dt.hour
    prep = (picked - ordered).dt.total_seconds() / 60
    df["prep_time_min"] = prep.where(prep >= 0, prep + 1440)
    df["day_of_week"] = df["order_date"].dt.day_name()
    df["is_weekend"] = df["order_date"].dt.dayofweek.isin([5, 6])

    # 5. distance (km). Some rows have 0/negative coordinates -> use abs and drop zeros
    for c in ["rest_lat", "rest_lon", "del_lat", "del_lon"]:
        df[c] = df[c].abs()
    valid = (df["rest_lat"] > 1) & (df["rest_lon"] > 1)
    df = df[valid].copy()
    df["distance_km"] = haversine(df["rest_lat"], df["rest_lon"], df["del_lat"], df["del_lon"]).round(2)
    df = df[df["distance_km"] <= 50]

    # 6. remove impossible values, handle missing data
    df = df[df["rider_rating"].between(1, 5) | df["rider_rating"].isna()]
    df = df.dropna(subset=["delivery_time_min", "order_hour", "distance_km", "order_date"])
    df["rider_age"] = df["rider_age"].fillna(df["rider_age"].median())
    df["rider_rating"] = df["rider_rating"].fillna(df["rider_rating"].median())
    df["multiple_deliveries"] = df["multiple_deliveries"].fillna(0)
    df["prep_time_min"] = df["prep_time_min"].fillna(df["prep_time_min"].median())
    for c in ["weather", "traffic", "order_type", "vehicle_type", "city", "festival"]:
        df[c] = df[c].fillna("Unknown")

    df["order_hour"] = df["order_hour"].astype(int)
    keep = ["order_id", "rider_id", "rider_age", "rider_rating", "rest_lat", "rest_lon",
            "del_lat", "del_lon", "order_date", "order_hour", "day_of_week", "is_weekend",
            "weather", "traffic", "vehicle_condition", "order_type", "vehicle_type",
            "multiple_deliveries", "festival", "city", "prep_time_min", "distance_km",
            "delivery_time_min"]
    df = df[keep]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Rows: {n0} raw -> {len(df)} clean. Saved to {OUT}")
    print(df.describe().T[["count", "mean", "min", "max"]].to_string())


if __name__ == "__main__":
    main()