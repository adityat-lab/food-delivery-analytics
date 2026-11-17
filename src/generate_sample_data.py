"""
OPTIONAL: creates a SYNTHETIC dataset in the same raw format as the Kaggle
"Food Delivery Dataset", so you can test the whole pipeline without downloading.
For your final portfolio, use the real Kaggle data.
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
N = 6000


def haversine(lat1, lon1, lat2, lon2):
    r = 6371
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


centers = np.array([[12.97, 77.59], [19.07, 72.87], [28.61, 77.20], [17.38, 78.48], [13.08, 80.27]])
idx = rng.integers(0, len(centers), N)
rest_lat = centers[idx, 0] + rng.normal(0, 0.03, N)
rest_lon = centers[idx, 1] + rng.normal(0, 0.03, N)
del_lat = rest_lat + rng.uniform(0.01, 0.12, N) * rng.choice([-1, 1], N)
del_lon = rest_lon + rng.uniform(0.01, 0.12, N) * rng.choice([-1, 1], N)
dist = haversine(rest_lat, rest_lon, del_lat, del_lon)

traffic = rng.choice(["Low ", "Medium ", "High ", "Jam "], N, p=[0.30, 0.35, 0.25, 0.10])
weather = rng.choice(["Sunny", "Cloudy", "Fog", "Windy", "Stormy", "Sandstorms"], N,
                     p=[0.30, 0.25, 0.15, 0.12, 0.10, 0.08])
festival = rng.choice(["No ", "Yes "], N, p=[0.97, 0.03])
multi = rng.choice([0, 1, 2, 3], N, p=[0.35, 0.45, 0.15, 0.05])
rating = np.clip(rng.normal(4.6, 0.35, N), 1, 5).round(1)
age = rng.integers(20, 40, N).astype(float)

t_add = pd.Series(traffic).str.strip().map({"Low": 0, "Medium": 3, "High": 6, "Jam": 10}).values
w_add = pd.Series(weather).map({"Sunny": 0, "Cloudy": 2, "Windy": 1, "Fog": 5, "Stormy": 4, "Sandstorms": 4}).values
minutes = (8 + 2.2 * dist + t_add + w_add + 3.5 * multi + (festival == "Yes ") * 7
           + (5 - rating) * 3 + rng.normal(0, 2.5, N))
minutes = np.clip(minutes, 10, 54).round().astype(int)

# order hour with lunch (12-14) and dinner (19-22) peaks
w = np.ones(24) * 0.2
w[[11, 12, 13, 14]] = [2, 4, 4, 2]
w[[19, 20, 21, 22]] = [3, 5, 5, 3]
w[[8, 9, 10, 15, 16, 17, 18]] = 0.8
w /= w.sum()
hours = rng.choice(24, N, p=w)
mins = rng.integers(0, 60, N)
ordered = pd.to_datetime("2022-01-01") + pd.to_timedelta(hours * 60 + mins, unit="m")
prep = rng.integers(5, 16, N)
picked = ordered + pd.to_timedelta(prep, unit="m")
dates = pd.to_datetime("2022-02-11") + pd.to_timedelta(rng.integers(0, 55, N), unit="D")

df = pd.DataFrame({
    "ID": [f"0x{i:04x}" for i in range(N)],
    "Delivery_person_ID": [f"RIDER{rng.integers(1, 400):04d}DEL01" for _ in range(N)],
    "Delivery_person_Age": age,
    "Delivery_person_Ratings": rating,
    "Restaurant_latitude": rest_lat,
    "Restaurant_longitude": rest_lon,
    "Delivery_location_latitude": del_lat,
    "Delivery_location_longitude": del_lon,
    "Order_Date": dates.strftime("%d-%m-%Y"),
    "Time_Orderd": ordered.strftime("%H:%M:%S"),
    "Time_Order_picked": picked.strftime("%H:%M:%S"),
    "Weatherconditions": "conditions " + pd.Series(weather),
    "Road_traffic_density": traffic,
    "Vehicle_condition": rng.integers(0, 4, N),
    "Type_of_order": rng.choice(["Snack ", "Meal ", "Drinks ", "Buffet "], N),
    "Type_of_vehicle": rng.choice(["motorcycle ", "scooter ", "electric_scooter ", "bicycle "], N,
                                  p=[0.55, 0.3, 0.1, 0.05]),
    "multiple_deliveries": multi.astype(float),
    "Festival": festival,
    "City": rng.choice(["Metropolitian ", "Urban ", "Semi-Urban "], N, p=[0.6, 0.37, 0.03]),
    "Time_taken(min)": ["(min) " + str(m) for m in minutes],
})
# inject some missing values like the real data
df.loc[rng.random(N) < 0.03, "Delivery_person_Age"] = np.nan
df.loc[rng.random(N) < 0.03, "Delivery_person_Ratings"] = np.nan
df.loc[rng.random(N) < 0.01, "Road_traffic_density"] = "NaN "

out = Path("data/raw/sample_orders.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Synthetic data saved to {out} ({len(df)} rows)")