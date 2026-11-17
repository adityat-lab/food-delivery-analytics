"""Step 2: exploratory analysis. Saves charts to images/ and insights to reports/insights.md"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
IMG = Path("images")
IMG.mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

df = pd.read_csv("data/processed/clean.csv")
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TRAFFIC = ["Low", "Medium", "High", "Jam"]
insights = []


def save(name):
    plt.tight_layout()
    plt.savefig(IMG / name, dpi=130)
    plt.close()


# 1. delivery time distribution
plt.figure(figsize=(7, 4))
sns.histplot(df["delivery_time_min"], bins=30, kde=True, color="#E8590C")
plt.title("Delivery Time Distribution"); plt.xlabel("Minutes")
save("01_delivery_time_distribution.png")
insights.append(f"Average delivery time is **{df.delivery_time_min.mean():.1f} min** "
                f"(median {df.delivery_time_min.median():.0f}, 90th percentile {df.delivery_time_min.quantile(.9):.0f}).")

# 2. traffic impact
t = df[df.traffic.isin(TRAFFIC)].groupby("traffic")["delivery_time_min"].mean().reindex(TRAFFIC).dropna()
plt.figure(figsize=(6, 4)); sns.barplot(x=t.index, y=t.values, color="#4C6EF5")
plt.title("Avg Delivery Time by Traffic Density"); plt.ylabel("Minutes")
save("02_traffic_impact.png")
if {"Jam", "Low"} <= set(t.index):
    insights.append(f"Deliveries in **Jam** traffic take {t['Jam']:.1f} min vs {t['Low']:.1f} min in Low traffic "
                    f"(**+{(t['Jam'] / t['Low'] - 1) * 100:.0f}%**).")

# 3. weather impact
w = df[df.weather != "Unknown"].groupby("weather")["delivery_time_min"].mean().sort_values(ascending=False)
plt.figure(figsize=(7, 4)); sns.barplot(x=w.index, y=w.values, color="#12B886")
plt.title("Avg Delivery Time by Weather"); plt.ylabel("Minutes")
save("03_weather_impact.png")
insights.append(f"**{w.index[0]}** weather is the slowest ({w.iloc[0]:.1f} min); "
                f"**{w.index[-1]}** is the fastest ({w.iloc[-1]:.1f} min).")

# 4. orders by hour
h = df.groupby("order_hour").size()
plt.figure(figsize=(8, 4)); sns.barplot(x=h.index, y=h.values, color="#FA5252")
plt.title("Orders by Hour of Day"); plt.xlabel("Hour"); plt.ylabel("Orders")
save("04_orders_by_hour.png")
top3 = h.sort_values(ascending=False).head(3).index.tolist()
insights.append(f"Peak ordering hours are **{', '.join(str(x) + ':00' for x in sorted(top3))}**.")

# 5. heatmap day x hour
pv = df.pivot_table(index="day_of_week", columns="order_hour", values="order_id", aggfunc="count").reindex(DAYS)
plt.figure(figsize=(11, 4.5)); sns.heatmap(pv, cmap="YlOrRd")
plt.title("Order Volume: Day of Week x Hour")
save("05_heatmap_day_hour.png")

# 6. distance vs time
plt.figure(figsize=(6, 4.5))
sns.scatterplot(data=df.sample(min(3000, len(df)), random_state=1), x="distance_km",
                y="delivery_time_min", alpha=0.35)
plt.title("Distance vs Delivery Time")
save("06_distance_vs_time.png")
insights.append(f"Distance and delivery time have a correlation of **{df.distance_km.corr(df.delivery_time_min):.2f}**.")

# 7. rider rating vs time
plt.figure(figsize=(6, 4.5))
sns.regplot(data=df.sample(min(3000, len(df)), random_state=1), x="rider_rating",
            y="delivery_time_min", scatter_kws={"alpha": 0.2}, line_kws={"color": "red"})
plt.title("Rider Rating vs Delivery Time")
save("07_rating_vs_time.png")
insights.append(f"Rider rating vs delivery time correlation: **{df.rider_rating.corr(df.delivery_time_min):.2f}** "
                "(negative = higher-rated riders deliver faster).")

# 8. festival + multiple deliveries
f = df.groupby("festival")["delivery_time_min"].mean()
if {"Yes", "No"} <= set(f.index):
    insights.append(f"Festival days add about **{f['Yes'] - f['No']:.1f} min** to the average delivery.")
m = df.groupby("multiple_deliveries")["delivery_time_min"].mean()
insights.append(f"Carrying multiple orders raises time from {m.iloc[0]:.1f} min ({int(m.index[0])}) "
                f"to {m.iloc[-1]:.1f} min ({int(m.index[-1])}).")

# 9. city + vehicle
c = df.groupby("city")["delivery_time_min"].mean().sort_values()
plt.figure(figsize=(6, 4)); sns.barplot(x=c.index, y=c.values, color="#845EF7")
plt.title("Avg Delivery Time by City Type")
save("08_city_type.png")
v = df.groupby("vehicle_type")["delivery_time_min"].agg(["mean", "count"]).sort_values("mean")
insights.append(f"Fastest vehicle type: **{v.index[0]}** ({v['mean'].iloc[0]:.1f} min avg).")

# 10. correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(df[["rider_age", "rider_rating", "distance_km", "prep_time_min", "multiple_deliveries",
                "vehicle_condition", "delivery_time_min"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
save("09_correlation.png")

md = "# Key Insights\n\n" + "\n".join(f"- {i}" for i in insights) + "\n"
Path("reports/insights.md").write_text(md, encoding="utf-8")
print(md)
print("Charts saved in images/")