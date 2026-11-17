"""Step 5: Streamlit dashboard.  Run: streamlit run app/dashboard.py"""
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Food Delivery Analytics", page_icon="🍔", layout="wide")
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TRAFFIC = ["Low", "Medium", "High", "Jam"]


@st.cache_data
def load():
    return pd.read_csv("data/processed/clean.csv", parse_dates=["order_date"])


df = load()
st.title("🍔 Food Delivery & Restaurant Analytics")

# ---- sidebar filters
st.sidebar.header("Filters")
city = st.sidebar.multiselect("City type", sorted(df.city.unique()), default=sorted(df.city.unique()))
traffic = st.sidebar.multiselect("Traffic", sorted(df.traffic.unique()), default=sorted(df.traffic.unique()))
weather = st.sidebar.multiselect("Weather", sorted(df.weather.unique()), default=sorted(df.weather.unique()))
d = df[df.city.isin(city) & df.traffic.isin(traffic) & df.weather.isin(weather)]
if d.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ---- KPIs
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Orders", f"{len(d):,}")
c2.metric("Avg Delivery Time", f"{d.delivery_time_min.mean():.1f} min")
c3.metric("Avg Distance", f"{d.distance_km.mean():.1f} km")
c4.metric("Avg Rider Rating", f"{d.rider_rating.mean():.2f}")
c5.metric("% Slow (>40 min)", f"{(d.delivery_time_min > 40).mean() * 100:.1f}%")

tab1, tab2, tab3, tab4 = st.tabs(["Demand", "Delivery Performance", "Riders & Vehicles", "Predict Time"])

with tab1:
    a, b = st.columns(2)
    hourly = d.groupby("order_hour").size().reset_index(name="orders")
    a.plotly_chart(px.bar(hourly, x="order_hour", y="orders", title="Orders by Hour"), use_container_width=True)
    daily = d.groupby("order_date").size().reset_index(name="orders")
    b.plotly_chart(px.line(daily, x="order_date", y="orders", title="Daily Orders"), use_container_width=True)
    pv = d.pivot_table(index="day_of_week", columns="order_hour", values="order_id", aggfunc="count").reindex(DAYS)
    st.plotly_chart(px.imshow(pv, aspect="auto", color_continuous_scale="YlOrRd",
                              title="Demand Heatmap: Day x Hour"), use_container_width=True)

with tab2:
    a, b = st.columns(2)
    t = d.groupby("traffic").delivery_time_min.mean().reindex([x for x in TRAFFIC if x in d.traffic.unique()]).reset_index()
    a.plotly_chart(px.bar(t, x="traffic", y="delivery_time_min", title="Avg Time by Traffic"), use_container_width=True)
    w = d.groupby("weather").delivery_time_min.mean().sort_values(ascending=False).reset_index()
    b.plotly_chart(px.bar(w, x="weather", y="delivery_time_min", title="Avg Time by Weather"), use_container_width=True)
    st.plotly_chart(px.scatter(d.sample(min(2500, len(d)), random_state=1), x="distance_km", y="delivery_time_min",
                               color="traffic", opacity=0.6, title="Distance vs Delivery Time"),
                    use_container_width=True)

with tab3:
    a, b = st.columns(2)
    v = d.groupby("vehicle_type").agg(avg_time=("delivery_time_min", "mean"), orders=("order_id", "count")).reset_index()
    a.plotly_chart(px.bar(v, x="vehicle_type", y="avg_time", title="Avg Time by Vehicle"), use_container_width=True)
    m = d.groupby("multiple_deliveries").delivery_time_min.mean().reset_index()
    b.plotly_chart(px.bar(m, x="multiple_deliveries", y="delivery_time_min", title="Effect of Multiple Orders"),
                   use_container_width=True)
    st.plotly_chart(px.box(d, x="order_type", y="delivery_time_min", title="Delivery Time by Order Type"),
                    use_container_width=True)

with tab4:
    model_path = Path("models/best_model.joblib")
    if not model_path.exists():
        st.info("Train the model first: `python src/model.py`")
    else:
        model = joblib.load(model_path)
        a, b, c = st.columns(3)
        dist = a.slider("Distance (km)", 0.5, 30.0, 5.0)
        hour = a.slider("Order hour", 0, 23, 20)
        prep = a.slider("Prep time (min)", 3, 30, 10)
        rating = a.slider("Rider rating", 1.0, 5.0, 4.6)
        tr = b.selectbox("Traffic", sorted(df.traffic.unique()))
        we = b.selectbox("Weather", sorted(df.weather.unique()))
        fe = b.selectbox("Festival", sorted(df.festival.unique()))
        multi = b.selectbox("Multiple deliveries", [0, 1, 2, 3])
        ve = c.selectbox("Vehicle", sorted(df.vehicle_type.unique()))
        ot = c.selectbox("Order type", sorted(df.order_type.unique()))
        ci = c.selectbox("City type", sorted(df.city.unique()))
        row = pd.DataFrame([{"rider_age": 28, "rider_rating": rating, "distance_km": dist, "order_hour": hour,
                             "vehicle_condition": 1, "multiple_deliveries": multi, "prep_time_min": prep,
                             "weather": we, "traffic": tr, "order_type": ot, "vehicle_type": ve,
                             "city": ci, "festival": fe}])
        st.metric("Predicted delivery time", f"{model.predict(row)[0]:.0f} min")

with st.expander("View data"):
    st.dataframe(d.head(500))