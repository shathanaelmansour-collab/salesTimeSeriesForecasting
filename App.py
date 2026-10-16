"""Interactive daily retail sales forecast using a previously trained Prophet model."""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from prophet.serialize import model_from_json

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "daily_sales.csv"
MODEL_PATH = ROOT / "models" / "prophet_model.json"

st.set_page_config(page_title="Retail Sales Forecasting", page_icon="📈", layout="wide")

@st.cache_data

def load_sales():
    data = pd.read_csv(DATA_PATH, parse_dates=["ds"])
    if not {"ds", "y"}.issubset(data.columns):
        raise ValueError("daily_sales.csv must contain 'ds' and 'y' columns")
    data = data[["ds", "y"]].copy()
    data["y"] = pd.to_numeric(data["y"], errors="raise")
    if data["ds"].isna().any() or data["y"].isna().any():
        raise ValueError("Historical sales contains missing dates or sales values")
    if data["ds"].duplicated().any():
        raise ValueError("Historical sales must have one row per date")
    return data.sort_values("ds").reset_index(drop=True)

@st.cache_resource

def load_prophet():
    with MODEL_PATH.open("r", encoding="utf-8") as handle:
        return model_from_json(handle.read())

st.title("📈 Retail Sales Forecasting")
st.caption("Ecuador Favorita dataset • Company-wide daily sales • Prophet")

try:
    history = load_sales()
    model = load_prophet()
except (OSError, ValueError, ImportError) as exc:
    st.error(f"Unable to load the data or model: {exc}")
    st.stop()

last_date = history["ds"].max()
st.info(
    f"Historical data ends on {last_date:%d %B %Y}. "
    "This is a demonstration using historical data, not a current live-sales forecast."
)

with st.sidebar:
    st.header("Forecast settings")
    horizon = st.slider("Days to forecast", min_value=7, max_value=180, value=90)
    history_days = st.slider("Historical days to show", min_value=30, max_value=365, value=180, step=5)
    generate = st.button("Generate forecast", type="primary", use_container_width=True)
    if horizon > 90:
        st.warning("Horizons longer than 90 days were not evaluated in the project's holdout test.")

st.subheader("Historical sales")
recent_history = history[history["ds"] > last_date - pd.Timedelta(days=history_days)]
history_fig = go.Figure()
history_fig.add_trace(go.Scatter(x=recent_history["ds"], y=recent_history["y"], name="Actual sales", mode="lines"))
history_fig.update_layout(xaxis_title="Date", yaxis_title="Total daily sales", hovermode="x unified")
st.plotly_chart(history_fig, use_container_width=True)

if generate:
    dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    future = pd.DataFrame({"ds": dates})
    with st.spinner("Calculating predictions..."):
        predictions = model.predict(future)
    result = predictions[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    # Clip negative sales predictions for presentation; this is a post-processing choice.
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        result[col] = result[col].clip(lower=0)

    a, b, c = st.columns(3)
    a.metric("Forecast days", f"{horizon}")
    b.metric("Average daily forecast", f"{result['yhat'].mean():,.0f}")
    c.metric("Forecast total", f"{result['yhat'].sum():,.0f}")

    st.subheader("Forecast and uncertainty interval")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent_history["ds"], y=recent_history["y"], mode="lines", name="Historical sales"))
    fig.add_trace(go.Scatter(x=result["ds"], y=result["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=result["ds"], y=result["yhat_lower"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(99,110,250,0.2)", name="Model uncertainty interval"))
    fig.add_trace(go.Scatter(x=result["ds"], y=result["yhat"], mode="lines", name="Predicted sales"))
    fig.update_layout(xaxis_title="Date", yaxis_title="Total daily sales", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily predictions")
    st.dataframe(result, use_container_width=True, hide_index=True)
    st.download_button("Download forecast CSV", result.to_csv(index=False).encode("utf-8"), "retail_sales_forecast.csv", "text/csv")
else:
    st.caption("Select a forecast horizon in the sidebar and click Generate forecast.")

st.subheader("Historical holdout comparison (90 days)")
st.dataframe(pd.DataFrame([
    {"Model": "Seasonal naive", "WAPE (%)": 10.93},
    {"Model": "Prophet V1", "WAPE (%)": 8.41},
    {"Model": "Prophet V2 + holidays", "WAPE (%)": 8.11},
    {"Model": "Tuned Prophet V1 (deployed configuration)", "WAPE (%)": 8.43},
]), hide_index=True, use_container_width=True)
st.caption("Metrics are from the historical test split, not live dashboard predictions. The deployed model was refit on all available historical data.")
