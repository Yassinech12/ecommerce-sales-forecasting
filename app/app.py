"""
Streamlit app: explore sales and forecast daily revenue.

Run from the project root:  streamlit run app/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import forecasting as fc  # noqa: E402

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8985"
BASELINE = "Seasonal naive (same day last year)"
BACKTEST_CUTOFF = pd.Timestamp("2011-08-31")

st.set_page_config(page_title="E-commerce Sales Forecasting", page_icon="📈", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "processed" / "daily_sales_by_country.csv", parse_dates=["date"])


def money(x: float) -> str:
    if abs(x) >= 1e6:
        return f"£{x / 1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"£{x / 1e3:.1f}k"
    return f"£{x:,.0f}"


@st.cache_resource(show_spinner=False)
def backtest(country: str, model_name: str):
    s = series_for(country)
    train = s[:BACKTEST_CUTOFF]
    test = s[BACKTEST_CUTOFF + pd.Timedelta(days=1):]
    model = fc.fit(fc.get_models()[model_name], train)
    pred = fc.recursive_forecast(model, train, test.index)
    naive = fc.seasonal_naive(s, test.index)
    metrics = pd.DataFrame({model_name: fc.evaluate(test, pred), BASELINE: fc.evaluate(test, naive)}).T
    return test, pred, naive, metrics


@st.cache_resource(show_spinner=False)
def future_forecast(country: str, model_name: str, horizon: int) -> pd.Series:
    s = series_for(country)
    model = fc.fit(fc.get_models()[model_name], s)
    dates = pd.date_range(s.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    return fc.recursive_forecast(model, s, dates)


def series_for(country: str) -> pd.Series:
    d = load_data()
    return d[d["country"] == country].set_index("date")["revenue"].asfreq("D").fillna(0)


def line(fig, x, y, name, color, width=2, dash=None):
    fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines",
                             line=dict(color=color, width=width, dash=dash),
                             hovertemplate="%{x|%a %d %b %Y}<br>£%{y:,.0f}<extra>" + name + "</extra>"))


def style(fig, title, height=380):
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=16)), height=height,
                      margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
                      yaxis=dict(tickprefix="£", tickformat="~s", gridcolor="rgba(128,128,128,0.2)"),
                      xaxis=dict(showgrid=False))
    return fig


# ------------------------------------------------------------------ sidebar
data = load_data()
st.sidebar.title("📈 Sales Forecasting")
st.sidebar.caption("Online retailer (UK), Dec. 2009 – Dec. 2011 · UCI Online Retail II")
country = st.sidebar.selectbox("Market", list(dict.fromkeys(data["country"])), index=0)
model_name = st.sidebar.selectbox("Model", ["XGBoost", "Random Forest", "Linear regression (Ridge)"], index=0)
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 90, 42, step=7)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built by **Yassine CHARIT** · [GitHub](https://github.com/Yassinech12/ecommerce-sales-forecasting)"
    " · [LinkedIn](https://www.linkedin.com/in/yassine-charit/)")

s = series_for(country)
d = data[data["country"] == country].set_index("date")

st.title("E-commerce sales forecasting")
st.caption(f"Market: **{country}** · Model: **{model_name}**")
if country not in ("All countries", "United Kingdom"):
    st.warning(f"**{country}** is a small market (a few orders per week, many days without sales): "
               "forecasts are much less reliable than for the UK or the whole business.")

tab_overview, tab_backtest, tab_forecast = st.tabs(["📊 Overview", "🎯 Model performance", "🔮 Forecast"])

# ------------------------------------------------------------------ overview
with tab_overview:
    last12 = d.loc[d.index.max() - pd.DateOffset(years=1) + pd.Timedelta(days=1):]
    prev12 = d.loc[d.index.max() - pd.DateOffset(years=2) + pd.Timedelta(days=1): d.index.max() - pd.DateOffset(years=1)]
    growth = last12["revenue"].sum() / prev12["revenue"].sum() - 1 if prev12["revenue"].sum() else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue, last 12 months", money(last12["revenue"].sum()),
              f"{growth:+.1%} vs previous 12 months" if growth is not None else None)
    c2.metric("Orders, last 12 months", f"{last12['orders'].sum():,.0f}")
    c3.metric("Average order value", money(last12["revenue"].sum() / max(last12["orders"].sum(), 1)))
    open_days = last12[~fc.is_closed(last12.index)]
    c4.metric("Average revenue per open day", money(open_days["revenue"].mean()))

    monthly = d["revenue"].resample("MS").sum()
    fig = go.Figure(go.Bar(x=monthly.index, y=monthly.values, marker_color=BLUE,
                           hovertemplate="%{x|%b %Y}<br>£%{y:,.0f}<extra></extra>"))
    st.plotly_chart(style(fig, "Monthly revenue (Dec. 2011 = 9 days only)"), width="stretch")

    col_a, col_b = st.columns(2)
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = d["revenue"].groupby(d.index.day_name()).mean().reindex(order)
    fig = go.Figure(go.Bar(x=[x[:3] for x in order], y=dow.values, marker_color=BLUE,
                           hovertemplate="%{x}<br>£%{y:,.0f}<extra></extra>"))
    col_a.plotly_chart(style(fig, "Average revenue by weekday", 320), width="stretch")
    roll = s.where(~fc.is_closed(s.index)).rolling(28, min_periods=10).mean()
    fig = go.Figure()
    line(fig, roll.index, roll.values, "28-day average (open days)", BLUE)
    col_b.plotly_chart(style(fig, "Revenue trend", 320), width="stretch")

# ------------------------------------------------------------------ backtest
with tab_backtest:
    st.markdown(
        "**Backtest:** the model is trained on data until **31 Aug. 2011**, then forecasts every day of the "
        "Christmas season (**1 Sept. – 9 Dec. 2011**) in one go, without seeing the actual values.")
    with st.spinner("Training the model and forecasting the season..."):
        test, pred, naive, metrics = backtest(country, model_name)

    m, b = metrics.loc[model_name], metrics.loc[BASELINE]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE (£ per open day)", money(m["MAE"]),
              f"{m['MAE'] / b['MAE'] - 1:+.0%} vs baseline", delta_color="inverse")
    c2.metric("Weekly error (MAPE)", f"{m['MAPE (weekly, %)']:.1f}%",
              f"{m['MAPE (weekly, %)'] - b['MAPE (weekly, %)']:+.1f} pts vs baseline", delta_color="inverse")
    c3.metric("Season revenue, actual", money(test.sum()))
    c4.metric("Season revenue, forecast", money(pred.sum()), f"{pred.sum() / test.sum() - 1:+.1%}", delta_color="off")

    level = st.radio("View", ["Weekly", "Daily"], horizontal=True)
    fig = go.Figure()
    if level == "Weekly":
        wa, wp, wn = (x.resample("W").sum().iloc[:-1] for x in (test, pred, naive))
        line(fig, wa.index, wa.values, "Actual", BLUE)
        line(fig, wp.index, wp.values, model_name, ORANGE)
        line(fig, wn.index, wn.values, "Baseline: same week last year", GRAY, dash="dash")
    else:
        line(fig, test.index, test.values, "Actual", BLUE, width=1.5)
        line(fig, pred.index, pred.values, model_name, ORANGE, width=1.8)
    st.plotly_chart(style(fig, f"{level} revenue: actual vs forecast", 420), width="stretch")

    with st.expander("All metrics"):
        st.dataframe(metrics.style.format("{:,.1f}"), width="stretch")
        st.caption("MAE / RMSE in £ on open days · MAPE = mean absolute percentage error · "
                   "Total error = forecast vs actual over the whole season.")

# ------------------------------------------------------------------ forecast
with tab_forecast:
    st.markdown(
        f"The model is retrained on **all the data** (until {s.index.max():%d %b %Y}) "
        f"and forecasts the next **{horizon} days**. Saturdays and the Christmas break (23 Dec. – 3 Jan.) are closed days.")
    with st.spinner("Forecasting..."):
        fut = future_forecast(country, model_name, horizon)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Forecast revenue, next {horizon} days", money(fut.sum()))
    c2.metric("Open days in the period", int((~fc.is_closed(fut.index)).sum()))
    c3.metric("Average per open day", money(fut[~fc.is_closed(fut.index)].mean()))

    hist = s[s.index.max() - pd.Timedelta(days=90):]
    fig = go.Figure()
    line(fig, hist.index, hist.values, "Actual (last 90 days)", BLUE, width=1.5)
    line(fig, fut.index, fut.values, "Forecast", ORANGE, width=1.8)
    fig.add_vline(x=s.index.max(), line_dash="dash", line_color=GRAY)
    st.plotly_chart(style(fig, "Daily revenue forecast", 420), width="stretch")

    weekly = fut.resample("W-SUN").sum().rename("forecast_revenue").to_frame()
    weekly.index = [f"Week ending {x:%d %b %Y}" for x in weekly.index]
    col_a, col_b = st.columns([2, 1])
    col_a.dataframe(weekly.style.format("£{:,.0f}"), width="stretch")
    csv = fut.rename("forecast_revenue").rename_axis("date").reset_index().to_csv(index=False).encode()
    col_b.download_button("⬇️ Download the daily forecast (CSV)", csv,
                          file_name=f"forecast_{country.replace(' ', '_').lower()}_{horizon}d.csv",
                          mime="text/csv")
    col_b.info("This is a point forecast. Large one-off wholesale orders can make daily values "
               "vary a lot: use weekly totals for planning.")
