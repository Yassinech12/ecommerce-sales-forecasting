"""
Feature engineering, models and evaluation for daily revenue forecasting.

Scenario: on the evening of the last known day, forecast every day of the
next N days (here the whole Sept. - Dec. 2011 season, ~100 days).
Models that use past revenue (lags, rolling means) are run *recursively*:
each forecast is fed back as input for the following days, exactly like in
real life where future revenue is unknown.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import holidays
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

LAGS = [1, 7, 14, 28]
ROLLS = [7, 28]
UK_HOLIDAYS = holidays.UnitedKingdom(years=range(2009, 2013))


# ---------------------------------------------------------------- features
def is_closed(dates: pd.DatetimeIndex) -> np.ndarray:
    """Days with (almost) no activity: Saturdays and the Christmas break."""
    d = pd.DatetimeIndex(dates)
    xmas_break = ((d.month == 12) & (d.day >= 23)) | ((d.month == 1) & (d.day <= 3))
    return (d.dayofweek == 5) | xmas_break


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    d = pd.DatetimeIndex(dates)
    xmas = pd.to_datetime([f"{y}-12-25" for y in d.year])
    days_to_xmas = (xmas - d).days
    days_to_xmas = np.where(days_to_xmas < 0, days_to_xmas + 365, days_to_xmas)
    f = pd.DataFrame(index=d)
    f["dayofweek"] = d.dayofweek
    f["month"] = d.month
    f["dayofyear"] = d.dayofyear
    f["weekofyear"] = d.isocalendar().week.astype(int).values
    f["days_to_christmas"] = days_to_xmas
    f["is_holiday"] = [int(x in UK_HOLIDAYS) for x in d.date]
    f["is_closed"] = is_closed(d).astype(int)
    # one-hot weekday for the linear model
    for k in range(7):
        f[f"dow_{k}"] = (d.dayofweek == k).astype(int)
    # smooth yearly seasonality for the linear model
    f["sin_year"] = np.sin(2 * np.pi * d.dayofyear / 365.25)
    f["cos_year"] = np.cos(2 * np.pi * d.dayofyear / 365.25)
    return f


def lag_features(history: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Lags and rolling means of revenue, using only values strictly before each date."""
    s = history.reindex(history.index.union(dates))
    f = pd.DataFrame(index=dates)
    for l in LAGS:
        f[f"lag_{l}"] = s.shift(l).reindex(dates).values
    open_s = s.where(~is_closed(s.index))  # level of open days only
    for w in ROLLS:
        f[f"roll_mean_{w}"] = open_s.shift(1).rolling(w, min_periods=3).mean().reindex(dates).values
    return f


def make_features(series: pd.Series) -> pd.DataFrame:
    X = calendar_features(series.index).join(lag_features(series, series.index))
    return X


# ---------------------------------------------------------------- models
def get_models(random_state: int = 42) -> dict:
    return {
        "Linear regression (Ridge)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=3, random_state=random_state, n_jobs=-1),
        "XGBoost": XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, random_state=random_state, n_jobs=4),
    }


def fit(model, series: pd.Series):
    X = make_features(series).dropna()
    y = series.loc[X.index]
    model.fit(X, y)
    return model


def recursive_forecast(model, history: pd.Series, horizon_dates: pd.DatetimeIndex) -> pd.Series:
    """Forecast day by day, feeding each prediction back as history."""
    s = history.copy()
    preds = {}
    cal = calendar_features(horizon_dates)
    for d in horizon_dates:
        x = cal.loc[[d]].join(lag_features(s, pd.DatetimeIndex([d])))
        x = x.fillna(s[~is_closed(s.index)].tail(28).mean())
        y = float(model.predict(x)[0])
        y = 0.0 if cal.at[d, "is_closed"] else max(y, 0.0)
        preds[d] = y
        s.loc[d] = y
    return pd.Series(preds, name="forecast")


def seasonal_naive(series: pd.Series, horizon_dates: pd.DatetimeIndex, season: int = 364) -> pd.Series:
    """Baseline: same weekday one year earlier (364 days = 52 weeks)."""
    return series.reindex(horizon_dates - pd.Timedelta(days=season)).set_axis(horizon_dates).fillna(0)


# ---------------------------------------------------------------- metrics
def evaluate(actual: pd.Series, forecast: pd.Series) -> dict:
    """Errors on open days (closed days are trivially 0 for every model)."""
    open_days = ~is_closed(actual.index) & (actual > 0)
    a, f = actual[open_days], forecast[open_days]
    weekly_a = actual.resample("W").sum()
    weekly_f = forecast.resample("W").sum()
    return {
        "MAE": np.mean(np.abs(a - f)),
        "RMSE": np.sqrt(np.mean((a - f) ** 2)),
        "MAPE (daily, %)": np.mean(np.abs(a - f) / a) * 100,
        "MAPE (weekly, %)": np.mean(np.abs(weekly_a - weekly_f) / weekly_a) * 100,
        "Total error (%)": (f.sum() - a.sum()) / a.sum() * 100,
    }
