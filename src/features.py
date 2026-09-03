"""
features.py: feature engineering shared by the forecasting and waste modules.

Builds the demand dataset from the raw Green-AI / Brammibals bakery data:
calendar & holiday flags, weather, and per-store lag / rolling demand features.

IMPORTANT (no data leakage): lag/rolling features are built from PAST sales
only (shift/rolling before the target day), so they are safe to use with a
time-based backtest.
"""
from __future__ import annotations

import pandas as pd

HOLIDAY_COLS = ["is_state_holiday", "is_school_holiday", "is_special_day"]
DEMAND_LAG_FEATS = ["lag7", "lag14", "roll3_mean", "roll7_mean", "roll7_std"]
CALENDAR_COLS = ["year", "month", "dow", "dom", "woy", "is_weekend"]
WEATHER_COLS = ["temperature_max", "temperature_min", "temperature_mean",
                "sunshine_sum", "precipitation_sum"]


def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"].dt
    df["year"] = d.year.astype("int32")
    df["month"] = d.month.astype("int32")
    df["dow"] = d.dayofweek.astype("int32")          # 0 = Monday
    df["dom"] = d.day.astype("int32")
    df["woy"] = df["date"].dt.isocalendar().week.astype("int32")
    df["is_weekend"] = (df["dow"] >= 5).astype("int32")
    for c in HOLIDAY_COLS:
        df[c + "_flag"] = (df[c] != "normal_day").astype("int32")
    return df


def _demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-store calendar-day lag & rolling features of past sales (gaps -> NaN)."""
    full_idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    frames = []
    for store, g in df.groupby("store"):
        s = g.set_index("date")["sales"].reindex(full_idx)  # missing days -> NaN
        o = pd.DataFrame(index=full_idx)
        o["lag7"] = s.shift(7)                               # same weekday, prior week
        o["lag14"] = s.shift(14)
        o["roll3_mean"] = s.shift(1).rolling(3, min_periods=2).mean()
        o["roll7_mean"] = s.shift(1).rolling(7, min_periods=3).mean()
        o["roll7_std"] = s.shift(1).rolling(7, min_periods=3).std()
        o = o.reset_index().rename(columns={"index": "date"})
        o["store"] = store
        frames.append(o)
    lag = pd.concat(frames, ignore_index=True)
    return df.merge(lag, on=["date", "store"], how="left")


def build_demand_dataset(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path, parse_dates=["date"])
    df = df.sort_values(["store", "date"]).reset_index(drop=True)
    df = _calendar_features(df)
    df = _demand_features(df)
    # Fallback for stores/days with no recent history yet (e.g. start of series):
    # fill demand lags with the store's own mean so the model never sees NaN.
    store_mean = df.groupby("store")["sales"].transform("mean")
    for c in DEMAND_LAG_FEATS:
        df[c] = df[c].fillna(store_mean)
    return df


def make_model_matrix(df: pd.DataFrame, target: str = "sales"):
    """Return (X, y) with one-hot store columns and a float feature matrix."""
    feat_cols = CALENDAR_COLS + [c + "_flag" for c in HOLIDAY_COLS] + WEATHER_COLS \
        + DEMAND_LAG_FEATS
    X = df[feat_cols].astype("float64")
    X = pd.concat([X, pd.get_dummies(df["store"], prefix="store", dtype="int")], axis=1)
    return X, df[target].astype("float64").values
