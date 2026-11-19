"""
Build the 119-column feature row the trained AeroCast models expect, from a stream of
simulated pollutant readings sent by the Wokwi ESP32.

Mirrors src/preprocessing/feature_engineering.py in the ML repo:
  * current pollutant values + CPCB AQI
  * temporal / cyclic features
  * lags 1, 2, 3, 6, 12, 24 steps
  * rolling means (3, 6, 12, 24) and stds (6, 24)

Differences from training, all reported back to the caller in `diagnostics`:
  * TIME-COMPRESSED: each received reading is treated as one model "hour" step.
    The ESP32 uploads every ~10 s, so 24 readings (~4 min) stand in for 24 hours.
  * Temporal features (hour, day, month, ...) use the server's current wall-clock time in IST.
  * Benzene, Eth-Benzene, MP-Xylene, NO, NOx, Toluene are not simulated by the circuit;
    they are left missing and filled by the training median imputer.
  * Until enough steps exist, lags fall back to the oldest available reading and rolling
    windows use the readings available, so early forecasts are flagged `warming_up`.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .cpcb_aqi import calculate_sub_index

POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone", "NH3"]
LAG_COLS = POLLUTANTS + ["AQI"]
LAGS = [1, 2, 3, 6, 12, 24]
WINDOWS = [3, 6, 12, 24]
UNSIMULATED = ["Benzene", "Eth-Benzene", "MP-Xylene", "NO", "NOx", "Toluene"]
AVG_24H = ["PM2.5", "PM10", "NO2", "SO2", "NH3"]
AVG_8H = ["CO", "Ozone"]

# A full feature context needs AQI_lag24, and AQI itself needs a 24-step average.
STEPS_FOR_FULL_CONTEXT = 48
STEPS_FOR_FULL_AQI = 24

IST = timezone(timedelta(hours=5, minutes=30))


def stream_aqi(df: pd.DataFrame) -> pd.Series:
    """CPCB AQI at every step: max sub-index of rolling 24-step (PM, NO2, SO2, NH3) and
    8-step (CO, O3) means. Uses whatever steps exist (min_periods=1) so a value is
    available from the first reading; callers flag it provisional until 24 steps exist."""
    subs = pd.DataFrame(index=df.index)
    for p in AVG_24H:
        subs[p] = df[p].rolling(24, min_periods=1).mean().apply(lambda c, p=p: calculate_sub_index(c, p))
    for p in AVG_8H:
        subs[p] = df[p].rolling(8, min_periods=1).mean().apply(lambda c, p=p: calculate_sub_index(c, p))
    return subs.max(axis=1)


def current_sub_indices(df: pd.DataFrame) -> dict:
    out = {}
    for p in POLLUTANTS:
        w = 24 if p in AVG_24H else 8
        out[p] = round(float(calculate_sub_index(df[p].iloc[-w:].mean(), p)), 1)
    return out


def build_feature_row(steps: list[dict], feature_names: list[str], now: datetime | None = None):
    """steps: oldest -> newest, each {pollutant: value}. Returns (1-row DataFrame, diagnostics)."""
    if not steps:
        raise ValueError("at least one step is required")
    now = (now or datetime.now(timezone.utc)).astimezone(IST)

    df = pd.DataFrame(steps, columns=POLLUTANTS).astype(float).reset_index(drop=True)
    df["AQI"] = stream_aqi(df)
    n = len(df)

    row = {c: np.nan for c in UNSIMULATED}
    for c in LAG_COLS:
        row[c] = df[c].iloc[-1]

    row["hour"] = now.hour
    row["day"] = now.day
    row["month"] = now.month
    row["dayofweek"] = now.weekday()
    row["is_weekend"] = int(now.weekday() >= 5)
    row["hour_sin"] = np.sin(2 * np.pi * now.hour / 24.0)
    row["hour_cos"] = np.cos(2 * np.pi * now.hour / 24.0)
    row["month_sin"] = np.sin(2 * np.pi * (now.month - 1) / 12.0)
    row["month_cos"] = np.cos(2 * np.pi * (now.month - 1) / 12.0)

    for c in LAG_COLS:
        s = df[c]
        for k in LAGS:
            row[f"{c}_lag{k}"] = s.iloc[max(n - 1 - k, 0)]
        for w in WINDOWS:
            row[f"{c}_roll_mean_{w}h"] = s.iloc[-w:].mean()
        for w in (6, 24):
            tail = s.iloc[-w:]
            row[f"{c}_roll_std_{w}h"] = tail.std() if len(tail) >= 2 else 0.0

    missing = [f for f in feature_names if f not in row]
    if missing:
        raise KeyError(f"feature builder does not produce: {missing}")
    X = pd.DataFrame([[row[f] for f in feature_names]], columns=feature_names)

    diagnostics = {
        "steps_available": n,
        "steps_for_full_context": STEPS_FOR_FULL_CONTEXT,
        "lags_backfilled": n <= max(LAGS) or n < STEPS_FOR_FULL_CONTEXT,
        "aqi_provisional": n < STEPS_FOR_FULL_AQI,
        "imputed_features": UNSIMULATED,
        "temporal_reference": now.isoformat(timespec="seconds"),
        "current_aqi": float(df["AQI"].iloc[-1]),
        "current_sub_indices": current_sub_indices(df),
    }
    return X, diagnostics
