"""
Load the trained AeroCast models (copied from the ML repo by scripts/export_models.py)
and produce next-step AQI and pollutant forecasts from simulated readings.
"""

import json
import math
from pathlib import Path

import joblib

from .cpcb_aqi import aqi_to_category
from .features import POLLUTANTS, STEPS_FOR_FULL_AQI, STEPS_FOR_FULL_CONTEXT, build_feature_row

UNITS = {"PM2.5": "ug/m3", "PM10": "ug/m3", "NO2": "ug/m3", "SO2": "ug/m3",
         "CO": "mg/m3", "Ozone": "ug/m3", "NH3": "ug/m3"}

BASIS = ("Forecast produced by models trained on CPCB site_113 (Shadipur, Delhi) hourly data, "
         "applied to SIMULATED pollutant inputs from Wokwi potentiometers. Each received reading is "
         "treated as one hourly step (time-compressed). This is a demonstration of the pipeline, "
         "not a forecast of real air quality.")


def _round(x, nd=2):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


class Predictor:
    def __init__(self, model_dir: str | Path):
        d = Path(model_dir)
        self.aqi_model = joblib.load(d / "xgboost" / "aqi_model.joblib")
        self.aqi_prep = joblib.load(d / "xgboost" / "preprocessing.joblib")
        self.pol_model = joblib.load(d / "linear_regression" / "pollutant_model.joblib")
        self.pol_prep = joblib.load(d / "linear_regression" / "preprocessing.joblib")
        self.aqi_meta = json.loads((d / "xgboost" / "metadata.json").read_text())
        self.pol_meta = json.loads((d / "linear_regression" / "metadata.json").read_text())
        self.ranges = json.loads((d / "training_ranges.json").read_text())
        self.feature_names = list(self.aqi_prep["feature_names"])
        if list(self.pol_prep["feature_names"]) != self.feature_names:
            raise ValueError("AQI and pollutant models expect different feature orders")
        self.pol_targets = list(self.pol_prep["target_pollutants"])

    # ------------------------------------------------------------------
    def model_info(self) -> dict:
        return {
            "aqi_model": {
                "family": "XGBoost",
                "file": "models/xgboost/aqi_model.joblib",
                "target": "next-hour CPCB AQI",
                "test_metrics": self.aqi_meta["aqi_metrics"]["test"],
            },
            "pollutant_model": {
                "family": "Linear Regression",
                "file": "models/linear_regression/pollutant_model.joblib",
                "targets": self.pol_targets,
                "test_r2": {p: m["R2"] for p, m in self.pol_meta["pollutant_test_metrics"].items()},
                "test_mae": {p: m["MAE"] for p, m in self.pol_meta["pollutant_test_metrics"].items()},
            },
            "selection_rationale": ("XGBoost has the lowest test RMSE / highest R2 for AQI; Linear Regression "
                                    "has the highest test R2 on 5 of 7 pollutants in the saved metadata."),
            "n_features": len(self.feature_names),
            "training_data": self.ranges.get("_source"),
            "training_ranges": {k: v for k, v in self.ranges.items() if not k.startswith("_")},
            "basis": BASIS,
        }

    # ------------------------------------------------------------------
    def predict(self, steps: list[dict]) -> dict:
        X, diag = build_feature_row(steps, self.feature_names)

        X_imp = self.aqi_prep["imputer"].transform(X)
        aqi_next = float(self.aqi_model.predict(X_imp)[0])

        X_scaled = self.pol_prep["scaler"].transform(self.pol_prep["imputer"].transform(X))
        pol_raw = self.pol_model.predict(X_scaled)[0]

        pollutants, clipped = {}, []
        for name, val in zip(self.pol_targets, pol_raw):
            val = float(val)
            if val < 0:
                clipped.append(name)
                val = 0.0
            pollutants[name] = {"value": _round(val), "unit": UNITS[name]}

        latest = steps[-1]
        out_of_range = {}
        for p in POLLUTANTS:
            r = self.ranges[p]
            if not (r["min"] <= latest[p] <= r["max"]):
                out_of_range[p] = {"value": latest[p], "train_min": r["min"], "train_max": r["max"]}

        n = diag["steps_available"]
        status = "ok" if n >= STEPS_FOR_FULL_CONTEXT else "warming_up"
        cur_aqi = diag["current_aqi"]

        return {
            "status": status,
            "basis": BASIS,
            "current_aqi_estimate": {
                "value": _round(cur_aqi, 1),
                "category": aqi_to_category(cur_aqi),
                "sub_indices": diag["current_sub_indices"],
                "method": ("CPCB max sub-index of rolling means over the last <=24 steps "
                           "(<=8 for CO/O3) of simulated readings"),
                "provisional": n < STEPS_FOR_FULL_AQI,
                "kind": "estimate_from_simulated_inputs",
            },
            "forecast_next_step": {
                "kind": "model_forecast",
                "aqi": {"value": _round(aqi_next, 1), "category": aqi_to_category(aqi_next), "model": "XGBoost"},
                "pollutants": pollutants,
                "pollutant_model": "Linear Regression",
            },
            "data_quality": {
                "steps_available": n,
                "steps_for_full_context": STEPS_FOR_FULL_CONTEXT,
                "lags_backfilled": diag["lags_backfilled"],
                "imputed_features": diag["imputed_features"],
                "out_of_training_range": out_of_range,
                "negative_forecasts_clipped": clipped,
                "temporal_reference": diag["temporal_reference"],
            },
        }
