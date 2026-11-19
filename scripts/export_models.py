"""
Copy the already-trained AeroCast models from the ML repository into this bridge.

Nothing is retrained here. The script copies:
  * XGBoost next-hour AQI model            (models/xgboost/aqi_model.joblib)
  * Linear Regression next-hour pollutants (models/linear_regression/pollutant_model.joblib)
  * their preprocessing bundles (median imputer + standard scaler, fitted on train split only)
  * their metadata.json (features, saved test metrics)
and derives models/training_ranges.json (min / median / max of each pollutant in train.csv),
used only to flag simulated inputs that fall outside what the models were trained on.

Usage:
    python scripts/export_models.py ../aerocast-iot-air-quality-forecasting
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ARTIFACTS = {
    "xgboost": ["aqi_model.joblib", "preprocessing.joblib", "metadata.json"],
    "linear_regression": ["pollutant_model.joblib", "preprocessing.joblib", "metadata.json"],
}
RANGE_COLS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone", "NH3", "AQI"]


def main(ml_repo: Path) -> None:
    out = Path(__file__).resolve().parent.parent / "models"
    for family, files in ARTIFACTS.items():
        (out / family).mkdir(parents=True, exist_ok=True)
        for name in files:
            src = ml_repo / "models" / family / name
            shutil.copy2(src, out / family / name)
            print(f"copied {src} -> {out / family / name}")

    shutil.copy2(ml_repo / "data" / "processed" / "feature_metadata.json", out / "feature_metadata.json")

    train = pd.read_csv(ml_repo / "data" / "processed" / "train.csv", index_col=0)
    ranges = {
        c: {
            "min": round(float(train[c].min()), 3),
            "median": round(float(train[c].median()), 3),
            "max": round(float(train[c].max()), 3),
        }
        for c in RANGE_COLS
    }
    ranges["_source"] = "data/processed/train.csv (CPCB site_113 Shadipur, Delhi; 2023-01 to 2024-05)"
    (out / "training_ranges.json").write_text(json.dumps(ranges, indent=2))
    print(f"wrote {out / 'training_ranges.json'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]).resolve())
