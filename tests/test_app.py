import math
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import create_app  # noqa: E402
from aerocast_ml.predictor import Predictor  # noqa: E402

PREDICTOR = Predictor(ROOT / "models")

BASE = {"device_id": "TEST_ESP32", "temperature": 37.9, "humidity": 67.5, "mq2_raw": 1200,
        "mq2_relative_percent": 29.3}
POLS = {"pm25": 120.0, "pm10": 250.0, "no2": 60.0, "so2": 15.0, "co": 1.5, "o3": 40.0, "nh3": 50.0, "aqi": 300}


@pytest.fixture
def client():
    return create_app(predictor=PREDICTOR).test_client()


def post(client, payload):
    return client.post("/sensor-data", json=payload)


# ---------------- original behaviour ----------------
def test_root_and_health(client):
    assert client.get("/").get_json()["status"] == "online"
    h = client.get("/health").get_json()
    assert h["status"] == "healthy" and h["ml_loaded"] is True


def test_latest_empty_is_404(client):
    assert client.get("/latest").status_code == 404
    assert client.get("/history").get_json() == []


def test_legacy_payload_without_pollutants(client):
    r = post(client, BASE)
    body = r.get_json()
    assert r.status_code == 200 and body["success"] is True
    assert body["prediction"] is None and body["display"]["status"] == "unavailable"
    assert client.get("/latest").get_json()["mq2_raw"] == 1200


@pytest.mark.parametrize("payload,code", [
    ("not json", 400),
    ({"temperature": 25}, 400),
    ({**BASE, "temperature": "hot"}, 400),
    ({**BASE, "humidity": 150}, 400),
    ({**BASE, "mq2_raw": 9999}, 400),
    ({**BASE, "temperature": True}, 400),
    ({**BASE, "pm25": 10.0}, 400),                 # partial pollutant set
    ({**BASE, **POLS, "so2": -1}, 400),            # negative concentration
])
def test_invalid_payloads(client, payload, code):
    if isinstance(payload, str):
        r = client.post("/sensor-data", data=payload, content_type="application/json")
    else:
        r = post(client, payload)
    assert r.status_code == code
    assert client.get("/history").get_json() == []


def test_nan_rejected(client):
    r = client.post("/sensor-data", data='{"temperature": NaN, "humidity": 50, "mq2_raw": 1}',
                    content_type="application/json")
    assert r.status_code == 400


def test_history_order_and_limit(client):
    for i in range(5):
        post(client, {**BASE, "mq2_raw": i})
    hist = client.get("/history?limit=3").get_json()
    assert [h["mq2_raw"] for h in hist] == [4, 3, 2]
    assert client.get("/history?limit=x").status_code == 400


def test_history_capped_at_100(client):
    for i in range(105):
        post(client, {**BASE, "mq2_raw": i % 4096})
    assert len(client.get("/history").get_json()) == 100


# ---------------- ML integration ----------------
def test_prediction_structure(client):
    body = post(client, {**BASE, **POLS}).get_json()
    p = body["prediction"]
    assert p["status"] == "warming_up"
    assert p["current_aqi_estimate"]["provisional"] is True
    fc = p["forecast_next_step"]
    assert math.isfinite(fc["aqi"]["value"]) and 0 <= fc["aqi"]["value"] <= 600
    assert set(fc["pollutants"]) == {"PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone", "NH3"}
    assert all(v["value"] >= 0 for v in fc["pollutants"].values())
    assert p["data_quality"]["imputed_features"] == ["Benzene", "Eth-Benzene", "MP-Xylene", "NO", "NOx", "Toluene"]
    d = body["display"]
    assert d["status"] == "warming_up" and d["steps"] == 1 and isinstance(d["aqi_next"], int)
    assert body["data"]["device_pm25_aqi"] == 300


def test_current_aqi_matches_cpcb_on_constant_stream(client):
    # PM2.5 = 120 -> sub-index 300 (top of 'Poor' band); it is the max sub-index here
    body = post(client, {**BASE, **POLS}).get_json()
    cur = body["prediction"]["current_aqi_estimate"]
    assert cur["value"] == pytest.approx(300.0, abs=0.1)
    assert cur["category"] == "Poor"


def test_warms_up_then_ok(client):
    for i in range(48):
        body = post(client, {**BASE, **POLS}).get_json()
    assert body["prediction"]["status"] == "ok"
    assert body["prediction"]["data_quality"]["lags_backfilled"] is False
    assert body["prediction"]["current_aqi_estimate"]["provisional"] is False


def test_compact_response_for_device(client):
    body = client.post("/sensor-data?compact=1", json={**BASE, **POLS}).get_json()
    assert set(body) == {"success", "received_at", "display"}
    assert body["display"]["cat_now"] == "Poor"
    assert client.get("/latest").get_json()["prediction"]["status"] == "warming_up"


def test_buffers_are_per_device(client):
    for _ in range(3):
        post(client, {**BASE, **POLS})
    other = post(client, {**BASE, **POLS, "device_id": "OTHER"}).get_json()
    assert other["prediction"]["data_quality"]["steps_available"] == 1


def test_out_of_training_range_flag(client):
    body = post(client, {**BASE, **POLS, "so2": 450.0, "nh3": 900.0}).get_json()
    assert set(body["prediction"]["data_quality"]["out_of_training_range"]) == {"SO2", "NH3"}
    assert body["display"]["ood"] == 2


def test_prediction_and_model_info_endpoints(client):
    assert client.get("/predictions/latest").status_code == 404
    post(client, {**BASE, **POLS})
    assert client.get("/predictions/latest").get_json()["forecast_next_step"]["aqi"]["model"] == "XGBoost"
    info = client.get("/model-info").get_json()
    assert info["n_features"] == 119 and info["aqi_model"]["family"] == "XGBoost"


def test_bridge_works_without_ml():
    c = create_app(predictor=None, ml_error="boom", load_ml=False).test_client()
    body = c.post("/sensor-data", json={**BASE, **POLS}).get_json()
    assert body["success"] is True and body["display"]["status"] == "unavailable"
    assert c.get("/model-info").status_code == 503


def test_dashboard_served(client):
    r = client.get("/dashboard")
    assert r.status_code == 200 and b"AeroCast" in r.data


# ---------------- parity with training pipeline ----------------
ML_REPO = Path(os.environ.get("AEROCAST_ML_REPO", ROOT.parent / "aerocast-iot-air-quality-forecasting"))


@pytest.mark.skipif(not (ML_REPO / "data/processed/test.csv").exists(), reason="ML repo not available")
def test_feature_parity_with_training_data():
    import numpy as np
    import pandas as pd
    from aerocast_ml.features import IST, POLLUTANTS, UNSIMULATED, build_feature_row

    df = pd.read_csv(ML_REPO / "data/processed/test.csv", index_col=0, parse_dates=True)
    contiguous = df.index.to_series().diff().eq(pd.Timedelta("1h"))
    checked = 0
    for end in range(60, len(df), 97):
        w = df.iloc[end - 47:end + 1]
        if not contiguous.iloc[end - 46:end + 1].all() or w[POLLUTANTS].isna().any().any():
            continue
        X, _ = build_feature_row(w[POLLUTANTS].to_dict("records"), PREDICTOR.feature_names,
                                 now=w.index[-1].to_pydatetime().replace(tzinfo=IST))
        ref = w.iloc[[-1]][PREDICTOR.feature_names]
        X[UNSIMULATED] = ref[UNSIMULATED].values
        live = PREDICTOR.aqi_model.predict(PREDICTOR.aqi_prep["imputer"].transform(X))[0]
        expected = PREDICTOR.aqi_model.predict(PREDICTOR.aqi_prep["imputer"].transform(ref))[0]
        assert live == pytest.approx(expected, abs=0.5)
        checked += 1
        if checked == 6:
            break
    assert checked >= 3
