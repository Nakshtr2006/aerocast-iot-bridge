"""
AeroCast IoT Bridge

Receives SIMULATED sensor readings from the Wokwi ESP32 (DHT22, MQ-2 and seven
potentiometer-driven pollutant channels), stores recent readings in memory, runs the
trained AeroCast models on the simulated pollutant stream and returns the forecast
to the device. Also serves a dashboard at /dashboard.

Backwards compatible with the original bridge: same routes, same required fields
(temperature, humidity, mq2_raw); pollutant fields are optional.
"""

import logging
import math
import os
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent
HISTORY_SIZE = 100
STEP_BUFFER_SIZE = 72  # >= 48 needed for full model context

# Device JSON key -> model pollutant name
POLLUTANT_FIELDS = {
    "pm25": "PM2.5",
    "pm10": "PM10",
    "no2": "NO2",
    "so2": "SO2",
    "co": "CO",
    "o3": "Ozone",
    "nh3": "NH3",
}

log = logging.getLogger("aerocast")


def _num(data, key, lo=None, hi=None):
    v = data[key]
    if isinstance(v, bool):
        raise ValueError(f"{key} must be numeric")
    v = float(v)
    if not math.isfinite(v):
        raise ValueError(f"{key} must be a finite number")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ValueError(f"{key} out of range [{lo}, {hi}]")
    return v


def load_predictor():
    try:
        from aerocast_ml.predictor import Predictor

        return Predictor(BASE_DIR / "models"), None
    except Exception as exc:  # keep the bridge usable even if ML can't load
        log.exception("ML models failed to load")
        return None, f"{type(exc).__name__}: {exc}"


def display_summary(prediction):
    """Small, flat object for the ESP32 (cheap to parse on the device)."""
    if not prediction or "error" in prediction:
        return {"status": "unavailable"}
    cur = prediction["current_aqi_estimate"]
    fc = prediction["forecast_next_step"]
    dq = prediction["data_quality"]
    return {
        "status": prediction["status"],
        "aqi_now": round(cur["value"]) if cur["value"] is not None else None,
        "cat_now": cur["category"],
        "aqi_next": round(fc["aqi"]["value"]) if fc["aqi"]["value"] is not None else None,
        "cat_next": fc["aqi"]["category"],
        "pm25_next": fc["pollutants"]["PM2.5"]["value"],
        "pm10_next": fc["pollutants"]["PM10"]["value"],
        "steps": dq["steps_available"],
        "steps_full": dq["steps_for_full_context"],
        "ood": len(dq["out_of_training_range"]),
    }


def create_app(predictor=None, ml_error=None, load_ml=True):
    app = Flask(__name__)
    CORS(app)

    if predictor is None and load_ml:
        predictor, ml_error = load_predictor()

    lock = threading.Lock()
    history = deque(maxlen=HISTORY_SIZE)
    steps = {}  # device_id -> deque of {pollutant: value}
    state = {"received": 0, "started_at": datetime.now(timezone.utc).isoformat()}

    @app.get("/")
    def home():
        return jsonify({
            "project": "AeroCast IoT Bridge",
            "status": "online",
            "ml_loaded": predictor is not None,
            "endpoints": ["/health", "/sensor-data (POST)", "/latest", "/history",
                          "/predictions/latest", "/model-info", "/dashboard"],
        })

    @app.get("/health")
    def health():
        return jsonify({
            "status": "healthy",
            "ml_loaded": predictor is not None,
            "ml_error": ml_error,
            "readings_in_memory": len(history),
            "readings_received_since_start": state["received"],
            "started_at": state["started_at"],
        }), 200

    @app.post("/sensor-data")
    def receive_sensor_data():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON"}), 400

        required = ["temperature", "humidity", "mq2_raw"]
        if any(key not in data for key in required):
            return jsonify({"error": "Missing sensor fields", "required": required}), 400

        present = [k for k in POLLUTANT_FIELDS if k in data]
        if present and len(present) != len(POLLUTANT_FIELDS):
            missing = [k for k in POLLUTANT_FIELDS if k not in data]
            return jsonify({"error": "Incomplete pollutant fields", "missing": missing}), 400

        try:
            reading = {
                "device_id": str(data.get("device_id", "AEROCAST_ESP32_01"))[:64],
                "temperature": _num(data, "temperature", -40, 80),
                "humidity": _num(data, "humidity", 0, 100),
                "mq2_raw": int(_num(data, "mq2_raw", 0, 4095)),
                "mq2_relative_percent": (
                    _num(data, "mq2_relative_percent", 0, 100) if "mq2_relative_percent" in data else 0.0
                ),
                "received_at": datetime.now(timezone.utc).isoformat(),
                "data_source": "wokwi_simulation",
            }
            pollutants = None
            if present:
                pollutants = {name: _num(data, key, 0, 10000) for key, name in POLLUTANT_FIELDS.items()}
                reading["pollutants"] = pollutants
                if "aqi" in data:
                    reading["device_pm25_aqi"] = _num(data, "aqi", 0, 1000)
        except (ValueError, TypeError) as exc:
            return jsonify({"error": "Sensor values must be numeric and in range", "detail": str(exc)}), 400

        prediction = None
        with lock:
            if pollutants is not None:
                buf = steps.setdefault(reading["device_id"], deque(maxlen=STEP_BUFFER_SIZE))
                buf.append(pollutants)
                snapshot = list(buf)
            state["received"] += 1

        if pollutants is not None:
            if predictor is None:
                prediction = {"error": "ML models not loaded", "detail": ml_error}
            else:
                try:
                    prediction = predictor.predict(snapshot)
                except Exception as exc:
                    log.exception("prediction failed")
                    prediction = {"error": f"prediction failed: {type(exc).__name__}: {exc}"}
            reading["prediction"] = prediction

        with lock:
            history.appendleft(reading)
        print("Received sensor data:", {k: v for k, v in reading.items() if k != "prediction"}, flush=True)

        if request.args.get("compact") == "1":  # small body for the ESP32
            return jsonify({"success": True, "received_at": reading["received_at"],
                            "display": display_summary(prediction)}), 200

        return jsonify({
            "success": True,
            "message": "Sensor data received",
            "data": {k: v for k, v in reading.items() if k != "prediction"},
            "prediction": prediction,
            "display": display_summary(prediction),
        }), 200

    @app.get("/latest")
    def latest():
        with lock:
            if not history:
                return jsonify({"message": "No sensor data received yet"}), 404
            return jsonify(history[0]), 200

    @app.get("/history")
    def get_history():
        try:
            limit = max(1, min(int(request.args.get("limit", HISTORY_SIZE)), HISTORY_SIZE))
        except ValueError:
            return jsonify({"error": "limit must be an integer"}), 400
        include_pred = request.args.get("predictions", "1") != "0"
        with lock:
            items = list(history)[:limit]
        if not include_pred:
            items = [{k: v for k, v in r.items() if k != "prediction"} for r in items]
        return jsonify(items), 200

    @app.get("/predictions/latest")
    def latest_prediction():
        with lock:
            for r in history:
                if r.get("prediction"):
                    return jsonify({"reading_received_at": r["received_at"], **r["prediction"]}), 200
        return jsonify({"message": "No prediction yet (send a reading that includes the 7 pollutant fields)"}), 404

    @app.get("/model-info")
    def model_info():
        if predictor is None:
            return jsonify({"ml_loaded": False, "ml_error": ml_error}), 503
        return jsonify({"ml_loaded": True, **predictor.model_info()}), 200

    @app.get("/dashboard")
    def dashboard():
        return send_from_directory(BASE_DIR / "dashboard", "index.html")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
