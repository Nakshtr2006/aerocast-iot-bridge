from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timezone
from collections import deque

app = Flask(__name__)
CORS(app)

history = deque(maxlen=100)


@app.get("/")
def home():
    return jsonify({
        "project": "AeroCast IoT Bridge",
        "status": "online"
    })


@app.get("/health")
def health():
    return jsonify({"status": "healthy"}), 200


@app.post("/sensor-data")
def receive_sensor_data():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON"}), 400

    required = ["temperature", "humidity", "mq2_raw"]

    if any(key not in data for key in required):
        return jsonify({"error": "Missing sensor fields"}), 400

    try:
        reading = {
            "device_id": str(data.get("device_id", "AEROCAST_ESP32_01")),
            "temperature": float(data["temperature"]),
            "humidity": float(data["humidity"]),
            "mq2_raw": int(data["mq2_raw"]),
            "mq2_relative_percent": float(
                data.get("mq2_relative_percent", 0)
            ),
            "received_at": datetime.now(timezone.utc).isoformat()
        }
    except (ValueError, TypeError):
        return jsonify({"error": "Sensor values must be numeric"}), 400

    history.appendleft(reading)
    print("Received sensor data:", reading, flush=True)

    return jsonify({
        "success": True,
        "message": "Sensor data received",
        "data": reading
    }), 200


@app.get("/latest")
def latest():
    if not history:
        return jsonify({"message": "No sensor data received yet"}), 404

    return jsonify(history[0]), 200


@app.get("/history")
def get_history():
    return jsonify(list(history)), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)