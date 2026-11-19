"""
Send ESP32-shaped payloads to the bridge without Wokwi (for testing the backend).

The payload schema is identical to firmware/aerocast_esp32/sketch.ino. Values are
SYNTHETIC: a slow random walk around user-chosen pollutant levels.

Usage:
    python scripts/simulate_device.py --url http://127.0.0.1:5000 --count 50 --interval 0.2
    python scripts/simulate_device.py --url https://aerocast-iot-bridge.onrender.com --count 5 --interval 10
"""

import argparse
import random
import time

import requests

BASE = {"pm25": 120.0, "pm10": 230.0, "no2": 55.0, "so2": 12.0, "co": 1.4, "o3": 35.0, "nh3": 45.0}
LIMITS = {"pm25": 250, "pm10": 600, "no2": 400, "so2": 500, "co": 10, "o3": 400, "nh3": 1000}  # pot full-scale


def pm25_aqi(c):  # same table the firmware uses (CPCB PM2.5 sub-index)
    bp = [(0, 30, 0, 50), (30, 60, 50, 100), (60, 90, 100, 200), (90, 120, 200, 300), (120, 250, 300, 400), (250, 500, 400, 500)]
    for lo, hi, ilo, ihi in bp:
        if c <= hi:
            return round(ilo + (ihi - ilo) * (c - lo) / (hi - lo))
    return 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5000")
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--interval", type=float, default=0.2)
    ap.add_argument("--device-id", default="SIM_SCRIPT_01")
    args = ap.parse_args()

    state = dict(BASE)
    for i in range(args.count):
        for k in state:
            state[k] = min(max(state[k] * random.uniform(0.95, 1.05), 0.0), LIMITS[k])
        mq2 = random.randint(900, 1600)
        payload = {
            "device_id": args.device_id,
            "temperature": round(random.uniform(24, 32), 2),
            "humidity": round(random.uniform(40, 70), 2),
            "mq2_raw": mq2,
            "mq2_relative_percent": round(mq2 / 4095 * 100, 2),
            **{k: round(v, 2) for k, v in state.items()},
            "aqi": pm25_aqi(state["pm25"]),
        }
        t0 = time.time()
        r = requests.post(args.url.rstrip("/") + "/sensor-data", json=payload, timeout=60)
        d = r.json().get("display", {}) if r.ok else r.text
        print(f"[{i + 1}/{args.count}] HTTP {r.status_code} {time.time() - t0:.2f}s  display={d}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
