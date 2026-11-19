# AeroCast IoT Bridge

Cloud bridge between the **Wokwi ESP32 simulation** and the **AeroCast ML models**
(trained in [aerocast-iot-air-quality-forecasting](https://github.com/Anjanikr137/aerocast-iot-air-quality-forecasting)).

```
Wokwi ESP32 (DHT22, MQ-2, 7 pots via CD74HC4067, OLED)
   │  HTTPS POST /sensor-data?compact=1   (every 10 s, Wokwi-GUEST Wi-Fi)
   ▼
Render: Flask bridge ── stores last 100 readings (in memory)
   │                 └─ builds 119 model features from the simulated pollutant stream
   │                    ├─ XGBoost            → next-step AQI
   │                    └─ Linear Regression  → next-step PM2.5, PM10, NO2, SO2, CO, O3, NH3
   ├─► response {display: aqi_now, aqi_next, pm25_next, status…} → ESP32 OLED page 2 + Serial
   └─► /dashboard (charts, tiles, tables)
```

**All sensor and pollutant values are simulated in Wokwi. Nothing here measures real air.**

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Status and endpoint list |
| GET | `/health` | Health, `ml_loaded`, reading counters |
| POST | `/sensor-data` | Receive a reading; returns stored reading, full `prediction`, compact `display` |
| POST | `/sensor-data?compact=1` | Same, but returns only `{success, received_at, display}` (used by the ESP32) |
| GET | `/latest` | Newest reading (with its prediction) |
| GET | `/history?limit=N&predictions=0` | Up to 100 recent readings, newest first |
| GET | `/predictions/latest` | Newest full prediction |
| GET | `/model-info` | Models used, saved test metrics, training ranges |
| GET | `/dashboard` | Live dashboard |

### Payload (what the ESP32 sends)

```json
{
  "device_id": "AEROCAST_ESP32_01",
  "temperature": 37.9, "humidity": 67.5,
  "mq2_raw": 1200, "mq2_relative_percent": 29.3,
  "pm25": 120.0, "pm10": 250.0, "no2": 60.0, "so2": 15.0,
  "co": 1.5, "o3": 40.0, "nh3": 50.0,
  "aqi": 300
}
```

`temperature`, `humidity` and `mq2_raw` are required (unchanged from v1). The seven pollutant fields
are optional, but if any is present all seven must be. `aqi` is the device's PM2.5-only sub-index and is
stored as `device_pm25_aqi`. Invalid JSON, missing or partial fields, non-numeric, NaN or out-of-range
values return HTTP 400.

## How the ML integration works (and its limits)

The models were trained on hourly CPCB data from Shadipur, Delhi (site_113, 2023–2024). Their 119
inputs are current pollutant concentrations, CPCB AQI, lags of 1–24 h, rolling means/stds and
time-of-day/month features. **Temperature, humidity and MQ-2 are not model inputs**, and they are not fed
into the models.

The Wokwi circuit has seven potentiometers that simulate the seven pollutant concentrations, so the bridge
can rebuild the training features from them:

* **Time-compressed:** each received reading counts as one hourly step. At 10 s uploads, 24 "hours" take
  about 4 minutes. Hour/day/month features use the server's current IST wall-clock time.
* **Warm-up:** full context needs 48 steps (AQI_lag24 needs 24 steps of AQI history, which itself needs
  24 steps of averaging). Before that, lags fall back to the oldest reading and `status` is `warming_up`.
* **Imputed:** Benzene, Eth-Benzene, MP-Xylene, NO, NOx and Toluene are not simulated, so they are filled
  with the training median imputer. On real test-set windows this moved the AQI forecast by 0–2 points.
* **Out-of-range flag:** the pot full scales exceed the training maxima for SO2 (500 vs 180), O3 (400 vs 200)
  and NH3 (1000 vs 396). Such inputs are flagged in `out_of_training_range` and as `ood` on the device.
* **Current AQI** (`current_aqi_estimate`) is computed in the bridge with the CPCB method (max sub-index of
  rolling means) over the simulated readings. It is an estimate, and provisional before 24 steps.
* **Parity test:** `tests/test_app.py::test_feature_parity_with_training_data` feeds real 48-hour windows
  from the ML repo's `test.csv` through the live feature builder and checks that the model output equals the
  output for the training pipeline's feature row.

The saved test metrics (`/model-info`) describe performance on held-out CPCB data. They say nothing about
accuracy on simulated potentiometer values.

## Run locally

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
python app.py                     # http://127.0.0.1:5000/dashboard
python scripts/simulate_device.py --count 60 --interval 0.2   # synthetic ESP32 traffic
pytest -q
```

Refreshing the models after retraining in the ML repo:

```bash
python scripts/export_models.py ../aerocast-iot-air-quality-forecasting
```

## Wokwi

`firmware/aerocast_esp32/` holds `sketch.ino`, `diagram.json` and `libraries.txt`. Paste `sketch.ino` over
the sketch in the Wokwi project and add **ArduinoJson** in the Library Manager. The circuit is unchanged.

## Deployment (Render)

`render.yaml` runs `gunicorn app:app --workers 1 --threads 4 --timeout 120`. Keep **one worker**, because
readings and the model's step buffer live in process memory. On the free plan the service sleeps after
about 15 minutes idle. The first request then takes roughly 20–60 s, and the history is lost.
