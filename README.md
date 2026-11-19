# AeroCast IoT Bridge

Cloud receiver for simulated ESP32 sensor data.

## Sensors
- DHT22 Temperature
- DHT22 Humidity
- MQ-2 Gas Sensor

## Endpoints
- GET / - Server status
- GET /health - Health check
- POST /sensor-data - Receive IoT readings
- GET /latest - Latest reading
- GET /history - Recent readings

## Deployment
Hosted on Render.