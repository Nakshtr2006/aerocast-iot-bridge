// AeroCast ESP32 firmware (Wokwi simulation)
//
// Based on the existing Wokwi sketch. Circuit, GPIO assignments, sensor code and OLED
// layout are unchanged. Additions are marked [AeroCast ML]:
//   * the bridge's ML response is parsed (ArduinoJson) and shown on a 2nd OLED page
//   * HTTP timeouts raised so a Render cold start (~20-30 s) doesn't fail every upload
//   * calculateAQI() no longer returns 0 for values between CPCB breakpoints (e.g. 30.5)
//
// ALL readings are SIMULATED (Wokwi DHT22, MQ-2 and potentiometers via CD74HC4067).

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <DHT.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ArduinoJson.h>  // [AeroCast ML] add "ArduinoJson" in Wokwi Library Manager

// ============== HARDWARE PINS ==============
#define DHT_PIN 15
#define DHT_TYPE DHT22
#define MQ2_PIN 34

// CD74HC4067 MUX
#define MUX_S0 18
#define MUX_S1 19
#define MUX_S2 23
#define MUX_S3 25
#define MUX_SIG 35

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_ADDRESS 0x3C

// ============== WIFI ==============
const char* WIFI_SSID = "Wokwi-GUEST";
const char* WIFI_PASSWORD = "";

// ============== RENDER URL ==============
// [AeroCast ML] compact=1 -> bridge returns only the small "display" object
const char* SERVER_URL =
  "https://aerocast-iot-bridge.onrender.com/sensor-data?compact=1";

// ============== SENSOR OBJECTS ==============
DHT dht(DHT_PIN, DHT_TYPE);

Adafruit_SSD1306 display(
  SCREEN_WIDTH,
  SCREEN_HEIGHT,
  &Wire,
  -1
);

// ============== TIMING ==============
const unsigned long SENSOR_INTERVAL = 5000;
const unsigned long UPLOAD_INTERVAL = 10000;
const unsigned long WIFI_RETRY_INTERVAL = 10000;
const unsigned long OLED_PAGE_INTERVAL = 4000;   // [AeroCast ML]
const unsigned long HTTP_TIMEOUT_MS = 30000;     // [AeroCast ML] was 10000

// Retry delays: 2, 4, 8 seconds
const unsigned long RETRY_DELAYS[] = {
  2000, 4000, 8000
};

const int MAX_RETRIES = 3;

unsigned long lastSensorRead = 0;
unsigned long lastUploadSchedule = 0;
unsigned long lastWiFiAttempt = 0;
unsigned long nextRetryTime = 0;
unsigned long lastPageSwitch = 0;  // [AeroCast ML]
uint8_t oledPage = 0;              // [AeroCast ML] 0 = sensors, 1 = forecast

// ============== SENSOR VALUES ==============
float temperature = 0;
float humidity = 0;
int mq2Raw = 0;

int muxRaw[7] = {0};

float pm25 = 0;
float pm10 = 0;
float no2 = 0;
float so2 = 0;
float co = 0;
float o3 = 0;
float nh3 = 0;

int aqi = 0;

bool sensorDataValid = false;

// ============== UPLOAD TRACKING ==============
String pendingJson = "";

bool uploadPending = false;
int retryCount = 0;

unsigned long uploadSuccessCount = 0;
unsigned long uploadFailureCount = 0;

bool lastUploadSuccess = false;
bool hasUploadResult = false;

// ============== [AeroCast ML] SERVER FORECAST ==============
struct Forecast {
  bool valid = false;
  String status = "none";   // ok | warming_up | unavailable | none
  int aqiNow = -1;          // bridge CPCB estimate from simulated readings
  String catNow = "";
  int aqiNext = -1;         // XGBoost next-step forecast
  String catNext = "";
  float pm25Next = NAN;     // Linear Regression next-step forecast
  int steps = 0;
  int stepsFull = 48;
  int ood = 0;              // inputs outside training range
  unsigned long receivedAt = 0;
} forecast;

// ============== WIFI ==============
void startWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD, 6);

  lastWiFiAttempt = millis();

  Serial.println("WiFi connection initiated...");
}

void maintainWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  if (millis() - lastWiFiAttempt >= WIFI_RETRY_INTERVAL) {
    Serial.println("WiFi disconnected. Reconnecting...");

    WiFi.disconnect();
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD, 6);

    lastWiFiAttempt = millis();
  }
}

// ============== MUX SELECT ==============
void selectMuxChannel(uint8_t channel) {
  digitalWrite(MUX_S0, channel & 0x01);
  digitalWrite(MUX_S1, (channel >> 1) & 0x01);
  digitalWrite(MUX_S2, (channel >> 2) & 0x01);
  digitalWrite(MUX_S3, (channel >> 3) & 0x01);

  delayMicroseconds(10);
}

// ============== READ MUX ==============
int readMuxChannel(uint8_t channel) {
  selectMuxChannel(channel);

  // Allow signal to settle after channel switching
  delayMicroseconds(50);

  // Average 5 ADC samples
  long total = 0;

  for (int i = 0; i < 5; i++) {
    total += analogRead(MUX_SIG);
    delayMicroseconds(200);
  }

  return total / 5;
}

// ============== MAP POT VALUES ==============
// Convert ADC (0-4095) into simulated pollutant ranges

float scalePot(int raw, float maxValue) {
  return (raw / 4095.0) * maxValue;
}

// ============== CPCB PM2.5 AQI ==============
// PM2.5-only sub-index computed on the device from the instantaneous pot value.
// The bridge computes the full multi-pollutant CPCB AQI over rolling averages.
// [AeroCast ML] fixed: contiguous bands, so e.g. 30.5 ug/m3 no longer returns 0.
int calculateAQI(float concentration) {
  // CPCB India PM2.5 breakpoints (upper bound of each band)
  const float bpHigh[] = {30, 60, 90, 120, 250, 500};
  const int aqiHigh[] = {50, 100, 200, 300, 400, 500};

  if (concentration <= 0) return 0;
  if (concentration >= 500) return 500;

  float bpLow = 0;
  int aqiLow = 0;
  for (int i = 0; i < 6; i++) {
    if (concentration <= bpHigh[i]) {
      float result = aqiLow +
        (aqiHigh[i] - aqiLow) * (concentration - bpLow) / (bpHigh[i] - bpLow);
      return round(result);
    }
    bpLow = bpHigh[i];
    aqiLow = aqiHigh[i];
  }

  return 500;
}

// ============== READ POLLUTANT POTS ==============
void readPollutantPots() {
  for (int i = 0; i < 7; i++) {
    muxRaw[i] = readMuxChannel(i);
  }

  // Convert potentiometer readings to simulated values
  pm25 = scalePot(muxRaw[0], 250.0);  // ug/m3
  pm10 = scalePot(muxRaw[1], 600.0);  // ug/m3
  no2  = scalePot(muxRaw[2], 400.0); // ug/m3
  so2  = scalePot(muxRaw[3], 500.0); // ug/m3
  co   = scalePot(muxRaw[4], 10.0);   // mg/m3
  o3   = scalePot(muxRaw[5], 400.0); // ug/m3
  nh3  = scalePot(muxRaw[6], 1000.0); // ug/m3

  aqi = calculateAQI(pm25);

  Serial.println("\n------ SIMULATED POLLUTANTS ------");

  Serial.print("PM2.5: ");
  Serial.print(pm25, 1);
  Serial.println(" ug/m3");

  Serial.print("PM10: ");
  Serial.print(pm10, 1);
  Serial.println(" ug/m3");

  Serial.print("NO2: ");
  Serial.print(no2, 1);
  Serial.println(" ug/m3");

  Serial.print("SO2: ");
  Serial.print(so2, 1);
  Serial.println(" ug/m3");

  Serial.print("CO: ");
  Serial.print(co, 2);
  Serial.println(" mg/m3");

  Serial.print("O3: ");
  Serial.print(o3, 1);
  Serial.println(" ug/m3");

  Serial.print("NH3: ");
  Serial.print(nh3, 1);
  Serial.println(" ug/m3");

  Serial.print("Device PM2.5 sub-index: ");
  Serial.println(aqi);
}

// ============== OLED ==============
// Page 0: original sensor screen (unchanged layout)
void drawSensorPage() {
  display.setCursor(0, 0);
  display.println("AeroCast IoT Monitor");

  display.drawLine(0, 10, 127, 10, SSD1306_WHITE);

  display.setCursor(0, 14);
  display.print("Temp: ");
  display.print(temperature, 1);
  display.println(" C");

  display.setCursor(0, 25);
  display.print("Humidity: ");
  display.print(humidity, 1);
  display.println("%");

  display.setCursor(0, 36);
  display.print("MQ2 Raw: ");
  display.println(mq2Raw);

  display.setCursor(0, 47);
  display.print("WiFi: ");
  display.println(
    WiFi.status() == WL_CONNECTED ? "Connected" : "Offline"
  );

  display.setCursor(0, 56);
  display.print("Cloud OK:");
  display.print(uploadSuccessCount);
  display.print(" Fail:");
  display.print(uploadFailureCount);
}

// [AeroCast ML] Page 1: bridge AQI estimate + model forecast
void drawForecastPage() {
  display.setCursor(0, 0);
  display.println("AeroCast Forecast");
  display.drawLine(0, 10, 127, 10, SSD1306_WHITE);

  if (!forecast.valid) {
    display.setCursor(0, 16);
    display.println(WiFi.status() == WL_CONNECTED ? "Waiting for server" : "WiFi offline");
    display.setCursor(0, 28);
    display.print("Dev PM2.5 idx: ");
    display.println(aqi);
    display.setCursor(0, 40);
    display.print("ML: ");
    display.println(forecast.status);
    return;
  }

  display.setCursor(0, 14);
  display.print("AQI now : ");
  display.print(forecast.aqiNow);
  display.print(" ");
  display.println(forecast.catNow.substring(0, 6));

  display.setCursor(0, 25);
  display.print("AQI next: ");
  display.print(forecast.aqiNext);
  display.print(" ");
  display.println(forecast.catNext.substring(0, 6));

  display.setCursor(0, 36);
  display.print("PM2.5 nxt: ");
  display.print(forecast.pm25Next, 1);

  display.setCursor(0, 47);
  display.print(forecast.status == "ok" ? "ML ok " : "ML warm ");
  display.print(forecast.steps > forecast.stepsFull ? forecast.stepsFull : forecast.steps);
  display.print("/");
  display.print(forecast.stepsFull);
  if (forecast.ood > 0) display.print(" OOD");

  display.setCursor(0, 56);
  display.print("Updated ");
  display.print((millis() - forecast.receivedAt) / 1000);
  display.print("s ago");
}

void updateOLED() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);

  if (oledPage == 0) {
    drawSensorPage();
  } else {
    drawForecastPage();
  }

  display.display();
}

// ============== READ SENSORS ==============
void readSensors() {
  float newTemperature = dht.readTemperature();
  float newHumidity = dht.readHumidity();
  int newMQ2 = analogRead(MQ2_PIN);

  if (isnan(newTemperature) || isnan(newHumidity)) {
    Serial.println("DHT22 reading failed. Keeping previous values.");
    return;
  }

  temperature = newTemperature;
  humidity = newHumidity;
  mq2Raw = newMQ2;

  // Read all 7 potentiometer channels
  readPollutantPots();

  sensorDataValid = true;

  Serial.println("\n------ SIMULATED SENSOR DATA (Wokwi) ------");

  Serial.print("Temperature: ");
  Serial.println(temperature);

  Serial.print("Humidity: ");
  Serial.println(humidity);

  Serial.print("MQ2 Raw: ");
  Serial.println(mq2Raw);

  updateOLED();
}

// ============== BUILD JSON ==============
String buildJSON() {
  float mq2Percent = (mq2Raw / 4095.0) * 100.0;

  String json = "{";

  json += "\"device_id\":\"AEROCAST_ESP32_01\",";
  json += "\"temperature\":" + String(temperature, 2) + ",";
  json += "\"humidity\":" + String(humidity, 2) + ",";
  json += "\"mq2_raw\":" + String(mq2Raw) + ",";
  json += "\"mq2_relative_percent\":" + String(mq2Percent, 2) + ",";

  json += "\"pm25\":" + String(pm25, 2) + ",";
  json += "\"pm10\":" + String(pm10, 2) + ",";
  json += "\"no2\":" + String(no2, 2) + ",";
  json += "\"so2\":" + String(so2, 2) + ",";
  json += "\"co\":" + String(co, 2) + ",";
  json += "\"o3\":" + String(o3, 2) + ",";
  json += "\"nh3\":" + String(nh3, 2) + ",";
  json += "\"aqi\":" + String(aqi);

  json += "}";

  return json;
}

// ============== [AeroCast ML] PARSE SERVER RESPONSE ==============
void parseServerResponse(const String& body) {
  JsonDocument doc;
  DeserializationError err = deserializeJson(doc, body);
  if (err) {
    Serial.print("Response JSON parse failed: ");
    Serial.println(err.c_str());
    return;
  }

  JsonObject d = doc["display"];
  if (d.isNull()) {
    Serial.println("Response has no 'display' object (old bridge version?).");
    return;
  }

  forecast.status = d["status"] | "unavailable";
  if (forecast.status == "unavailable") {
    forecast.valid = false;
    Serial.println("Server: ML forecast unavailable.");
    return;
  }

  forecast.aqiNow = d["aqi_now"] | -1;
  forecast.catNow = d["cat_now"] | "";
  forecast.aqiNext = d["aqi_next"] | -1;
  forecast.catNext = d["cat_next"] | "";
  forecast.pm25Next = d["pm25_next"] | NAN;
  forecast.steps = d["steps"] | 0;
  forecast.stepsFull = d["steps_full"] | 48;
  forecast.ood = d["ood"] | 0;
  forecast.receivedAt = millis();
  forecast.valid = true;

  Serial.println("------ SERVER FORECAST (models on simulated inputs) ------");
  Serial.printf("AQI estimate now : %d (%s)\n", forecast.aqiNow, forecast.catNow.c_str());
  Serial.printf("AQI next step    : %d (%s)  [XGBoost]\n", forecast.aqiNext, forecast.catNext.c_str());
  Serial.printf("PM2.5 next step  : %.1f ug/m3  [Linear Regression]\n", forecast.pm25Next);
  Serial.printf("Model context    : %s, %d/%d steps%s\n", forecast.status.c_str(),
                forecast.steps, forecast.stepsFull,
                forecast.ood ? ", some inputs outside training range" : "");
}

// ============== ATTEMPT UPLOAD ==============
bool uploadData() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Upload postponed: WiFi offline.");
    return false;
  }

  WiFiClientSecure secureClient;
  secureClient.setInsecure();
  secureClient.setTimeout(HTTP_TIMEOUT_MS / 1000);

  HTTPClient http;
  http.setConnectTimeout(HTTP_TIMEOUT_MS);
  http.setTimeout(HTTP_TIMEOUT_MS);

  Serial.println("\nSending data to Render:");
  Serial.println(pendingJson);

  if (!http.begin(secureClient, SERVER_URL)) {
    Serial.println("HTTP initialization failed.");
    http.end();
    secureClient.stop();
    return false;
  }

  http.addHeader("Content-Type", "application/json");
  http.addHeader("Connection", "close");

  int responseCode = http.POST(pendingJson);

  Serial.print("HTTP Response Code: ");
  Serial.println(responseCode);

  if (responseCode >= 200 && responseCode < 300) {
    Serial.println("UPLOAD SUCCESS!");

    String body = http.getString();
    Serial.println("Server response:");
    Serial.println(body);
    parseServerResponse(body);  // [AeroCast ML]

    http.end();
    secureClient.stop();

    uploadSuccessCount++;
    lastUploadSuccess = true;
    hasUploadResult = true;

    return true;
  }

  if (responseCode > 0) {
    Serial.println("HTTP server error:");
    Serial.println(http.getString());
  } else {
    Serial.print("Connection error: ");
    Serial.println(HTTPClient::errorToString(responseCode));
  }

  http.end();
  secureClient.stop();

  return false;
}

// ============== SCHEDULE UPLOAD ==============
void scheduleUpload() {
  if (!sensorDataValid) {
    return;
  }

  // Don't replace a reading that is currently waiting for retries
  if (uploadPending) {
    return;
  }

  pendingJson = buildJSON();
  uploadPending = true;
  retryCount = 0;
  nextRetryTime = millis();

  Serial.println("\nCloud upload scheduled.");
}

// ============== HANDLE UPLOAD / RETRIES ==============
void handleUpload() {
  if (!uploadPending) {
    return;
  }

  if ((long)(millis() - nextRetryTime) < 0) {
    return;
  }

  Serial.printf("\nUpload attempt %d of %d\n",
               retryCount + 1, MAX_RETRIES + 1);

  bool success = uploadData();

  if (success) {
    uploadPending = false;
    retryCount = 0;
    updateOLED();
    return;
  }

  // Retry delays: 2 sec, 4 sec, 8 sec
  if (retryCount < MAX_RETRIES) {
    unsigned long delayTime = RETRY_DELAYS[retryCount];

    Serial.print("Retry scheduled in ");
    Serial.print(delayTime / 1000);
    Serial.println(" seconds.");

    nextRetryTime = millis() + delayTime;
    retryCount++;
  } else {
    Serial.println("All retry attempts exhausted. Reading will be dropped.");

    uploadFailureCount++;
    lastUploadSuccess = false;
    hasUploadResult = true;

    uploadPending = false;
    retryCount = 0;

    updateOLED();
  }
}

// ============== SETUP ==============
void setup() {
  Serial.begin(115200);

  dht.begin();
  analogReadResolution(12);

  // Configure multiplexer selector pins
  pinMode(MUX_S0, OUTPUT);
  pinMode(MUX_S1, OUTPUT);
  pinMode(MUX_S2, OUTPUT);
  pinMode(MUX_S3, OUTPUT);

  pinMode(MUX_SIG, INPUT);

  // Ensure mux starts at channel 0
  selectMuxChannel(0);

  Wire.begin(21, 22);

  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS)) {
    Serial.println("OLED initialization failed!");
    while (true) {
      delay(1000);
    }
  }

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("AeroCast Starting...");
  display.display();

  startWiFi();

  Serial.println("AeroCast IoT System Started.");
  Serial.println("7 pollutant simulation inputs initialized.");
}

// ============== MAIN LOOP ==============
void loop() {
  unsigned long now = millis();

  // Maintain WiFi without blocking the loop
  maintainWiFi();

  // Read sensors every 5 seconds
  if (now - lastSensorRead >= SENSOR_INTERVAL) {
    lastSensorRead = now;
    readSensors();
  }

  // Schedule cloud upload every 10 seconds
  if (now - lastUploadSchedule >= UPLOAD_INTERVAL) {
    lastUploadSchedule = now;
    scheduleUpload();
  }

  // [AeroCast ML] Alternate OLED between sensor and forecast pages
  if (now - lastPageSwitch >= OLED_PAGE_INTERVAL) {
    lastPageSwitch = now;
    oledPage = (oledPage + 1) % 2;
    updateOLED();
  }

  // Handle uploads and retries
  handleUpload();
}
