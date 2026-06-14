// ===================== ESP32 (FULL) =====================
// 功能改动：
// 1) 识别 Mega->ESP32 的 S,0/ / S,1/：
//    - S,0: 停止推理（忽略 I/R），并 BLE 通知手机 "S,0\n"
//    - S,1: 恢复推理，并 BLE 通知手机 "S,1\n"（并清空窗口状态）
// 2) 手机->ESP32->Mega 的命令（如 CAL/PING）统一用 '/' 作为终止符（发送 CAL/）
// 3) UART 分帧终止符统一 '/'

#include <Arduino.h>
#include <ArduinoBLE.h>
#include <math.h>
#include <string.h>

// ====== SVM 模型头文件（AML_SVM.py 导出的 AML_SVM.h）======
#include "AML_SVM.h"

// ============================================================
// 可调全局参数
// ============================================================
constexpr bool SERIAL_DEBUG = true;

constexpr int WINDOW_LEN   = 10;
constexpr int NUM_POSTURES = AML_NUM_CLASSES;

constexpr uint32_t INFERENCE_STEP_MS = 500;
constexpr bool BLE_SEND_ONLY_ON_CHANGE = false;

// UART (Mega -> ESP32)
constexpr uint32_t UART_BAUD   = 115200;
constexpr int      UART_RX_PIN = D0;
constexpr int      UART_TX_PIN = D1;

// BLE NUS
#define DEV_NAME "AML_DEVICE_IOT_FINAL"
static const char* NUS_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E";
static const char* NUS_TX_UUID      = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E";
static const char* NUS_RX_UUID      = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E";

constexpr int BLE_MAX_CHARS = 240;

// posture -> code (aligned to AML_CLASS_NAMES order)
constexpr uint8_t CODE_BACK       = 1;
constexpr uint8_t CODE_FORWARD    = 2;
constexpr uint8_t CODE_LEFT       = 3;
constexpr uint8_t CODE_LEFTCROSS  = 4;   // "Leftcross"
constexpr uint8_t CODE_LYINGDOWN  = 5;   // "Lyingdown"
constexpr uint8_t CODE_MIDDLE     = 6;
constexpr uint8_t CODE_RIGHT      = 7;
constexpr uint8_t CODE_RIGHTCROSS = 8;   // "Rightcross"
constexpr uint8_t CODE_SITFRONT   = 9;   // "Sitfront"
constexpr uint8_t CODE_STRAIGHT   = 10;


// UART frame: use '/' as terminator
constexpr char UART_FRAME_TERM = '/';
constexpr size_t UART_FRAME_MAX = 384;

// ============================================================
// BLE service/characteristics
// ============================================================
BLEService nus(NUS_SERVICE_UUID);

BLECharacteristic txChar(
  NUS_TX_UUID,
  BLERead | BLENotify,
  BLE_MAX_CHARS
);

BLEStringCharacteristic rxChar(
  NUS_RX_UUID,
  BLEWrite | BLEWriteWithoutResponse,
  BLE_MAX_CHARS
);

static void startAdvertising() {
  BLE.setAdvertisedService(nus);
  BLE.advertise();
}

// ============================================================
// Debug: cache + print last BLE TX payload
// ============================================================
static char lastBleTx[BLE_MAX_CHARS + 1] = {0};  // 最后一次准备发送到 BLE 的内容(截断到 BLE_MAX_CHARS)
static uint32_t lastBleTxMs = 0;

static inline void cacheAndPrintBleTx(const char* s, size_t len, bool subscribed) {
  // 缓存（最多 BLE_MAX_CHARS）
  size_t n = len;
  if (n > (size_t)BLE_MAX_CHARS) n = (size_t)BLE_MAX_CHARS;
  memcpy(lastBleTx, s, n);
  lastBleTx[n] = '\0';
  lastBleTxMs = millis();

  if (SERIAL_DEBUG) {
    Serial.print(F("[BLE TX LAST] ms="));
    Serial.print(lastBleTxMs);
    Serial.print(F(" subscribed="));
    Serial.print(subscribed ? 1 : 0);
    Serial.print(F(" len="));
    Serial.print((unsigned)len);
    Serial.print(F(" payload=\""));
    Serial.print(lastBleTx);  // 注意：里面如果有 '\n'，串口会换行，这是正常的
    Serial.println(F("\""));
  }
}

// 单次 notify（不分片）
// 单次 notify（不分片）
static bool bleNotifyRaw(const char* s) {
  if (!BLE.connected()) return false;

  const bool sub = txChar.subscribed();
  if (SERIAL_DEBUG) {
    Serial.print(F("[BLE TX] subscribed="));
    Serial.println(sub ? 1 : 0);
  }

  size_t len = strlen(s);
  if (len == 0) return true;

  if (len > (size_t)BLE_MAX_CHARS) {
    if (SERIAL_DEBUG) Serial.println(F("[BLE TX] too long"));
    // 这里也打印一下被拒绝的内容（会被截断缓存）
    cacheAndPrintBleTx(s, len, sub);
    return false;
  }

  // ✅ 在真正发送前，把“最后一次准备发送到 BLE 的内容”打印到串口
  cacheAndPrintBleTx(s, len, sub);

  bool ok = txChar.writeValue((const unsigned char*)s, (int)len);

  if (SERIAL_DEBUG) {
    Serial.print(F("[BLE TX] writeValue bytes="));
    Serial.print((unsigned)len);
    Serial.print(F(", ok="));
    Serial.println(ok ? 1 : 0);
  }
  return ok;
}

// ============================================================
// UART cmd to Mega: terminate with '/'
// ============================================================
static inline void uartSendCmdFramed(const char* cmd) {
  Serial1.print(cmd);
  Serial1.print(UART_FRAME_TERM); // '/'
  Serial1.print('\n');            // 可选：Mega 会忽略 '\n'
}

static inline void uartSendCmdFramed(const String& cmd) {
  Serial1.print(cmd);
  Serial1.print(UART_FRAME_TERM); // '/'
  Serial1.print('\n');
}


// ============================================================
// UART frame buffer (terminated by '/')
// ============================================================
static char   uartFrame[UART_FRAME_MAX];
static size_t uartFrameLen = 0;
static bool   uartFrameOverflow = false;

static inline void resetUartFrame() {
  uartFrameLen = 0;
  uartFrameOverflow = false;
}

// ============================================================
// I/R cache
// ============================================================
static char lastILine[UART_FRAME_MAX];
static bool hasILine = false;

// ============================================================
// Moving window of probabilities
// ============================================================
static float probWindow[WINDOW_LEN][NUM_POSTURES];
static int   winCount = 0;
static int   winPos   = 0;

static uint32_t lastInferMs = 0;
static bool     lastConnected = false;
static uint8_t  lastSentCode  = 0;

// ============================================================
// Seat state
// ============================================================
static bool seatOccupied = true;

static void resetInferenceState() {
  hasILine = false;
  lastILine[0] = '\0';
  winCount = 0;
  winPos = 0;
  lastSentCode = 0;
  lastInferMs = 0;

  for (int i = 0; i < WINDOW_LEN; ++i) {
    for (int k = 0; k < NUM_POSTURES; ++k) probWindow[i][k] = 0.0f;
  }
}

static void bleNotifySeatState(bool occupied) {
  char msg[8];
  snprintf(msg, sizeof(msg), "S,%d\n", occupied ? 1 : 0);
  if (BLE.connected()) bleNotifyRaw(msg);
}

// ============================================================
// class-name -> code mapping
// ============================================================
struct NameCode { const char* name; uint8_t code; };

static const NameCode NAME_CODE_TABLE[] = {
  {"Back",       CODE_BACK},
  {"Forward",    CODE_FORWARD},
  {"Left",       CODE_LEFT},
  {"Leftcross",  CODE_LEFTCROSS},
  {"Lyingdown",  CODE_LYINGDOWN},
  {"Middle",     CODE_MIDDLE},
  {"Right",      CODE_RIGHT},
  {"Rightcross", CODE_RIGHTCROSS},
  {"Sitfront",   CODE_SITFRONT},
  {"Straight",   CODE_STRAIGHT},
};



static uint8_t classIndexToCode[NUM_POSTURES];

static void buildClassIndexToCodeMap() {
  for (int k = 0; k < NUM_POSTURES; ++k) classIndexToCode[k] = (uint8_t)(k + 1);

  for (int k = 0; k < NUM_POSTURES; ++k) {
    const char* cname = AML_CLASS_NAMES[k];
    for (size_t i = 0; i < sizeof(NAME_CODE_TABLE) / sizeof(NAME_CODE_TABLE[0]); ++i) {
      if (strcmp(cname, NAME_CODE_TABLE[i].name) == 0) {
        classIndexToCode[k] = NAME_CODE_TABLE[i].code;
        break;
      }
    }
  }
}

// ============================================================
// Math: softmax / median
// ============================================================
static void softmaxScoresToProb(const float* scores, float* prob, int n) {
  float maxS = scores[0];
  for (int i = 1; i < n; ++i) if (scores[i] > maxS) maxS = scores[i];

  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    float e = expf(scores[i] - maxS);
    prob[i] = e;
    sum += e;
  }

  if (sum <= 0.0f || !isfinite(sum)) {
    float u = 1.0f / (float)n;
    for (int i = 0; i < n; ++i) prob[i] = u;
    return;
  }

  float inv = 1.0f / sum;
  for (int i = 0; i < n; ++i) prob[i] *= inv;
}

static float medianInPlace(float* a, int n) {
  for (int i = 1; i < n; ++i) {
    float key = a[i];
    int j = i - 1;
    while (j >= 0 && a[j] > key) { a[j + 1] = a[j]; --j; }
    a[j + 1] = key;
  }
  if (n <= 0) return 0.0f;
  if (n & 1) return a[n / 2];
  return 0.5f * (a[(n / 2) - 1] + a[n / 2]);
}

// ============================================================
// Compute SVM scores + RobustScaler (from AML_SVM.h)
// ============================================================
static void computeScores(const float x_in[AML_FEATURES], float scores[NUM_POSTURES]) {
  float x[AML_FEATURES];
  for (int j = 0; j < AML_FEATURES; ++j) x[j] = x_in[j];

  for (int j = 0; j < AML_FEATURES; ++j) {
    x[j] = (x[j] - AML_SCALER_MEAN[j]) / AML_SCALER_SCALE[j];
  }
  x[10] *= 2.0f;
#if defined(AML_RBF_TOTAL_SV)
  float x_norm2 = 0.0f;
  for (int j = 0; j < AML_FEATURES; ++j) x_norm2 += x[j] * x[j];

  for (int k = 0; k < NUM_POSTURES; ++k) {
    float s = AML_RBF_INTERCEPT[k];
    int start = AML_RBF_SV_OFFSET[k];
    int end   = AML_RBF_SV_OFFSET[k + 1];
    for (int i = start; i < end; ++i) {
      s += AML_RBF_COEF[i] * aml_rbf_kernel(x, x_norm2, i);
    }
    scores[k] = s;
  }

#elif defined(AML_POLY_TOTAL_SV)
  for (int k = 0; k < NUM_POSTURES; ++k) {
    float s = AML_POLY_INTERCEPT[k];
    int start = AML_POLY_SV_OFFSET[k];
    int end   = AML_POLY_SV_OFFSET[k + 1];
    for (int i = start; i < end; ++i) {
      s += AML_POLY_COEF[i] * aml_poly_kernel(x, i);
    }
    scores[k] = s;
  }

#else
  for (int k = 0; k < NUM_POSTURES; ++k) {
    float s = AML_SVM_INTERCEPT[k];
    for (int j = 0; j < AML_FEATURES; ++j) s += AML_SVM_COEF[k][j] * x[j];
    scores[k] = s;
  }
#endif
}

// ============================================================
// Window update + median smoothing + argmax
// ============================================================
static int updateWindowAndInfer(const float p_t[NUM_POSTURES]) {
  for (int k = 0; k < NUM_POSTURES; ++k) probWindow[winPos][k] = p_t[k];

  winPos = (winPos + 1) % WINDOW_LEN;
  if (winCount < WINDOW_LEN) winCount++;

  float med[NUM_POSTURES];
  float tmp[WINDOW_LEN];

  for (int k = 0; k < NUM_POSTURES; ++k) {
    for (int i = 0; i < winCount; ++i) {
      int idx = (winPos - 1 - i + WINDOW_LEN) % WINDOW_LEN;
      tmp[i] = probWindow[idx][k];
    }
    med[k] = medianInPlace(tmp, winCount);
  }

  int bestK = 0;
  float bestV = med[0];
  for (int k = 1; k < NUM_POSTURES; ++k) {
    if (med[k] > bestV) { bestV = med[k]; bestK = k; }
  }
  return bestK;
}

// ============================================================
// Parse R frame: "R,<AML_FEATURES floats ...>"
// ============================================================
static bool parseRFeatures(const char* line, float out[AML_FEATURES]) {
  if (!(line[0] == 'R' && line[1] == ',')) return false;

  const char* p = line + 2;
  int count = 0;

  while (count < AML_FEATURES) {
    while (*p && !( (*p >= '0' && *p <= '9') || *p == '-' || *p == '+' || *p == '.' )) ++p;
    if (!*p) break;

    char* end = nullptr;
    float v = strtof(p, &end);
    if (end == p || end == nullptr) return false;
    if (!isfinite(v)) return false;

    out[count++] = v;
    p = end;
  }

  return (count == AML_FEATURES);
}

// ============================================================
// Build payload: remove leading "I," and trim trailing commas/spaces,
// then append ",<code>\n"
// ============================================================
static void buildInferencePayloadNoI(const char* iLine, uint8_t code, char* out, size_t outSize) {
  const char* p = iLine;

  if (p[0] == 'I' && p[1] == ',') p += 2;

  char tmp[UART_FRAME_MAX];
  strncpy(tmp, p, sizeof(tmp) - 1);
  tmp[sizeof(tmp) - 1] = '\0';

  int L = (int)strlen(tmp);
  while (L > 0) {
    char ch = tmp[L - 1];
    if (ch == ',' || ch == ' ' || ch == '\t') { tmp[L - 1] = '\0'; --L; }
    else break;
  }

  if (L <= 0) snprintf(out, outSize, "%u\n", (unsigned)code);
  else        snprintf(out, outSize, "%s,%u\n", tmp, (unsigned)code);
}

// ============================================================
// Handle complete frame (without '/')
// ============================================================
static void handleFrame(char* frame) {
  // S frame: seat state from Mega: "S,0" or "S,1"
  if (frame[0] == 'S' && frame[1] == ',') {
    const char* p = frame + 2;
    while (*p == ' ' || *p == '\t') ++p;
    bool occ = (*p == '1');

    if (occ != seatOccupied) {
      seatOccupied = occ;
      resetInferenceState();
    } else {
      seatOccupied = occ;
    }

    if (SERIAL_DEBUG) {
      Serial.print(F("[SEAT] occupied="));
      Serial.println(seatOccupied ? 1 : 0);
    }

    bleNotifySeatState(seatOccupied);
    return;
  }

  // C frame: forward to phone
  if (frame[0] == 'C' && frame[1] == ',') {
    if (SERIAL_DEBUG) {
      Serial.print(F("[UART] C-frame: "));
      Serial.println(frame);
    }

    char outLine[UART_FRAME_MAX + 8];
    snprintf(outLine, sizeof(outLine), "%s\n", frame);

    if (BLE.connected()) bleNotifyRaw(outLine);
    return;
  }

  // I frame: cache (ignore when seat empty)
  if (frame[0] == 'I' && frame[1] == ',') {
    if (!seatOccupied) {
      if (SERIAL_DEBUG) Serial.println(F("[UART] I-frame ignored (seat empty)."));
      return;
    }

    strncpy(lastILine, frame, UART_FRAME_MAX - 1);
    lastILine[UART_FRAME_MAX - 1] = '\0';
    hasILine = true;

    if (SERIAL_DEBUG) {
      Serial.print(F("[UART] I-frame cached: "));
      Serial.println(lastILine);
    }
    return;
  }

  // R frame: inference (ignore when seat empty)
  if (frame[0] == 'R' && frame[1] == ',') {
    if (!seatOccupied) {
      if (SERIAL_DEBUG) Serial.println(F("[UART] R-frame ignored (seat empty)."));
      return;
    }

    if (!hasILine) {
      if (SERIAL_DEBUG) Serial.println(F("[WARN] R-frame arrived but no I-frame cached yet."));
      return;
    }

    if (INFERENCE_STEP_MS > 0) {
      uint32_t now = millis();
      if (now - lastInferMs < INFERENCE_STEP_MS) return;
      lastInferMs = now;
    }

    float x_in[AML_FEATURES];
    if (!parseRFeatures(frame, x_in)) {
      if (SERIAL_DEBUG) Serial.println(F("[ERR] parseRFeatures failed."));
      return;
    }

    float scores[NUM_POSTURES];
    float p_t[NUM_POSTURES];
    computeScores(x_in, scores);
    softmaxScoresToProb(scores, p_t, NUM_POSTURES);

    int y = updateWindowAndInfer(p_t);
    uint8_t code = classIndexToCode[y];

    if (SERIAL_DEBUG) {
      Serial.print(F("[INFER] class_idx="));
      Serial.print(y);
      Serial.print(F(", class_name="));
      Serial.print(AML_CLASS_NAMES[y]);
      Serial.print(F(", code="));
      Serial.println(code);
    }

    char outLine[UART_FRAME_MAX + 24];
    buildInferencePayloadNoI(lastILine, code, outLine, sizeof(outLine));

    if (SERIAL_DEBUG) {
      Serial.print(F("[BLE] payload: "));
      Serial.print(outLine);
    }

    if (BLE.connected()) {
      if (!BLE_SEND_ONLY_ON_CHANGE || code != lastSentCode) {
        bleNotifyRaw(outLine);
        lastSentCode = code;
      }
    }
    return;
  }

  if (SERIAL_DEBUG) {
    Serial.print(F("[UART] unknown frame: "));
    Serial.println(frame);
  }
}

// ============================================================
// Poll UART frames: use '/' as terminator, ignore '\n'
// ============================================================
static void pollUartFramesBySlash() {
  while (Serial1.available()) {
    char c = (char)Serial1.read();
    if (c == '\r' || c == '\n') continue;

    if (c == UART_FRAME_TERM) {
      if (!uartFrameOverflow && uartFrameLen > 0) {
        uartFrame[uartFrameLen] = '\0';
        handleFrame(uartFrame);
      } else if (SERIAL_DEBUG && uartFrameOverflow) {
        Serial.println(F("[UART] frame overflowed, dropped."));
      }
      resetUartFrame();
      continue;
    }

    if (uartFrameLen < UART_FRAME_MAX - 1) {
      uartFrame[uartFrameLen++] = c;
    } else {
      uartFrameOverflow = true;
    }
  }
}

void setup() {
  if (SERIAL_DEBUG) {
    Serial.begin(115200);
    delay(50);
    Serial.println(F("\n[ESP32] Boot"));
  }

  // UART
  Serial1.begin(UART_BAUD, SERIAL_8N1, UART_RX_PIN, UART_TX_PIN);
  Serial1.setRxBufferSize(1024);

  // BLE
  BLE.begin();
  BLE.setLocalName(DEV_NAME);
  BLE.setDeviceName(DEV_NAME);

  nus.addCharacteristic(txChar);
  nus.addCharacteristic(rxChar);
  BLE.addService(nus);

  startAdvertising();

  // class->code mapping
  buildClassIndexToCodeMap();

  if (SERIAL_DEBUG) {
    Serial.print(F("[ESP32] AML_FEATURES="));
    Serial.print(AML_FEATURES);
    Serial.print(F(", NUM_POSTURES="));
    Serial.println(NUM_POSTURES);
    Serial.println(F("[ESP32] Ready. Waiting UART frames ('/' terminated) and BLE commands."));
  }

  resetInferenceState();
  resetUartFrame();
}

void loop() {
  BLE.poll();

  // 手机 -> ESP RX -> 转发给 Mega (CAL/PING/...)
  if (rxChar.written()) {
    String cmd = rxChar.value();
    cmd.trim();

    // 兼容手机端如果带了 '/'
    if (cmd.endsWith("/")) cmd.remove(cmd.length() - 1);

    if (SERIAL_DEBUG) {
      Serial.print(F("[BLE RX] cmd="));
      Serial.println(cmd);
    }

    if (cmd.equalsIgnoreCase("CAL") || cmd.equalsIgnoreCase("CAL_PRESS")) {
      uartSendCmdFramed("CAL");   // ✅ CAL/
      if (SERIAL_DEBUG) Serial.println(F("[UART TX] -> Mega: CAL/"));
    } else if (cmd.equalsIgnoreCase("PING")) {
      uartSendCmdFramed("PING");  // ✅ PING/
      if (SERIAL_DEBUG) Serial.println(F("[UART TX] -> Mega: PING/"));
    } else {
      uartSendCmdFramed(cmd);     // ✅ 默认都用 /
      if (SERIAL_DEBUG) {
        Serial.print(F("[UART TX] -> Mega: "));
        Serial.print(cmd);
        Serial.println(F("/"));
      }
    }
  }

  // 断开后自动重启广播
  bool nowConnected = BLE.connected();
  if (lastConnected && !nowConnected) {
    startAdvertising();
    if (SERIAL_DEBUG) Serial.println(F("[BLE] disconnected -> advertising restarted"));
  }
  if (!lastConnected && nowConnected) {
    if (SERIAL_DEBUG) Serial.println(F("[BLE] connected"));
    bleNotifyRaw("On\n");
    bleNotifySeatState(seatOccupied); // ✅ 重连后同步 seat 状态
  }
  lastConnected = nowConnected;

  // Mega UART -> frames -> inference -> BLE
  pollUartFramesBySlash();
}
