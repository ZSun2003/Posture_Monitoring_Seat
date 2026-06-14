// ===================== Arduino Mega 2560 (FULL) =====================
// 功能改动：
// 1) 连续 EMPTY_WINDOWS_TO_PAUSE 个窗口总压力均值 < FSR_TOTAL_MEAN_THRESHOLD -> 暂停采样/输出；
//    暂停后仅轮询 FSR 总压力，超过阈值恢复。
//    低于阈值时 Mega->ESP32 发送 S,0/；恢复发送 S,1/；ESP32 转发给手机用于 UI 显示“无人坐”。
// 2) RECORD_MODE=true：仅输出 CSV 到 USB(Serial)，不走 ESP32（不发 Serial1 I/R/S/C 帧）
//    RECORD_MODE=false：不输出 CSV，走 ESP32（发 Serial1 I/R/S/C 帧）
// 3) 所有 UART 分帧终止符统一使用 '/'
// 4) 【新增】支持在程序里配置：A0~A9 端口 -> 逻辑FSR编号(0..9) 的映射；
//              逻辑FSR(0..9) 的坐标比值；左右/上下分组按逻辑编号配置。

#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <vl53l4ed_class.h>
#include <math.h>

#define DEV_I2C    Wire
#define SerialPort Serial

// ===================== 全局可配置参数（全部放这里） =====================

// ---------- 节拍/扫描率 ----------
static const uint32_t WINDOW_INTERVAL_MS = 100;   // 每个窗口输出一次
static const uint32_t SAMPLE_INTERVAL_MS = 20;    // 同步采样周期（FSR+ToF），建议能整除 WINDOW
static const uint8_t  CSV_DECIMALS = 4;           // 输出小数位

// ---------- 工作模式 ----------
// true  = 记录模式：只输出 CSV 到 USB(SerialPort)，不走 ESP32（不发 Serial1 数据帧）
// false = 真实设备：不输出 CSV，走 ESP32（Serial1 I/R/S/C 帧）
static const bool RECORD_MODE = false;

// ---------- 无人坐/暂停 ----------
// 连续多少个窗口 (WINDOW_INTERVAL_MS) 总压力均值 < FSR_TOTAL_MEAN_THRESHOLD 时进入暂停
static const uint8_t EMPTY_WINDOWS_TO_PAUSE = 5;

// 暂停状态下多久读一次 FSR 判断是否恢复（只读 FSR，总压力超过阈值就恢复）
static const uint32_t PAUSE_CHECK_INTERVAL_MS = SAMPLE_INTERVAL_MS;

// ---------- FSR ----------
static const uint8_t NUM_FSR = 10;

/*
  ===================== 【FSR 端口 -> 逻辑编号 映射配置】 =====================
  你现在可以明确指定：
    A0~A9 每个端口连的是哪一个逻辑FSR编号（0..9）

  - FSR_PORTS[]：你要扫描的端口列表（通常就是 A0..A9）
  - FSR_ID_AT_PORT[]：与 FSR_PORTS 一一对应，表示该端口上接的是哪个逻辑FSR编号

  例子：
    FSR_PORTS[0] = A0
    FSR_ID_AT_PORT[0] = 3
  表示：A0 上接的是逻辑 FSR#3

  注意：
  - 逻辑FSR编号 0..9 必须都出现且不重复（程序启动时会检查）
  - 输出数据（fsr0..fsr9、COP、分组比等）都是按“逻辑编号 0..9”定义的
*/
static const uint8_t FSR_PORTS[NUM_FSR] = {A0, A1, A2, A3, A4, A5, A6, A7, A8, A9};

// 端口 -> 逻辑FSR编号（按上面 FSR_PORTS[] 顺序逐一对应）
// 默认：A0->0, A1->1, ..., A9->9
static const int8_t FSR_ID_AT_PORT[NUM_FSR] = {2, 0, 3, 8, 6, 7, 9, 1, 4, 5};

// 总压力阈值：低于阈值认为无人坐/很轻，不用于训练/推理特征
static const float FSR_TOTAL_MEAN_THRESHOLD = 50.0f;

// ---------- 坐垫几何 ----------
static const float CUSHION_WIDTH_MM  = 400.0f;
static const float CUSHION_LENGTH_MM = 400.0f;

/*
  ===================== 【逻辑FSR编号(0..9) -> 坐标比值配置】 =====================
  下面两个数组是按“逻辑FSR编号”索引的：
    fsrXRatio[id], fsrYRatio[id]  (id=0..9)

  这意味着：你只要保证“逻辑编号”代表某个固定物理位置，
  无论它实际接在 A0 还是 A9，上层算法（COP、分组比、ToF门控）都不会被线序影响。
*/
static const float fsrXRatio[NUM_FSR] = {
  0.375f, 0.625f, 0.1875f, 0.375f, 0.625f,
  0.8125f, 0.1875f, 0.375f, 0.625f, 0.8125f
};
static const float fsrYRatio[NUM_FSR] = {
  0.75f, 0.75f, 0.5f, 0.5f, 0.5f,
  0.5f, 0.25f, 0.25f, 0.25f, 0.25f
};

// ---------- COP 计算策略 ----------
// true : 使用“单一权重归一化”同时算 copX/copy（更符合二维重心的几何意义，推荐）
// false: 保留原来“X/Y 分别归一化”的做法（更接近你当前输出），但仍用 LUT 提速
static const bool COP_USE_SINGLE_WEIGHT_NORM = true;

// COP LUT（setup 时初始化一次）
static uint8_t copXLevelCount[NUM_FSR];
static uint8_t copYLevelCount[NUM_FSR];
static float   copXPos[NUM_FSR];
static float   copYPos[NUM_FSR];

static void initCopLUT();

static const bool CLAMP_COORD_RATIO = true;
static const bool OUTPUT_COP_AS_MM  = false;
// ---------- COP 去偏（可选但建议开）----------
// 用“同一列/同一行取均值权重”的方式，减少传感器密度不均带来的 COP 偏移
static const bool  COP_DEBIAS_X_BY_LEVEL_COUNT = true;   // copX：按同一 x-level 计数去偏（列均值）
static const bool  COP_DEBIAS_Y_BY_LEVEL_COUNT = true;   // copY：按同一 y-level 计数去偏（行均值）
static const float COP_LEVEL_MATCH_TOL = 1e-6f;          // x/y level 判断为“同一行/列”的容差

/*
  ===================== 【左右/上下分组：按逻辑FSR编号配置】 =====================
  以前你用的是“按引脚分组”（A0..A4 左，A5..A9 右），换线序就会错。
  现在改成“按逻辑编号分组”，只要逻辑编号固定代表物理位置，线怎么接都不影响分组结果。
*/
static const uint8_t LEFT_GROUP_IDS[]  = {0, 2, 3, 6, 7};
static const uint8_t RIGHT_GROUP_IDS[] = {1, 4, 5, 8, 9};
// ---------- 前 / 中 / 后 分组（按逻辑FSR编号）----------
// 约定：yRatio 越大越靠“前”（靠膝盖/坐垫前沿），yRatio 越小越靠“后”（靠背）
// 如果你的坐标系相反：直接交换 FRONT_GROUP_IDS 和 BACK_GROUP_IDS 即可。
static const uint8_t FRONT_GROUP_IDS[]  = {6, 7, 8, 9};         // y=0.75
static const uint8_t MIDDLE_GROUP_IDS[] = {2, 3, 4, 5};   // y=0.5
static const uint8_t BACK_GROUP_IDS[]   = {0, 1};   // y=0.25

static const uint8_t FRONT_GROUP_COUNT  = sizeof(FRONT_GROUP_IDS)  / sizeof(FRONT_GROUP_IDS[0]);
static const uint8_t MIDDLE_GROUP_COUNT = sizeof(MIDDLE_GROUP_IDS) / sizeof(MIDDLE_GROUP_IDS[0]);
static const uint8_t BACK_GROUP_COUNT   = sizeof(BACK_GROUP_IDS)   / sizeof(BACK_GROUP_IDS[0]);

static const uint8_t LEFT_GROUP_COUNT  = sizeof(LEFT_GROUP_IDS)  / sizeof(LEFT_GROUP_IDS[0]);
static const uint8_t RIGHT_GROUP_COUNT = sizeof(RIGHT_GROUP_IDS) / sizeof(RIGHT_GROUP_IDS[0]);

static const float RATIO_EPS = 1.0f;

// ---------- ToF ----------
static const uint8_t  TOF_XSHUT_PIN = A10;
static const uint32_t I2C_CLOCK_HZ   = 400000;
static const uint16_t TOF_TIMING_BUDGET_MS = 10;
static const uint16_t TOF_INTERMEASUREMENT_MS = 0;

// ---------- ToF 校准按钮 ----------
static const uint16_t CAL_SAMPLES = 30;
static const uint16_t CAL_TIMEOUT_MS = 6000;
static const uint16_t CAL_MIN_DELTA_MM = 20;
static const uint16_t CAL_LIVE_PRINT_MS = 1000;

// ---------- ToF 门控阈值 ----------
static const float TOF_LATERAL_THR = 0.05f;

// ---------- Serial1 帧终止符 ----------
static const char FRAME_TERM = '/';

// ---------- Serial1 整数缩放 ----------
static const uint16_t INT_SCALE = 1000;

// ===== ToF debug vars =====
static uint8_t  tofDbg_status = 0;
static uint8_t  tofDbg_newReady = 0;
static uint8_t  tofDbg_rangeStatus = 255;
static uint16_t tofDbg_rawMm = 0;
static uint32_t tofDbg_updatesOk = 0;
static uint32_t tofDbg_updatesAny = 0;

// ===================== 固定实现区 =====================

VL53L4ED tof(&DEV_I2C, TOF_XSHUT_PIN);

// ToF 三点标定变量：后靠/坐正/前倾
enum CalStage : uint8_t {
  CAL_NONE = 0,
  CAL_HAVE_BACK = 1,
  CAL_HAVE_UPRIGHT = 2,
  CAL_DONE = 3
};
static CalStage calStage = CAL_NONE;

static uint16_t tofBackMm    = 0;
static uint16_t tofUprightMm = 0;
static uint16_t tofForwardMm = 0;

// 每窗口最大样本数（给 ToF median buffer 用）
static const uint16_t MAX_SAMPLES_PER_WINDOW =
  (uint16_t)(((WINDOW_INTERVAL_MS + SAMPLE_INTERVAL_MS - 1) / SAMPLE_INTERVAL_MS) + 2);

// 窗口累加器（按逻辑FSR编号 0..9 存储）
static uint32_t fsrSum[NUM_FSR] = {0};
static uint64_t fsrSumSq[NUM_FSR] = {0};
static uint16_t tofWin[MAX_SAMPLES_PER_WINDOW];
static uint32_t syncSamples = 0;

// ToF last
static bool     tofHasLast = false;
static uint16_t tofLastMm  = 0;

// 调度
static uint32_t startMs      = 0;
static uint32_t nextSampleMs = 0;
static uint32_t nextOutMs    = 0;
static uint32_t windowIndex  = 0;

// ====== virtual calibration trigger from Serial1/USB ======
static volatile bool calVirtualPressed = false;
// ====== runtime commands (from ESP32/phone) ======
static volatile bool calRequested   = false; // 运行期：请求重新进入三步 ToF 校准
static volatile bool calPressQueued = false; // 运行期：触发校准的那一次 CAL 直接当作 BACK 阶段按键
static bool          calInProgress  = false; // true：正在校准流程中（此时 CAL 表示“按键”）

// ====== Seat empty/pause control ======
static bool     seatPaused = false;
static uint8_t  emptyWinCount = 0;         // 连续“无人坐(<threshold)”窗口数
static bool     seatEmptyReported = false; // 已向 ESP32 报告“无人坐”？
static uint32_t lastPauseCheckMs = 0;

// ---------- 前置声明 ----------
static void validateFsrPortMapOrHalt();

static void pollCalVirtualFromSerial1();
static void pollCalVirtualFromSerialPort();
static void sendPongFrame();
static void runToFCalibrationBlocking();
static void resetStreamStateAfterCalibration();
static void waitForCalPressWithLiveToF(const __FlashStringHelper *stageName);

static void sendCFrameKV(const char* key, const char* val);
static void sendCFrameStageTof(const __FlashStringHelper* stageName, int tofMm);
static void sendCFrameMsg(const __FlashStringHelper* msg);
static void sendCFrameCaptured(const __FlashStringHelper* stageName, int baseMm);
static float medianF32InPlace(float *a, uint8_t n);
static float medianGroupByIdsF32(const float *vals, const uint8_t *ids, uint8_t count);

static uint32_t readFsrTotalInstant();
static void sendSeatStateFrame(bool occupied);
static void enterSeatPaused();
static void handleSeatPaused();

static uint32_t sumGroupByIds(const uint8_t *ids, uint8_t count);
static float    sumShareByIds(const float *fsrNorm, const uint8_t *ids, uint8_t count);
static uint8_t countSameLevel(const float *levels, uint8_t idx, float tol);
static uint8_t countSameXLevel(uint8_t idx);
static uint8_t countSameYLevel(uint8_t idx);
static void initCopLUT();

// ---------- 小工具 ----------
static void calPrintBothLn(const __FlashStringHelper* s) {
  SerialPort.println(s);
  // ⚠️ Serial1 是帧协议（以 '/' 分帧），不要输出纯文本到 Serial1（会污染 ESP32 分帧）
}
static void calPrintBothLn(const char* s) {
  SerialPort.println(s);
  // 同上
}

static float clamp01(float v) {
  if (v < 0.0f) return 0.0f;
  if (v > 1.0f) return 1.0f;
  return v;
}

static void printCsvHeader(Print &out) {
  out.print(F("t_ms"));
  for (uint8_t i = 0; i < NUM_FSR; i++) {
    out.print(F(",fsr")); out.print(i); out.print(F("_norm"));
  }
  out.print(F(",tof_norm"));
  out.print(F(",tof_gated,tof_use"));
  for (uint8_t i = 0; i < NUM_FSR; i++) {
    out.print(F(",fsr")); out.print(i); out.print(F("_std"));
  }
  out.print(F(",lr_ratio,ud_ratio,"));
  if (OUTPUT_COP_AS_MM) out.println(F("cop_x_mm,cop_y_mm"));
  else                 out.println(F("cop_x_ratio,cop_y_ratio"));
}

static uint32_t trunc_x1000_nonneg(float v) {
  if (v <= 0.0f) return 0;
  return (uint32_t)(v * (float)INT_SCALE);
}

static uint8_t trunc_use01(float v) {
  return (v >= 0.5f) ? 1 : 0;
}

// ============ Serial1 双通道输出（均以 '/' 终止）===========
//
// I format (int x1000, used by phone):
//   I,fsr0..fsr9,tof_gated,tof_use,cop_x,cop_y,/
//
// R format (raw float, used by ESP local inference/training):
//   R,fsr0_norm..fsr9_norm,tof_gated,tof_use,lr_ratio,ud_ratio,cop_x,cop_y,/

static void writeLineSerial1_I(Print &out,
                              const float *fsrNorm,
                              float tofGated,
                              float tofUse,
                              float copXOut,
                              float copYOut) {
  out.print(F("I"));
  for (uint8_t i = 0; i < NUM_FSR; i++) {
    out.print(',');
    out.print(trunc_x1000_nonneg(fsrNorm[i]));
  }
  out.print(',');
  out.print(trunc_x1000_nonneg(tofGated));
  out.print(',');
  out.print((unsigned)trunc_use01(tofUse));
  out.print(',');
  out.print(trunc_x1000_nonneg(copXOut));
  out.print(',');
  out.print(trunc_x1000_nonneg(copYOut));
  out.print(',');
  out.print(FRAME_TERM);
  out.print('\n');
}

static void writeLineSerial1_R(Print &out,
                              const float *fsrNorm,
                              float tofGated,
                              float tofUse,
                              float lrRatio,
                              float udRatio,
                              float copXOut,
                              float copYOut,
                              uint8_t decimals) {
  out.print(F("R"));
  for (uint8_t i = 0; i < NUM_FSR; i++) {
    out.print(',');
    out.print(fsrNorm[i], decimals);
  }
  out.print(',');
  out.print(tofGated, decimals);
  out.print(',');
  out.print(tofUse, decimals);
  out.print(',');
  out.print(lrRatio, decimals);
  out.print(',');
  out.print(udRatio, decimals);
  out.print(',');
  out.print(copXOut, decimals);
  out.print(',');
  out.print(copYOut, decimals);
  out.print(',');
  out.print(FRAME_TERM);
  out.print('\n');
}

static void writeCsvLine(Print &out,
                         uint32_t t_ms,
                         const float *fsrNorm,
                         float tofNorm,
                         float tofGated,
                         float tofUse,
                         const float *fsrStd,
                         float lrRatio,
                         float udRatio,
                         float copX,
                         float copY,
                         uint8_t decimals) {
  out.print(t_ms);

  for (uint8_t i = 0; i < NUM_FSR; i++) { out.print(','); out.print(fsrNorm[i], decimals); }

  out.print(','); out.print(tofNorm, decimals);
  out.print(','); out.print(tofGated, decimals);
  out.print(','); out.print(tofUse, decimals);

  for (uint8_t i = 0; i < NUM_FSR; i++) { out.print(','); out.print(fsrStd[i], decimals); }

  out.print(','); out.print(lrRatio, decimals);
  out.print(','); out.print(udRatio, decimals);
  out.print(','); out.print(copX, decimals);
  out.print(','); out.println(copY, decimals);
}

/*
  ===================== 配置校验：端口->逻辑FSR编号映射必须合法 =====================
  - 每个端口的 id 必须在 [0..NUM_FSR-1]
  - id 不能重复
  - 0..NUM_FSR-1 必须全部出现一次
*/
static void validateFsrPortMapOrHalt() {
  bool seen[NUM_FSR];
  for (uint8_t i = 0; i < NUM_FSR; i++) seen[i] = false;

  bool ok = true;

  SerialPort.println(F("# [FSR_MAP] Port -> LogicalID"));
  for (uint8_t p = 0; p < NUM_FSR; p++) {
    int8_t id = FSR_ID_AT_PORT[p];
    SerialPort.print(F("#   port="));
    SerialPort.print(FSR_PORTS[p]);
    SerialPort.print(F("  -> id="));
    SerialPort.println(id);

    if (id < 0 || id >= (int8_t)NUM_FSR) {
      SerialPort.println(F("# [FSR_MAP][ERR] id out of range (must be 0..9)"));
      ok = false;
      continue;
    }
    if (seen[(uint8_t)id]) {
      SerialPort.println(F("# [FSR_MAP][ERR] duplicated id"));
      ok = false;
      continue;
    }
    seen[(uint8_t)id] = true;
  }

  for (uint8_t id = 0; id < NUM_FSR; id++) {
    if (!seen[id]) {
      SerialPort.print(F("# [FSR_MAP][ERR] missing id="));
      SerialPort.println(id);
      ok = false;
    }
  }

  if (!ok) {
    SerialPort.println(F("# [FSR_MAP] INVALID CONFIG. Fix FSR_ID_AT_PORT[] then reset."));
    while (true) { delay(200); }
  }

  SerialPort.println(F("# [FSR_MAP] OK"));
}

static uint32_t sumGroupByIds(const uint8_t *ids, uint8_t count) {
  uint32_t sum = 0;
  for (uint8_t k = 0; k < count; k++) {
    uint8_t id = ids[k];
    if (id < NUM_FSR) sum += fsrSum[id];
  }
  return sum;
}

static float sumShareByIds(const float *fsrNorm, const uint8_t *ids, uint8_t count) {
  float sum = 0.0f;
  for (uint8_t k = 0; k < count; k++) {
    uint8_t id = ids[k];
    if (id < NUM_FSR) sum += fsrNorm[id];
  }
  return sum;
}

static float medianF32InPlace(float *a, uint8_t n) {
  if (n == 0) return 0.0f;

  // insertion sort（n<=10，很快）
  for (uint8_t i = 1; i < n; i++) {
    float key = a[i];
    int8_t j = (int8_t)i - 1;
    while (j >= 0 && a[(uint8_t)j] > key) {
      a[(uint8_t)j + 1] = a[(uint8_t)j];
      j--;
    }
    a[(uint8_t)j + 1] = key;
  }

  if (n & 1) return a[n / 2];
  return 0.5f * (a[n / 2 - 1] + a[n / 2]);
}

// 对某组 ids 取 vals[id] 的中位数
static float medianGroupByIdsF32(const float *vals, const uint8_t *ids, uint8_t count) {
  if (count == 0) return 0.0f;

  float tmp[NUM_FSR];
  uint8_t n = 0;

  for (uint8_t k = 0; k < count; k++) {
    uint8_t id = ids[k];
    if (id < NUM_FSR) tmp[n++] = vals[id];
  }
  return medianF32InPlace(tmp, n);
}

static uint8_t countSameLevel(const float *levels, uint8_t idx, float tol) {
  float v = levels[idx];
  uint8_t cnt = 0;
  for (uint8_t j = 0; j < NUM_FSR; j++) {
    if (fabsf(levels[j] - v) <= tol) cnt++;
  }
  // 理论上不应为 0，兜底返回 1 防止除0
  return (cnt == 0) ? 1 : cnt;
}

static void initCopLUT() {
  // 1) 预计算位置（含 clamp）
  for (uint8_t i = 0; i < NUM_FSR; i++) {
    float xr = fsrXRatio[i];
    float yr = fsrYRatio[i];
    if (CLAMP_COORD_RATIO) { xr = clamp01(xr); yr = clamp01(yr); }
    copXPos[i] = xr;
    copYPos[i] = yr;
  }

  // 2) 预计算每个点所在的“同列/同行”数量（基于 clamped 坐标，避免坐标越界时不一致）
  for (uint8_t i = 0; i < NUM_FSR; i++) {
    copXLevelCount[i] = COP_DEBIAS_X_BY_LEVEL_COUNT ? countSameLevel(copXPos, i, COP_LEVEL_MATCH_TOL) : 1;
    copYLevelCount[i] = COP_DEBIAS_Y_BY_LEVEL_COUNT ? countSameLevel(copYPos, i, COP_LEVEL_MATCH_TOL) : 1;
  }
}

static uint8_t countSameXLevel(uint8_t idx) {
  return countSameLevel(fsrXRatio, idx, COP_LEVEL_MATCH_TOL);
}

static uint8_t countSameYLevel(uint8_t idx) {
  return countSameLevel(fsrYRatio, idx, COP_LEVEL_MATCH_TOL);
}

static bool captureToFBaselineMm(uint16_t samplesWanted, uint16_t timeoutMs, uint16_t &outMm) {
  uint32_t start = millis();
  uint32_t sum = 0;
  uint16_t count = 0;

  while (count < samplesWanted && (uint32_t)(millis() - start) < timeoutMs) {
    uint8_t newDataReady = 0;
    uint8_t status = tof.VL53L4ED_CheckForDataReady(&newDataReady);

    if ((status == 0) && (newDataReady != 0)) {
      VL53L4ED_ResultsData_t results;
      tof.VL53L4ED_ClearInterrupt();
      tof.VL53L4ED_GetResult(&results);

      if (results.range_status == 0) {
        sum += results.distance_mm;
        count++;
      }
    }
    delay(1);
  }

  if (count == 0) return false;
  outMm = (uint16_t)((sum + (count / 2)) / count);
  return true;
}

static void updateToFLastMmNonBlocking() {
  uint8_t newDataReady = 0;
  uint8_t status = tof.VL53L4ED_CheckForDataReady(&newDataReady);

  tofDbg_status = status;
  tofDbg_newReady = newDataReady;

  if ((status == 0) && (newDataReady != 0)) {
    VL53L4ED_ResultsData_t results;
    tof.VL53L4ED_ClearInterrupt();
    tof.VL53L4ED_GetResult(&results);

    tofDbg_rangeStatus = results.range_status;
    tofDbg_rawMm = results.distance_mm;
    tofDbg_updatesAny++;

    tofLastMm = results.distance_mm;
    tofHasLast = true;

    if (results.range_status == 0) {
      tofDbg_updatesOk++;
    }
  }
}

static float medianU16InPlace(uint16_t *a, uint16_t n) {
  if (n == 0) return -1.0f;

  for (uint16_t i = 1; i < n; i++) {
    uint16_t key = a[i];
    int16_t j = (int16_t)i - 1;
    while (j >= 0 && a[j] > key) {
      a[j + 1] = a[j];
      j--;
    }
    a[j + 1] = key;
  }

  if (n & 1) return (float)a[n / 2];
  return 0.5f * ((float)a[n / 2 - 1] + (float)a[n / 2]);
}

static void resetWindowAcc() {
  for (uint8_t i = 0; i < NUM_FSR; i++) {
    fsrSum[i] = 0;
    fsrSumSq[i] = 0;
  }
  syncSamples = 0;
}

// ===== Mega -> ESP32: calibration frame sender (C,...,/)
static void sendCFrameKV(const char* key, const char* val) {
  if (RECORD_MODE) return;
  Serial1.print(F("C,"));
  Serial1.print(key);
  Serial1.print('=');
  Serial1.print(val);
  Serial1.print(F(",/"));
  Serial1.print('\n');
}

static void sendCFrameStageTof(const __FlashStringHelper* stageName, int tofMm) {
  if (RECORD_MODE) return;
  Serial1.print(F("C,stage="));
  Serial1.print(stageName);
  Serial1.print(F(",tof_mm="));
  Serial1.print(tofMm);
  Serial1.print(F(",/"));
  Serial1.print('\n');
}

static void sendCFrameMsg(const __FlashStringHelper* msg) {
  if (RECORD_MODE) return;
  Serial1.print(F("C,msg="));
  Serial1.print(msg);
  Serial1.print(F(",/"));
  Serial1.print('\n');
}

static void sendCFrameCaptured(const __FlashStringHelper* stageName, int baseMm) {
  if (RECORD_MODE) return;
  Serial1.print(F("C,stage="));
  Serial1.print(stageName);
  Serial1.print(F(",captured_mm="));
  Serial1.print(baseMm);
  Serial1.print(F(",/"));
  Serial1.print('\n');
}

static void sendPongFrame() {
  if (RECORD_MODE) {
    // 记录模式：不走 Serial1 帧协议，但在 USB 串口给一个明确回包，便于 Python 端测试
    SerialPort.println(F("# [CMD] PONG"));
    return;
  }
  // exactly: C,msg=PONG/
  Serial1.print(F("C,msg=PONG"));
  Serial1.print(FRAME_TERM); // '/'
  Serial1.print('\n');
}


// ---------- NEW: reset streaming scheduler/state after (re)calibration ----------
static void resetStreamStateAfterCalibration() {
  // 用坐正值作为 ToF 初始 last
  tofLastMm = tofUprightMm;
  tofHasLast = true;

  // 清空窗口累积
  resetWindowAcc();

  // 重启调度（从当前时刻重新计时）
  uint32_t now = millis();
  startMs = now;
  nextSampleMs = now;
  nextOutMs = now + WINDOW_INTERVAL_MS;
  windowIndex = 0;

  // 清空 pause/empty 状态，避免校准后卡在暂停
  seatPaused = false;
  emptyWinCount = 0;
  lastPauseCheckMs = now;

  // 立即把 seat 状态同步给 ESP32/手机（用瞬时总压力）
  if (!RECORD_MODE) {
    uint32_t total = readFsrTotalInstant();
    bool occ = ((float)total >= FSR_TOTAL_MEAN_THRESHOLD);
    sendSeatStateFrame(occ);
    seatEmptyReported = !occ;
  } else {
    seatEmptyReported = false;
  }
}

// ---------- NEW: 3-step ToF calibration (callable in setup() and runtime) ----------
static void runToFCalibrationBlocking() {
  calInProgress = true;
  calStage = CAL_NONE;

  // 如果是运行期手机发 CAL 触发的：把这一按当做 BACK 阶段的 "按键"
  if (calPressQueued) {
    calVirtualPressed = true;
    calPressQueued = false;
  } else {
    calVirtualPressed = false;
  }

  calPrintBothLn(F("# ToF calibration (3 steps):"));
  calPrintBothLn(F("# 1) Lean BACK (recline), then press CAL."));
  calPrintBothLn(F("# 2) Sit UPRIGHT, then press CAL."));
  calPrintBothLn(F("# 3) Lean FORWARD, then press CAL."));
  calPrintBothLn(F("# Streaming starts after calibration."));

  if (!RECORD_MODE) {
    Serial1.print(F("C,msg=CAL_BEGIN"));
    Serial1.print(FRAME_TERM);
    Serial1.print('\n');
  }

  // Stage 1: BACK
  while (true) {
    waitForCalPressWithLiveToF(F("BACK"));
    uint16_t d = 0;
    if (captureToFBaselineMm(CAL_SAMPLES, CAL_TIMEOUT_MS, d)) {
      tofBackMm = d;
      sendCFrameCaptured(F("BACK"), (int)tofBackMm);
      calStage = CAL_HAVE_BACK;
      break;
    }
    calPrintBothLn(F("# [CAL] Back capture failed, try again."));
  }

  // Stage 2: UPRIGHT
  while (true) {
    waitForCalPressWithLiveToF(F("UPRIGHT"));
    uint16_t d = 0;
    if (captureToFBaselineMm(CAL_SAMPLES, CAL_TIMEOUT_MS, d)) {
      tofUprightMm = d;
      sendCFrameCaptured(F("UPRIGHT"), (int)tofUprightMm);
      if (tofUprightMm > (uint16_t)(tofBackMm + CAL_MIN_DELTA_MM)) {
        calStage = CAL_HAVE_UPRIGHT;
        break;
      }
      calPrintBothLn(F("# [CAL] Upright too close to BACK. Sit more upright and try again."));
    } else {
      calPrintBothLn(F("# [CAL] Upright capture failed, try again."));
    }
  }

  // Stage 3: FORWARD
  while (true) {
    waitForCalPressWithLiveToF(F("FORWARD"));
    uint16_t d = 0;
    if (captureToFBaselineMm(CAL_SAMPLES, CAL_TIMEOUT_MS, d)) {
      tofForwardMm = d;
      sendCFrameCaptured(F("FORWARD"), (int)tofForwardMm);
      if (tofForwardMm > (uint16_t)(tofUprightMm + CAL_MIN_DELTA_MM)) {
        calStage = CAL_DONE;
        break;
      }
      calPrintBothLn(F("# [CAL] Forward too close to UPRIGHT, lean more and try again."));
    } else {
      calPrintBothLn(F("# [CAL] Forward capture failed, try again."));
    }
  }

  if (!RECORD_MODE) {
    Serial1.print(F("C,msg=CAL_DONE"));
    Serial1.print(FRAME_TERM);
    Serial1.print('\n');
  }

  calInProgress = false;
}

// ---------- Seat state / pause helpers ----------
static uint32_t readFsrTotalInstant() {
  uint32_t total = 0;
  // 注意：这里是按“端口”读瞬时值，不依赖逻辑编号
  for (uint8_t p = 0; p < NUM_FSR; p++) {
    total += (uint16_t)analogRead(FSR_PORTS[p]);
  }
  return total;
}

// Mega -> ESP32: seat state frame (S,<0|1>/)
static void sendSeatStateFrame(bool occupied) {
  if (RECORD_MODE) return;
  Serial1.print(F("S,"));
  Serial1.print(occupied ? 1 : 0);
  Serial1.print(FRAME_TERM); // '/'
  Serial1.print('\n');
}

static void enterSeatPaused() {
  seatPaused = true;
  lastPauseCheckMs = millis();
  resetWindowAcc();
}

static void handleSeatPaused() {
  uint32_t now = millis();
  if ((uint32_t)(now - lastPauseCheckMs) < PAUSE_CHECK_INTERVAL_MS) return;
  lastPauseCheckMs = now;

  uint32_t total = readFsrTotalInstant();
  if ((float)total >= FSR_TOTAL_MEAN_THRESHOLD) {
    seatPaused = false;
    emptyWinCount = 0;

    if (seatEmptyReported) {
      sendSeatStateFrame(true);
      seatEmptyReported = false;
    }

    nextSampleMs = now;
    nextOutMs    = now + WINDOW_INTERVAL_MS;
    resetWindowAcc();
  }
}

// ---------- UART/USB CAL polling ----------
static void pollCalVirtualFromSerial1() {
  while (Serial1.available()) {
    char c = (char)Serial1.read();
    static char buf[16];
    static uint8_t n = 0;

    if (c == '\r' || c == '\n') continue;

    // 统一大小写（兼容手工输入）
    if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');

    if (c == FRAME_TERM) { // '/'
      buf[n] = '\0';
      n = 0;

      if (strcmp(buf, "PING") == 0) {
        sendPongFrame();
        SerialPort.println(F("# [CMD] PING -> PONG"));

      } else if (strcmp(buf, "CAL") == 0 || strcmp(buf, "C") == 0) {
        // 允许输入 "CAL" 或 "C"
        if (calInProgress) {
          calVirtualPressed = true;
        } else {
          calRequested = true;
          calPressQueued = true;
          SerialPort.println(F("# [CMD] CAL requested (runtime)"));
        }
      }

      // 其它命令先忽略（避免污染协议）
      continue;
    }

    if (n < sizeof(buf) - 1) buf[n++] = c;
    else n = 0;
  }
}

static void pollCalVirtualFromSerialPort() {
  while (SerialPort.available()) {
    char c = (char)SerialPort.read();
    static char buf[16];
    static uint8_t n = 0;

    if (c == '\r') continue;

    // 统一大小写
    if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');

    // 兼容 Serial Monitor 的 '\n'，也兼容手工输入 "CAL/"、"PING/"
    if (c == '\n' || c == FRAME_TERM) {
      buf[n] = '\0';
      n = 0;

      if (strcmp(buf, "PING") == 0) {
        sendPongFrame();
        SerialPort.println(F("# [CMD] PING -> PONG"));
      } else if (strcmp(buf, "CAL") == 0) {
        if (calInProgress) {
          calVirtualPressed = true;
        } else {
          calRequested = true;
          calPressQueued = true;
          SerialPort.println(F("# [CMD] CAL requested (runtime)"));
        }
      }
      continue;
    }

    if (n < sizeof(buf) - 1) buf[n++] = c;
    else n = 0;
  }
}

static void waitForCalPressWithLiveToF(const __FlashStringHelper *stageName) {
  uint32_t lastPrint = 0;

  SerialPort.print(F("# [CAL] "));
  SerialPort.print(stageName);
  SerialPort.println(F(" - live ToF distance (mm). Send CAL/ from phone, or type CAL in USB Serial Monitor."));

  if (!RECORD_MODE) {
    sendCFrameMsg(F("CAL_WAIT_PRESS"));
    Serial1.print(F("C,stage="));
    Serial1.print(stageName);
    Serial1.print(F(",action=WAIT_CAL,/"));
    Serial1.print('\n');
  }

  // ⚠️ 不要在这里强制清零 calVirtualPressed！
  // 运行期手机发 CAL 触发重新校准时，我们会把“这一按”排队给 BACK 阶段使用。
  while (!calVirtualPressed) {
    pollCalVirtualFromSerial1();
    pollCalVirtualFromSerialPort();
    updateToFLastMmNonBlocking();

    uint32_t now = millis();
    if (now - lastPrint >= CAL_LIVE_PRINT_MS) {
      lastPrint = now;

      SerialPort.print(F("# [CAL_LIVE] stage="));
      SerialPort.print(stageName);
      SerialPort.print(F(",tof_mm="));
      SerialPort.println(tofHasLast ? (int)tofLastMm : -1);

      if (!RECORD_MODE) {
        sendCFrameStageTof(stageName, tofHasLast ? (int)tofLastMm : -1);
      }
    }
    delay(1);
  }

  // consume press
  calVirtualPressed = false;

  if (!RECORD_MODE) {
    Serial1.print(F("C,stage="));
    Serial1.print(stageName);
    Serial1.print(F(",action=CAL_RECEIVED,/"));
    Serial1.print('\n');
  }
}

void setup() {
  SerialPort.begin(115200);
  Serial1.begin(115200);

  // 先做配置校验（否则下面采样会乱）
  validateFsrPortMapOrHalt();
  initCopLUT();
  DEV_I2C.begin();
  DEV_I2C.setClock(I2C_CLOCK_HZ);

  // ToF init
  tof.begin();
  tof.VL53L4ED_Off();
  tof.InitSensor();
  tof.VL53L4ED_SetRangeTiming(TOF_TIMING_BUDGET_MS, TOF_INTERMEASUREMENT_MS);
  tof.VL53L4ED_StartRanging();

  // 三点校准提示
  calPrintBothLn(F("# ToF calibration (3 steps):"));
  calPrintBothLn(F("# 1) Lean BACK (recline), then press CAL."));
  calPrintBothLn(F("# 2) Sit UPRIGHT, then press CAL."));
  calPrintBothLn(F("# 3) Lean FORWARD, then press CAL."));
  calPrintBothLn(F("# Streaming starts after calibration."));

  // 启动时三步 ToF 校准
  runToFCalibrationBlocking();

  // 校准完成后：重置窗口/调度，并同步 seat 状态给手机
  resetStreamStateAfterCalibration();

  // 记录模式才输出 CSV 头
  if (RECORD_MODE) {
    printCsvHeader(SerialPort);
  }
}

void loop() {
  // ✅ 运行期也要轮询 UART/USB 命令（PING/CAL）
  pollCalVirtualFromSerial1();
  pollCalVirtualFromSerialPort();

  // ✅ 运行期：手机发 CAL/ 重新触发三步校准
  if (calRequested) {
    calRequested = false;
    runToFCalibrationBlocking();
    resetStreamStateAfterCalibration();
    return; // 校准结束后从新调度开始
  }

  // PAUSED：只检查是否恢复
  if (seatPaused) {
    handleSeatPaused();
    return;
  }

  uint32_t now = millis();
  bool bailForPause = false;

  while (true) {
    uint32_t nextEventMs = (nextOutMs <= nextSampleMs) ? nextOutMs : nextSampleMs;
    if ((int32_t)(now - nextEventMs) < 0) break;

    // 1) 输出事件（窗口结束）
    if (nextOutMs <= nextSampleMs) {
      float leftShare  = 0.0f;
      float rightShare = 0.0f;
      float fsrNorm[NUM_FSR];
      float fsrStd[NUM_FSR];

      float tofNorm  = -1.0f;
      float tofGated = 0.5f;
      float tofUse   = 0.0f;

      float lrRatio = 0.0f;
      float udRatio = 0.0f;

      float copXRatio = 0.0f;
      float copYRatio = 0.0f;
      float copXOut   = 0.0f;
      float copYOut   = 0.0f;

      uint32_t fsrTotal = 0;
      for (uint8_t i = 0; i < NUM_FSR; i++) fsrTotal += fsrSum[i];

      bool seatOk = false;
      if (syncSamples > 0) {
        float fsrTotalMean = (float)fsrTotal / (float)syncSamples;
        seatOk = (fsrTotalMean >= FSR_TOTAL_MEAN_THRESHOLD);
      }

      // seat empty 计数 + 状态帧（S）
      if (!seatOk) {
        emptyWinCount++;
        if (!seatEmptyReported) {
          sendSeatStateFrame(false);     // S,0/
          seatEmptyReported = true;
        }
      } else {
        emptyWinCount = 0;
        if (seatEmptyReported) {
          sendSeatStateFrame(true);      // S,1/
          seatEmptyReported = false;
        }
      }

      bool enterPause = (!seatOk && (emptyWinCount >= EMPTY_WINDOWS_TO_PAUSE));

      // (1) FSR 归一化
      if (!seatOk || fsrTotal == 0) {
        for (uint8_t i = 0; i < NUM_FSR; i++) fsrNorm[i] = 0.0f;
      } else {
        for (uint8_t i = 0; i < NUM_FSR; i++) fsrNorm[i] = (float)fsrSum[i] / (float)fsrTotal;
      }

      // (2) std
      if (!seatOk || syncSamples < 2) {
        for (uint8_t i = 0; i < NUM_FSR; i++) fsrStd[i] = 0.0f;
      } else {
        float invN = 1.0f / (float)syncSamples;
        for (uint8_t i = 0; i < NUM_FSR; i++) {
          float mean = (float)fsrSum[i] * invN;
          float meanSq = (float)fsrSumSq[i] * invN;
          float var = meanSq - mean * mean;
          if (var < 0.0f) var = 0.0f;
          fsrStd[i] = sqrt(var);
        }
      }

      // (3) ToF norm
      if (seatOk && syncSamples > 0 && calStage == CAL_DONE) {
        uint16_t n = (syncSamples > MAX_SAMPLES_PER_WINDOW) ? MAX_SAMPLES_PER_WINDOW : (uint16_t)syncSamples;
        float tofMedMm = medianU16InPlace(tofWin, n);

        float dB = (float)tofBackMm;
        float dU = (float)tofUprightMm;
        float dF = (float)tofForwardMm;

        float denomBack = dU - dB;
        float denomFwd  = dF - dU;

        if (denomBack > 1e-6f && denomFwd > 1e-6f) {
          float norm;
          if (tofMedMm <= dU) norm = 0.5f * (tofMedMm - dB) / denomBack;
          else               norm = 0.5f + 0.5f * (tofMedMm - dU) / denomFwd;
          tofNorm = clamp01(norm);
        } else {
          tofNorm = -1.0f;
        }
      }

      // (4) lr/ud ratio
      //     - lrRatio：保持左右（归一化 fsrNorm 的组均值比）
      //     - udRatio：三组(前/中/后)，组内用中位数；输出范围 0..1
      //         * 0.5：中部基准（前后相当 or 中部占主导）
      //         * (0.5, 1]：更偏前坐（front > back）
      //         * [0, 0.5)：更偏后坐（back > front）
      if (seatOk) {
        // eps：把原先“原始ADC域的RATIO_EPS”映射到归一化域，避免除0
        float epsNorm = 1e-6f;
        if (fsrTotal > 0) {
          epsNorm = (float)RATIO_EPS / (float)fsrTotal;
          if (epsNorm < 1e-6f) epsNorm = 1e-6f;
        }

        // ---- LR：均值比（不变）----
        leftShare  = (LEFT_GROUP_COUNT  > 0) ? sumShareByIds(fsrNorm, LEFT_GROUP_IDS,  LEFT_GROUP_COUNT)  : 0.0f;
        rightShare = (RIGHT_GROUP_COUNT > 0) ? sumShareByIds(fsrNorm, RIGHT_GROUP_IDS, RIGHT_GROUP_COUNT) : 0.0f;

        float leftMean  = (LEFT_GROUP_COUNT  > 0) ? (leftShare  / (float)LEFT_GROUP_COUNT)  : 0.0f;
        float rightMean = (RIGHT_GROUP_COUNT > 0) ? (rightShare / (float)RIGHT_GROUP_COUNT) : 0.0f;

        lrRatio = (leftMean + epsNorm) / (rightMean + epsNorm);

        // ---- UD：三组中位数 + 0..1 映射（关键改动）----
        float frontMed  = medianGroupByIdsF32(fsrNorm, FRONT_GROUP_IDS,  FRONT_GROUP_COUNT);
        float middleMed = medianGroupByIdsF32(fsrNorm, MIDDLE_GROUP_IDS, MIDDLE_GROUP_COUNT);
        float backMed   = medianGroupByIdsF32(fsrNorm, BACK_GROUP_IDS,   BACK_GROUP_COUNT);

        // signedStrength ∈ [-1,1]（理论上），加入 middleMed 让“中部压力越大 -> 越趋近0.5”
        float denom = frontMed + backMed + middleMed + epsNorm;
        float signedStrength = (denom > 0.0f) ? ((frontMed - backMed) / denom) : 0.0f;

        // 映射到 [0,1]，并夹紧（数值误差兜底）
        udRatio = clamp01(0.5f + 0.5f * signedStrength);

      } else {
        lrRatio = 0.0f;
        udRatio = 0.0f;
      }


      // (5) COP（用 LUT + 可选单一权重归一化）
      if (seatOk && fsrTotal > 0) {
        float wx = 0.0f, wy = 0.0f;

        if (COP_USE_SINGLE_WEIGHT_NORM) {
          // ---- 推荐：同一套 debias 权重 -> 同一个 wSum 归一化（更“二维重心”）----
          float wSum = 0.0f;

          for (uint8_t i = 0; i < NUM_FSR; i++) {
            float w = fsrNorm[i]; // share

            // debias（同一个 wAdj 同时用于 X/Y）
            float wAdj = w;
            if (COP_DEBIAS_X_BY_LEVEL_COUNT) wAdj /= (float)copXLevelCount[i];
            if (COP_DEBIAS_Y_BY_LEVEL_COUNT) wAdj /= (float)copYLevelCount[i];

            wx   += wAdj * copXPos[i];
            wy   += wAdj * copYPos[i];
            wSum += wAdj;
          }

          if (wSum > 0.0f) {
            copXRatio = wx / wSum;
            copYRatio = wy / wSum;
          } else {
            copXRatio = 0.0f;
            copYRatio = 0.0f;
          }

        } else {
          // ---- 保留你原来的“X/Y 分别归一化”逻辑（但用 LUT 提速）----
          float wSumX = 0.0f;
          float wSumY = 0.0f;

          for (uint8_t i = 0; i < NUM_FSR; i++) {
            float w = fsrNorm[i];

            float wX = w;
            if (COP_DEBIAS_X_BY_LEVEL_COUNT) wX /= (float)copXLevelCount[i];
            wx    += wX * copXPos[i];
            wSumX += wX;

            float wY = w;
            if (COP_DEBIAS_Y_BY_LEVEL_COUNT) wY /= (float)copYLevelCount[i];
            wy    += wY * copYPos[i];
            wSumY += wY;
          }

          copXRatio = (wSumX > 0.0f) ? (wx / wSumX) : 0.0f;
          copYRatio = (wSumY > 0.0f) ? (wy / wSumY) : 0.0f;
        }

        // 兜底夹紧（浮点误差）
        copXRatio = clamp01(copXRatio);
        copYRatio = clamp01(copYRatio);

        if (OUTPUT_COP_AS_MM) {
          copXOut = copXRatio * CUSHION_WIDTH_MM;
          copYOut = copYRatio * CUSHION_LENGTH_MM;
        } else {
          copXOut = copXRatio;
          copYOut = copYRatio;
        }
      } else {
        copXRatio = 0.0f; copYRatio = 0.0f;
        copXOut   = 0.0f; copYOut   = 0.0f;
      }



      // (6) tof_use / tof_gated
      if (seatOk) {
        float copXCentered = copXRatio - 0.5f;
        float s1 = fabs(copXCentered);
        float s2 = 0.5f * fabs(leftShare - rightShare);
        float lateralStrength = (s1 > s2) ? s1 : s2;

        bool tofValid = (tofNorm >= 0.0f);
        if (tofValid && (lateralStrength < TOF_LATERAL_THR)) {
          tofUse = 1.0f;
          tofGated = tofNorm;
        } else {
          tofUse = 0.0f;
          tofGated = 0.5f;
        }
      } else {
        tofUse = 0.0f;
        tofGated = 0.5f;
      }

      uint32_t t_ms = windowIndex * WINDOW_INTERVAL_MS;

      // 输出模式切换
      if (RECORD_MODE) {
        // 记录模式：只写 CSV 到 USB
        writeCsvLine(SerialPort, t_ms, fsrNorm, tofNorm, tofGated, tofUse,
                     fsrStd, lrRatio, udRatio, copXOut, copYOut, CSV_DECIMALS);
      } else {
        // 真实模式：只有 seatOk 才发 I/R（无人坐时 ESP32 已停止推理，手机 UI 看 S,0）
        if (seatOk) {
          writeLineSerial1_I(Serial1, fsrNorm, tofGated, tofUse, copXOut, copYOut);
          writeLineSerial1_R(Serial1, fsrNorm, tofGated, tofUse, lrRatio, udRatio, copXOut, copYOut, CSV_DECIMALS);
        }
      }

      windowIndex++;
      resetWindowAcc();
      nextOutMs += WINDOW_INTERVAL_MS;

      if (enterPause) {
        enterSeatPaused();
        bailForPause = true;
      }
    }

    // 2) 采样事件
    else {
      if (syncSamples < MAX_SAMPLES_PER_WINDOW) {
        // 【关键修改】按端口扫描，然后根据 FSR_ID_AT_PORT 映射写入 fsrSum[逻辑编号]
        for (uint8_t p = 0; p < NUM_FSR; p++) {
          int8_t id = FSR_ID_AT_PORT[p];
          if (id < 0 || id >= (int8_t)NUM_FSR) continue; // 理论上不会发生（setup 已校验）

          uint16_t v = (uint16_t)analogRead(FSR_PORTS[p]);
          fsrSum[(uint8_t)id] += v;
          fsrSumSq[(uint8_t)id] += (uint32_t)v * (uint32_t)v;
        }

        updateToFLastMmNonBlocking();
        uint16_t d = tofHasLast ? tofLastMm : 0;
        tofWin[syncSamples] = d;

        syncSamples++;
      }
      nextSampleMs += SAMPLE_INTERVAL_MS;
    }

    if (bailForPause) break;
    now = millis();
  }
}
