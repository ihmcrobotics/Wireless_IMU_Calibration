// =============================================================================
//  ReBait_imu_firemware.ino  -  ESP32 + MPU9250 wireless IMU node
//
//  Revised for ReBAIT-style processing and IMU-vs-mocap validation.
//
//  WHAT CHANGED vs. the original
//  -----------------------------
//   1. Sends the orientation QUATERNION (required by ReBAIT's calibration
//      chain) and gravity-compensated LINEAR ACCELERATION.
//   2. Sends MAGNETOMETER data, so magnetic disturbance in the capture volume
//      can be mapped, and so alternative fusion filters remain testable
//      offline from the raw signals.
//   3. Outputs SI units (rad/s, m/s^2) to match what ReBAIT's DataCollector
//      expects. The original sent deg/s and g, which silently breaks the
//      zero-velocity threshold (W_LIM = 45*pi/180 rad/s).
//   4. Magnetometer calibration is FROZEN to constants instead of being
//      re-run every boot, so between-session variability is removed.
//   5. Accel/gyro calibration happens after a thermal warm-up, because MEMS
//      gyro bias moves substantially in the first minutes after power-up.
//   6. 64-bit rollover-safe timestamps (micros() wraps at ~71.6 min).
//   7. Non-blocking WiFi reconnect, so a dropout doesn't silently end a trial.
//   8. FRAME SELF-TEST mode - see the note below. Run this before trusting
//      any calibration result.
//
//  >>> IMPORTANT: SENSOR FRAME vs. FILTER FRAME <<<
//  The MPU9250 library does NOT feed the raw axes to its quaternion filter.
//  It remaps them (gyro as (gx, -gy, -gz), accel with a sign inversion, mag
//  as (my, -mx, mz)) to reach an aircraft-style Z-down convention. That means
//  the quaternion is expressed in a DIFFERENT frame from getAccX()/getGyroX().
//
//  ReBAIT's math assumes the quaternion and the vectors it rotates share one
//  frame: a_world = q * a_body * conj(q). If they don't, every downstream
//  angle is wrong in a way that looks like sensor noise.
//
//  REMAP_TO_FILTER_FRAME below applies a 180-degree rotation about X to the
//  accel/gyro/linear-accel vectors so they match the quaternion. This is my
//  best reading of the library source, but VERIFY IT EMPIRICALLY rather than
//  trusting it: set SELF_TEST_ON_BOOT to 1, then slowly tumble the sensor
//  through many orientations while watching the serial output. World-frame
//  gravity must stay CONSTANT (magnitude ~9.81, pointing along one axis).
//  If it wanders as you rotate, the frame mapping or the quaternion direction
//  convention is wrong - the self-test prints both conventions so you can see
//  which one holds still.
// =============================================================================

#include "MPU9250.h"
#include <WiFi.h>
#include <WiFiUdp.h>

// ----------------------------------------------------------------- identity
const uint8_t IMU_ID = 5;          // 5 = right shank. Set per sensor.

// ----------------------------------------------------------------- network
const char*    SSID       = "IMU Testing";
const char*    PASSWORD   = "12345678";
const char*    JETSON_IP  = "192.168.137.1";
const uint16_t UDP_PORT   = 5000;

// ----------------------------------------------------------------- options
#define OUTPUT_SI               1   // 1 = rad/s and m/s^2 (what ReBAIT wants)
#define REMAP_TO_FILTER_FRAME   1   // 1 = align vectors with the quaternion frame
#define SELF_TEST_ON_BOOT       0   // 1 = print world-frame gravity, do not stream
#define MAG_CALIBRATION_MODE    0   // 1 = run figure-eight cal and print values

const uint32_t WARMUP_SECONDS      = 180;   // thermal soak before gyro bias cal
const float    MAG_DECLINATION_DEG = 0.0f;  // look up for your site; 0 = raw magnetic

// ------------------------------------------------- frozen mag calibration
// Run once with MAG_CALIBRATION_MODE 1, copy the printed numbers here, then
// set MAG_CALIBRATION_MODE back to 0. Do this PER SENSOR - these values are
// specific to one board and its surrounding hardware.
const float MAG_BIAS_X  = 0.0f;   // mG
const float MAG_BIAS_Y  = 0.0f;
const float MAG_BIAS_Z  = 0.0f;
const float MAG_SCALE_X = 1.0f;
const float MAG_SCALE_Y = 1.0f;
const float MAG_SCALE_Z = 1.0f;

// ----------------------------------------------------------------- constants
static const float G_TO_MS2   = 9.80665f;
static const float DEG_TO_RADF = 0.017453292519943295f;

MPU9250  mpu;
WiFiUDP  udp;

// ----------------------------------------------------------------- packet
// Python receiver format string:  '<BBHIQ19f'   (92 bytes)
//   B  imu_id
//   B  flags
//   H  reserved
//   I  sequence
//   Q  device_time_us
//   19f  gx gy gz  ax ay az  lax lay laz  mx my mz  qw qx qy qz  roll pitch yaw
struct __attribute__((packed)) IMUPacket {
  uint8_t  imu_id;
  uint8_t  flags;            // bit0 accel/gyro cal done, bit1 mag cal frozen
  uint16_t reserved;
  uint32_t sequence;
  uint64_t device_time_us;

  float gx, gy, gz;          // angular velocity      (rad/s if OUTPUT_SI)
  float ax, ay, az;          // acceleration INCLUDING gravity (m/s^2 if SI)
  float lax, lay, laz;       // acceleration WITHOUT gravity   (m/s^2 if SI)
  float mx, my, mz;          // magnetic field, milliGauss
  float qw, qx, qy, qz;      // orientation quaternion
  float roll, pitch, yaw;    // degrees - convenience only, redundant with q
};

uint32_t packetSequence = 0;
uint8_t  packetFlags    = 0;
bool     selfTestMode   = SELF_TEST_ON_BOOT;

// ----------------------------------------------------------------- helpers

// micros() is 32-bit and wraps every ~71.6 minutes. Accumulate wraps so the
// Jetson never has to unwrap, and long trials stay monotonic.
uint64_t micros64() {
  static uint32_t last  = 0;
  static uint32_t wraps = 0;
  uint32_t now = micros();
  if (now < last) wraps++;
  last = now;
  return ((uint64_t)wraps << 32) | (uint64_t)now;
}

// Rotate a sensor-frame vector into the frame the quaternion filter uses.
// 180 degrees about X:  (x, y, z) -> (x, -y, -z)
inline void toFilterFrame(float& x, float& y, float& z) {
#if REMAP_TO_FILTER_FRAME
  y = -y;
  z = -z;
#else
  (void)x; (void)y; (void)z;
#endif
}

void printCalibration() {
  Serial.println(F("< calibration parameters >"));

  Serial.print(F("accel bias [mg]: "));
  Serial.print(mpu.getAccBiasX() * 1000.f / (float)MPU9250::CALIB_ACCEL_SENSITIVITY);
  Serial.print(F(", "));
  Serial.print(mpu.getAccBiasY() * 1000.f / (float)MPU9250::CALIB_ACCEL_SENSITIVITY);
  Serial.print(F(", "));
  Serial.println(mpu.getAccBiasZ() * 1000.f / (float)MPU9250::CALIB_ACCEL_SENSITIVITY);

  Serial.print(F("gyro bias [deg/s]: "));
  Serial.print(mpu.getGyroBiasX() / (float)MPU9250::CALIB_GYRO_SENSITIVITY);
  Serial.print(F(", "));
  Serial.print(mpu.getGyroBiasY() / (float)MPU9250::CALIB_GYRO_SENSITIVITY);
  Serial.print(F(", "));
  Serial.println(mpu.getGyroBiasZ() / (float)MPU9250::CALIB_GYRO_SENSITIVITY);

  Serial.print(F("mag bias [mG]: "));
  Serial.print(mpu.getMagBiasX()); Serial.print(F(", "));
  Serial.print(mpu.getMagBiasY()); Serial.print(F(", "));
  Serial.println(mpu.getMagBiasZ());

  Serial.print(F("mag scale []: "));
  Serial.print(mpu.getMagScaleX()); Serial.print(F(", "));
  Serial.print(mpu.getMagScaleY()); Serial.print(F(", "));
  Serial.println(mpu.getMagScaleZ());

  Serial.println(F("--------------------------------"));
  Serial.println(F("Copy the mag values into MAG_BIAS_* / MAG_SCALE_* above,"));
  Serial.println(F("then set MAG_CALIBRATION_MODE back to 0."));
  Serial.println(F("--------------------------------"));
}

// Rotate v by quaternion q. Returns q * v * conj(q) if forward, else conj(q) * v * q.
void rotateByQuat(float qw, float qx, float qy, float qz,
                  float vx, float vy, float vz,
                  bool forward,
                  float& ox, float& oy, float& oz) {
  if (!forward) { qx = -qx; qy = -qy; qz = -qz; }
  // t = 2 * (qvec x v)
  float tx = 2.0f * (qy * vz - qz * vy);
  float ty = 2.0f * (qz * vx - qx * vz);
  float tz = 2.0f * (qx * vy - qy * vx);
  ox = vx + qw * tx + (qy * tz - qz * ty);
  oy = vy + qw * ty + (qz * tx - qx * tz);
  oz = vz + qw * tz + (qx * ty - qy * tx);
}

// Frame consistency check. Tumble the sensor slowly. Whichever column stays
// constant at ~9.81 along a single axis is the correct convention; if NEITHER
// stays constant, REMAP_TO_FILTER_FRAME is wrong.
void runSelfTest() {
  float ax = mpu.getAccX(), ay = mpu.getAccY(), az = mpu.getAccZ();
  toFilterFrame(ax, ay, az);
#if OUTPUT_SI
  ax *= G_TO_MS2; ay *= G_TO_MS2; az *= G_TO_MS2;
#endif

  float qw = mpu.getQuaternionW(), qx = mpu.getQuaternionX();
  float qy = mpu.getQuaternionY(), qz = mpu.getQuaternionZ();

  float fx, fy, fz, bx, by, bz;
  rotateByQuat(qw, qx, qy, qz, ax, ay, az, true,  fx, fy, fz);
  rotateByQuat(qw, qx, qy, qz, ax, ay, az, false, bx, by, bz);

  Serial.print(F("q*a*q~ [")); Serial.print(fx, 2); Serial.print(F(", "));
  Serial.print(fy, 2); Serial.print(F(", ")); Serial.print(fz, 2);
  Serial.print(F("]   q~*a*q ["));
  Serial.print(bx, 2); Serial.print(F(", "));
  Serial.print(by, 2); Serial.print(F(", ")); Serial.print(bz, 2);
  Serial.print(F("]   |a| ")); Serial.println(sqrtf(ax*ax + ay*ay + az*az), 2);
}

void connectWiFi(bool blocking) {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);            // latency and jitter matter more than power
  WiFi.begin(SSID, PASSWORD);
  if (!blocking) return;

  Serial.print(F("Connecting to WiFi"));
  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 30000) {
    delay(500);
    Serial.print('.');
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nConnected! IP: " + WiFi.localIP().toString());
  } else {
    Serial.println(F("\nWiFi timeout - will keep retrying in loop()."));
  }
}

void handleSerial() {
  if (!Serial.available()) return;
  char c = Serial.read();
  switch (c) {
    case 'p': printCalibration(); break;
    case 's': selfTestMode = !selfTestMode;
              Serial.println(selfTestMode ? F("self-test ON") : F("self-test OFF"));
              break;
    case 'z': packetSequence = 0; Serial.println(F("sequence reset")); break;
    default: break;
  }
}

// ----------------------------------------------------------------- setup
void setup() {
  Serial.begin(115200);
  delay(200);
  Wire.begin();
  Wire.setClock(400000);

  Serial.print(F("\n=== IMU node, ID ")); Serial.print(IMU_ID); Serial.println(F(" ==="));

  connectWiFi(true);

  MPU9250Setting setting;
  setting.accel_fs_sel     = ACCEL_FS_SEL::A4G;
  setting.gyro_fs_sel      = GYRO_FS_SEL::G500DPS;
  setting.mag_output_bits  = MAG_OUTPUT_BITS::M16BITS;
  setting.fifo_sample_rate = FIFO_SAMPLE_RATE::SMPL_500HZ;
  setting.gyro_fchoice     = 0x00;
  setting.gyro_dlpf_cfg    = GYRO_DLPF_CFG::DLPF_184HZ;
  setting.accel_fchoice    = 0x01;
  setting.accel_dlpf_cfg   = ACCEL_DLPF_CFG::DLPF_218HZ_1;

  if (!mpu.setup(0x68, setting)) {
    while (1) {
      Serial.println(F("MPU9250 not found!"));
      delay(1000);
    }
  }

  mpu.selectFilter(QuatFilterSel::MADGWICK);
  // mpu.setFilterIterations(5);   // more iterations = faster convergence, more CPU.
                                   // If you enable this, keep it identical across
                                   // ALL sensors and ALL sessions.
  mpu.setMagneticDeclination(MAG_DECLINATION_DEG);

  // --- thermal warm-up -------------------------------------------------
  // Gyro bias moves as the die heats. Calibrating cold gives you a bias for a
  // temperature the sensor will not be at during the trial.
  Serial.print(F("Thermal warm-up: "));
  Serial.print(WARMUP_SECONDS);
  Serial.println(F(" s. Leave the sensor powered, still and level."));
  for (uint32_t s = WARMUP_SECONDS; s > 0; --s) {
    if (s % 15 == 0 || s <= 5) { Serial.print(s); Serial.println(F(" s...")); }
    mpu.update();               // keep the filter running so it converges too
    delay(1000);
  }

  // --- accel / gyro bias ------------------------------------------------
  Serial.println(F("Accel/gyro calibration - do not touch the sensor..."));
  delay(1000);
  mpu.calibrateAccelGyro();
  packetFlags |= 0x01;

  // --- magnetometer -----------------------------------------------------
#if MAG_CALIBRATION_MODE
  Serial.println(F("MAG CALIBRATION MODE."));
  Serial.println(F("Wave and TUMBLE the sensor through all orientations now -"));
  Serial.println(F("figure-eights while rotating, not sliding it flat."));
  delay(2000);
  mpu.calibrateMag();
  printCalibration();
  Serial.println(F("Calibration mode complete. Not streaming. Reflash with"));
  Serial.println(F("MAG_CALIBRATION_MODE 0 and the values pasted in."));
  while (1) { handleSerial(); delay(100); }
#else
  mpu.setMagBias(MAG_BIAS_X, MAG_BIAS_Y, MAG_BIAS_Z);
  mpu.setMagScale(MAG_SCALE_X, MAG_SCALE_Y, MAG_SCALE_Z);
  packetFlags |= 0x02;
  Serial.println(F("Magnetometer calibration loaded from constants."));
  if (MAG_BIAS_X == 0.0f && MAG_BIAS_Y == 0.0f && MAG_BIAS_Z == 0.0f) {
    Serial.println(F("*** WARNING: mag bias is all zeros. Yaw will be unreliable."));
    Serial.println(F("*** Run once with MAG_CALIBRATION_MODE 1 first."));
  }
#endif

  printCalibration();
  udp.begin(UDP_PORT);

  if (selfTestMode) {
    Serial.println(F("\nSELF-TEST: tumble the sensor slowly."));
    Serial.println(F("One of the two bracketed vectors should stay CONSTANT"));
    Serial.println(F("at about 9.81 along a single axis. Not streaming UDP."));
  } else {
    Serial.println(F("\nStreaming. Serial: 'p' calibration, 's' self-test, 'z' reset seq."));
  }
}

// ----------------------------------------------------------------- loop
void loop() {
  handleSerial();

  // --- non-blocking WiFi recovery ---------------------------------------
  static uint32_t lastWifiCheck = 0;
  if (millis() - lastWifiCheck > 2000) {
    lastWifiCheck = millis();
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println(F("WiFi lost - reconnecting"));
      WiFi.disconnect();
      connectWiFi(false);
    }
  }

  if (!mpu.update()) return;

  // --- self-test path ----------------------------------------------------
  if (selfTestMode) {
    static uint32_t lastPrint = 0;
    if (millis() - lastPrint > 200) { lastPrint = millis(); runSelfTest(); }
    return;
  }

  // --- assemble packet ---------------------------------------------------
  IMUPacket pkt;
  pkt.imu_id         = IMU_ID;
  pkt.flags          = packetFlags;
  pkt.reserved       = 0;
  pkt.sequence       = packetSequence++;
  pkt.device_time_us = micros64();

  float gx = mpu.getGyroX(),      gy = mpu.getGyroY(),      gz = mpu.getGyroZ();       // deg/s
  float ax = mpu.getAccX(),       ay = mpu.getAccY(),       az = mpu.getAccZ();        // g
  float lx = mpu.getLinearAccX(), ly = mpu.getLinearAccY(), lz = mpu.getLinearAccZ();  // g

  toFilterFrame(gx, gy, gz);
  toFilterFrame(ax, ay, az);
  toFilterFrame(lx, ly, lz);

#if OUTPUT_SI
  gx *= DEG_TO_RADF; gy *= DEG_TO_RADF; gz *= DEG_TO_RADF;
  ax *= G_TO_MS2;    ay *= G_TO_MS2;    az *= G_TO_MS2;
  lx *= G_TO_MS2;    ly *= G_TO_MS2;    lz *= G_TO_MS2;
#endif

  pkt.gx = gx;  pkt.gy = gy;  pkt.gz = gz;
  pkt.ax = ax;  pkt.ay = ay;  pkt.az = az;
  pkt.lax = lx; pkt.lay = ly; pkt.laz = lz;

  // Magnetometer, remapped the same way the filter remaps it: (my, -mx, mz)
  pkt.mx =  mpu.getMagY();
  pkt.my = -mpu.getMagX();
  pkt.mz =  mpu.getMagZ();

  pkt.qw = mpu.getQuaternionW();
  pkt.qx = mpu.getQuaternionX();
  pkt.qy = mpu.getQuaternionY();
  pkt.qz = mpu.getQuaternionZ();

  pkt.roll  = mpu.getRoll();
  pkt.pitch = mpu.getPitch();
  pkt.yaw   = mpu.getYaw();

  udp.beginPacket(JETSON_IP, UDP_PORT);
  udp.write((uint8_t*)&pkt, sizeof(IMUPacket));
  udp.endPacket();

  // --- rate report -------------------------------------------------------
  // Measure the ACTUAL output rate. Do not assume it equals the FIFO rate;
  // preallocation on the Jetson and any resampling depend on this number.
  static uint32_t lastReport = 0, lastSeq = 0;
  if (millis() - lastReport >= 5000) {
    float hz = (packetSequence - lastSeq) * 1000.0f / (millis() - lastReport);
    Serial.print(F("rate ")); Serial.print(hz, 1);
    Serial.print(F(" Hz   seq ")); Serial.println(packetSequence);
    lastReport = millis();
    lastSeq    = packetSequence;
  }
}
