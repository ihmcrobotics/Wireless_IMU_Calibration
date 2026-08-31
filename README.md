# Wireless IMU Calibration & Joint Angle Measurement 

**Real-time Biofeedback and Analysis using IMU Tracking**

This is a system for collecting IMU data and computing real-time biomechanical joint angles for biofeedback, clinical monitoring, and musculoskeletal simulation. It pairs low-cost wireless IMU hardware (ESP32 + MPU9250) with a gravity-plus-PCA functional calibration pipeline and streams joint angles into a MuJoCo musculoskeletal model at ~330–350 Hz with zero packet loss.


## What It Does

Strap on up to four wireless IMU sensors (pelvis, thigh, shank, toe), run a short guided calibration, and get real-time joint angles for hip flexion/extension, hip ab/adduction, hip rotation, knee flexion/extension, and ankle dorsi/plantarflexion — visualized live in a MuJoCo musculoskeletal model.

The calibration requires no external equipment (no motion capture lab, no cameras, no special jigs). A standing static hold and a few functional movements (knee swings, ab/adduction swings) are all that is needed. The calibration math is based on gravity direction and principal component analysis of angular velocity during known motions.

## Repository Layout

```
 Wireless IMU Calibraiton/
├── firmware/                  # ESP32 Arduino firmware
│   └── imu_firmware.ino
├── Wireless IMU Calibraiton & Joint Angle Measurement/
│   ├── calibration/           # Calibration math
│   │   ├── shank_calibration.py          # Single-sensor gravity + PCA calibration
│   │   ├── rebait_calibration_adapter.py # Multi-joint calibration interface
│   │   └── Joint_angle.py               # Two-sensor functional alignment
│   ├── streaming/             # Real-time streaming to MuJoCo
│   │   ├── imu_receiver.py               # UDP packet receiver and parser
│   │   ├── mujoco_leg_stream.py          # Full leg pipeline (4 sensors, 5 DOF)
│   │   ├── mujoco_knee_stream.py         # Knee-only pipeline (2 sensors)
│   │   └── record_knee_to_opensim.py     # Record joint angles to OpenSim .mot
│   ├── diagnostics/           # Link quality and calibration verification
│   │   ├── link_diag.py                  # Single-sensor link diagnostics
│   │   ├── multi_link_diag.py            # Multi-sensor simultaneous diagnostics
│   │   ├── bench_checks.py              # Accuracy benchmarks (no mocap needed)
│   │   ├── ankle_align_test.py           # Isolated ankle calibration test
│   │   ├── hip_abad_align_test.py        # Hip ab/adduction axis verification
│   │   └── flex_vs_abad_raw_check.py     # Hip Flexion vs. ab/adduction crosstalk check
│   ├── tools/                 # Utility scripts
│   │   ├── mag_fit.py                    # Magnetometer ellipsoid calibration
│   │   ├── mag_survey.py                 # Magnetic field environment survey
│   │   ├── gen_angle_block.py            # 3D-printable angle test fixture
│   │   └── mujoco_sanity_check.py        # Verify MuJoCo install (no IMU needed)
│   └── core/                  # Original ReBAIT framework (Xsens-based)
│       ├── ReBAIT.py                     # Main orchestrator
│       ├── DataCollector.py              # Data storage and processing
│       └── ...
├── templates/
│   └── custom_imu_callback_template.py   # Adapter template for other IMU hardware
├── docs/
│   └── calibration_protocol.md           # Detailed test protocol
├── examples/
│   ├── session1_knee.csv                 # Example recorded knee angles
│   └── session1_knee.mot                 # Example OpenSim motion file
├── requirements.txt
└── LICENSE                               # Apache 2.0
```

## Hardware

Each sensor node is built from inexpensive, off-the-shelf components.

**Per sensor node:**
- Seeed XIAO ESP32-C3 microcontroller
- MPU9250 (or MPU6500) IMU breakout
- FPC external antenna (Seeed A-01) — strongly recommended for reliable WiFi
- USB-C cable for programming and power

**Host computer:**
- Any machine that can run Python and MuJoCo (Windows, Linux, macOS)
- WiFi access point — a mobile hotspot or a dedicated router on a known SSID

The firmware streams 92-byte UDP quaternion packets at the IMU's native rate (~330–350 Hz). Four sensors share a single UDP port with per-packet device IDs, so no per-sensor port management is needed.

## Installation

**1. Clone the repository:**

```bash
git clone https://github.com/your-username/Wireless_IMU_Calibration.git
cd Wireless_IMU_Calibration
```

**2. Install Python dependencies:**

```bash
pip install -r requirements.txt
```

The core pipeline needs `numpy`, `scipy`, `scikit-learn`, `matplotlib`, and `mujoco`. MuJoCo 3.x installs via pip with no separate license.

**3. Flash the firmware:**

Open `firmware/imu_firmware.ino` in the Arduino IDE. Set the per-sensor constants at the top of the file:

- `IMU_ID` — unique integer per sensor (e.g., 3 = pelvis, 6 = thigh, 5 = shank, 7 = toe)
- `SSID` and `PASSWORD` — your WiFi network
- `JETSON_IP` — the IP address of the host computer on that network
- `UDP_PORT` — default 5000

Flash each board with its own `IMU_ID`, then strap it to the appropriate body segment.

**4. Obtain a MuJoCo musculoskeletal model:**

The streaming scripts look for a MuJoCo XML model with named joints (`knee_angle_r`, `hip_flexion_r`, `hip_adduction_r`, `hip_rotation_r`, `ankle_angle_r`). The [MyoSuite](https://github.com/MyoHub/myosuite) body model works. Pass the model directory with `--model-dir` if it is not auto-detected.

## Quick Start

All commands below are run from the `rebait/streaming/` directory. Make sure the sensor nodes are powered and connected to the same WiFi network as the host.

### Verify the link first

Before calibrating, confirm packets are arriving:

```bash
cd wireless_imu/diagnostics
python link_diag.py --ids 5 --seconds 10
```

You should see ~330 Hz with 0% loss. For multiple sensors simultaneously:

```bash
python multi_link_diag.py --ids 3 5 6 7 --seconds 10
```

### Two-sensor knee streaming (simplest)

```bash
cd wireless_imu/streaming
python mujoco_knee_stream.py --thigh-id 6 --shank-id 5
```

### Full leg streaming (4 sensors, 5 DOF)

```bash
python mujoco_leg_stream.py --pelvis-id 3 --thigh-id 6 --shank-id 5 --toe-id 7
```

The `--toe-id` and `--pelvis-id` flags are optional. Omit them to run with fewer sensors (knee-only with just thigh + shank, for example).

 


 

## Calibration

The streaming scripts walk you through calibration interactively. The protocol has four phases per segment:

**Phase 1 — Gravity (5 s):** Stand or sit still. The mean acceleration direction defines the gravity axis in the sensor frame.

**Phase 2 — Rotation (~20 s):** Perform seated knee swings (large amplitude, 8–10 reps). PCA on the angular velocity samples recovers the joint flexion axis in each sensor's own frame. Follow with a few walking steps if possible — symmetric swings alone can leave the flexion-sign ambiguous.

**Phase 3 — Zero (~5 s):** Stand in a neutral pose. This captures the reference orientation that all subsequent angles are measured from.

**Phase 4 — Sign check:** One deliberate, slow, known-direction flexion. This resolves sign ambiguity even if the automatic heuristic got it wrong. Do not skip this step.

For the hip ab/adduction axis, an additional swing is prompted: stand on one leg and swing the other leg out to the side and back, several times. This is calibrated on the same thigh sensor that was already calibrated for flexion.

Detailed protocols and pass/fail criteria are in `docs/calibration_protocol.md`.

## Key Design Decisions

**Gravity + PCA calibration (no magnetometer).** The joint flexion axis is found from the first principal component of gyro data during a calibration swing. Gravity gives two of three orientation DOFs for free. This combination works without a magnetically clean environment, which matters for office and clinic settings where magnetometer-based heading is unreliable.

**Quaternion packets, not Euler angles.** The firmware sends orientation as a quaternion to avoid gimbal lock and Euler-convention ambiguity. All downstream math operates on quaternions and rotation matrices.

**Single calibration, shared across joints.** The shank sensor is calibrated once and reused for both the knee joint (paired with thigh) and the ankle joint (paired with toe). Calibrating it twice independently would produce two different axis estimates with no guarantee of agreement.

**Hip ab/adduction from gravity orthogonalization, not swing PCA.** Orthogonalizing the ab/adduction PCA axis against gravity (but not against the flexion axis) is analytically equivalent to the cross product of the gravity axis and the flexion axis, making the result independent of ab/adduction swing quality. This was validated synthetically and confirmed on hardware.

**Hip rotation is drift-prone by design.** The rotation axis is nearly aligned with gravity, so there is no accelerometer-based drift correction possible for this DOF. It is treated as a short-burst relative measurement, resettable with the `z` key in the MuJoCo viewer.

## Adapting for Other IMU Hardware

ReBAIT's architecture separates hardware communication (the Callback class) from calibration and processing. To use a different IMU:

1. Copy `templates/custom_imu_callback_template.py` and implement the marked `TODO` sections for your sensor's SDK.
2. Your adapter must provide: `devIds`, `samp_freq`, `enable()`, `attach()`, `detach()`, `packetAvailable()`, `getNextPacket()`, and `close()`.
3. Each packet must return `[timestamp, free_accel, gyro, quaternion, calibrated_accel, euler]` with acceleration in m/s², angular velocity in rad/s, and quaternion in (w, x, y, z) order.

See the docstrings in `wireless_imu/streaming/imu_receiver.py` for a complete working example with the ESP32/MPU9250 hardware.

## Diagnostics

The `wireless_imu/diagnostics/` directory contains standalone tools for verifying each part of the system without a motion capture reference:

- **`link_diag.py`** — packet rate, loss, jitter, sequence gaps, and WiFi stability for a single sensor.
- **`multi_link_diag.py`** — simultaneous multi-sensor link quality on a shared UDP socket.
- **`bench_checks.py`** — gravity stability, orientation drift, magnetometer consistency, Madgwick filter response, closure error over return-to-home maneuvers, and static angle sweep linearity against a printed test fixture.
- **`ankle_align_test.py`** — isolated ankle calibration iteration (faster than full-session reruns).
- **`hip_abad_align_test.py`** / **`flex_vs_abad_raw_check.py`** — verify that hip ab/adduction and flexion axes are correctly separated.

## Recording to OpenSim

To record knee angle data for use in OpenSim:

```bash
cd wireless_imu/streaming
python record_knee_to_opensim.py --thigh-id 6 --shank-id 5 --output session.mot
```

This produces a `.mot` file compatible with OpenSim's inverse kinematics workflow, along with a `.csv` of timestamped angle data and a plot of the session.

## Citation

If you use ReBAIT in your research, please cite:

> Hoegberg, Z., Donahue, S., & Major, M. (2025). ReBAIT: Real-time Biofeedback and Analysis using IMU Tracking. Institute for Human and Machine Cognition.

## License

Apache License 2.0. See [LICENSE](LICENSE) for the full text.
