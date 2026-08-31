# Test D1: On-Leg, One Sensor, No Reference

Purpose: measure real on-body repeatability and basic signal quality with the
sensor mounted on the leg, without motion capture or a camera reference.

Important: the current one-sensor code reports shank sagittal inclination
relative to the neutral standing pose. It does not report true knee flexion,
because true knee angle needs at least a thigh sensor and a shank sensor.

## Setup

1. Put the IMU on the shank in the intended wearable location.
2. Use the same strap tension and landmark placement you expect in real use.
3. Start each recording only after the ESP32 reports that it is streaming.
4. For every calibration trial, keep the first 5 seconds as quiet standing:
   stand relaxed, weight even, knee unlocked.
5. Do not split the static and dynamic parts into separate files. The
   calibration code uses `t < 5 s` as the neutral static window.

All commands below assume you are in `C:\Users\skim\Wireless_IMU_Calibration\Onesensor_Test`.

## Part A: Re-Donning Repeatability

This is the most important D1 number. Remove and remount the sensor five times,
recalibrating each time. Each recording should be one continuous 70 second file:

- 0-5 s: quiet standing neutral pose
- 5-25 s: seated knee swings, large and clean, mostly sagittal
- 25-45 s: quiet standing again in the same neutral pose
- 45-70 s: optional short walk or a few asymmetric swings to help sign
  disambiguation

Record the five donning trials:

```powershell
python .\imu_receiver.py --seconds 70 --out d1_donning_01.npz --ids 5
python .\imu_receiver.py --seconds 70 --out d1_donning_02.npz --ids 5
python .\imu_receiver.py --seconds 70 --out d1_donning_03.npz --ids 5
python .\imu_receiver.py --seconds 70 --out d1_donning_04.npz --ids 5
python .\imu_receiver.py --seconds 70 --out d1_donning_05.npz --ids 5
```

After each capture, run:

```powershell
python .\shank_calibration.py .\d1_donning_01.npz
python .\bench_checks.py .\d1_donning_01.npz --test link
```

Repeat those two analysis commands for files `02` through `05`.

Record this table manually from the printed output:

| Donning | Standing angle, deg | PCA variance ratio |
| --- | ---: | ---: |
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |
| SD | | |
| Range | | |

Use the `neutral (static)` value from `shank_calibration.py` as the standing
angle. Use `PCA explained variance` as the PCA variance ratio.

Interpretation:

- SD is the inter-donning repeatability error floor.
- Range shows the worst spread a user may see from remounting alone.
- PCA ratio should ideally be high. If it is poor or ambiguous, add larger,
  cleaner seated knee swings to the dynamic part of the trial.

## Part B: Movement Progression

Run three separate 65 second trials. Each still begins with the 5 second neutral
standing calibration window.

### D1.1 Seated Knee Swings

Purpose: best-case large, clean motion with minimal soft-tissue artifact. If
this looks bad, walking will not improve it.

Protocol:

- 0-5 s: quiet standing neutral pose
- 5-65 s: sit and swing the knee through a comfortable large range

Commands:

```powershell
python .\imu_receiver.py --seconds 65 --out d1_seated_swings.npz --ids 5
python .\shank_calibration.py .\d1_seated_swings.npz --plot
python .\bench_checks.py .\d1_seated_swings.npz --test all
```

### D1.2 Standing Sway

Purpose: small-amplitude behavior and low-rate noise floor.

Protocol:

- 0-5 s: quiet standing neutral pose
- 5-65 s: stand and perform small natural sway or small controlled knee bends

Commands:

```powershell
python .\imu_receiver.py --seconds 65 --out d1_standing_sway.npz --ids 5
python .\shank_calibration.py .\d1_standing_sway.npz --plot
python .\bench_checks.py .\d1_standing_sway.npz --test all
```

### D1.3 Walking, Self-Selected

Purpose: realistic on-leg use with soft-tissue artifact, impact, and gait-rate
asymmetry.

Protocol:

- 0-5 s: quiet standing neutral pose
- 5-65 s: walk at a self-selected pace

Commands:

```powershell
python .\imu_receiver.py --seconds 65 --out d1_walking_self_selected.npz --ids 5
python .\shank_calibration.py .\d1_walking_self_selected.npz --plot
python .\bench_checks.py .\d1_walking_self_selected.npz --test all
```

## Pass/Fail Notes

Inspect these values first:

- Packet loss from `bench_checks.py --test link`: should be below 1%.
- Static gravity magnitude: should be near 9.81 m/s^2.
- Gravity direction wander p95: this is an upper bound on roll/pitch angle
  accuracy.
- PCA explained variance: low values mean the sagittal axis was not cleanly
  determined.
- Sign margin: if reported as ambiguous, include walking or an asymmetric
  deliberate movement in the dynamic window and verify the angle direction.

The final D1 result is not a single accuracy number. Report:

- re-donning standing angle SD and range,
- PCA ratio range across the five donnings,
- seated swing angle trace quality,
- standing sway noise/low-rate behavior,
- walking trace quality and any packet loss or sign ambiguity.
