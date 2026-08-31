#!/usr/bin/env python3
"""
hip_abad_align_test.py  -  fast, isolated testing of the hip ab/adduction
axis. ONE sensor (thigh), flexion calibrated once, then loop on ONLY the
ab/adduction swing -- same fast-iteration pattern as ankle_align_test.py,
adapted to a single segment finding its OWN second axis rather than two
segments checking mutual alignment.

SCOPE, DELIBERATELY NARROW (per the current plan): ab/adduction only.
No internal/external rotation, no pelvis. One sensor, one new axis.

WHAT THIS CHECKS, AND WHY THERE IS NO SIMPLE PASS/FAIL LINE
----------------------------------------------------------------
Unlike shank-vs-toe (two independent sensors, so "do they agree" is a
clean yes/no), this is one sensor finding a second axis on itself. There
is no second, independent measurement to check it against. Two different
diagnostics are reported instead, and they mean different things:

  svd_ratio           Was THIS swing itself clean (one dominant direction,
                      not multiple things happening at once)? Low here
                      means the swing was contaminated -- try again.

  vs. math-derived X   How far the measured axis is from Y-cross-Z (what
                      you'd get by just assuming perfect orthogonality
                      with flexion, no second swing needed). This is
                      informational, NOT a pass/fail: real hip anatomy is
                      not perfectly orthogonal, so some disagreement here
                      can be CORRECT, not an error. Validated against
                      synthetic ground truth: when the true axis is
                      genuinely tilted 15 deg from "clean" orthogonality,
                      the measured axis recovers it exactly while the
                      math-derived one is measurably wrong. Don't chase
                      agreement with X for its own sake.

  vs. previous attempt Repeatability across attempts is the more useful
                      signal when there's no independent cross-check:
                      if repeated swings keep finding roughly the same
                      direction, that's real evidence it's measuring
                      something reproducible, not noise.

USAGE
    python hip_abad_align_test.py --thigh-id 6
"""

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation

from imu_receiver import UdpImuCallback
from mujoco_leg_stream import LatestQuatState, calibrate_segment, resolve_segment_sign
from rebait_calibration_adapter import _average_rotation


def main():
    ap = argparse.ArgumentParser(description="Fast hip ab/adduction axis testing loop.")
    ap.add_argument("--thigh-id", type=int, required=True)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--gravity-s", type=float, default=5.0)
    ap.add_argument("--rotation-s", type=float, default=10.0)
    a = ap.parse_args()

    thigh_dev = f"imu{a.thigh_id:02d}"

    cb = UdpImuCallback(port=a.port, expected_ids=(a.thigh_id,))
    cb.enable()
    cb.attach()
    state = LatestQuatState(cb)

    print("Waiting for thigh sensor ...")
    while not state.ready(thigh_dev):
        state.poll()
        time.sleep(0.05)
    print(f"  thigh ({thigh_dev}) streaming.\n")

    thigh_fc = calibrate_segment("thigh", state, thigh_dev, a.gravity_s, a.rotation_s)
    resolve_segment_sign("thigh", state, thigh_dev, thigh_fc)
    print("\nFlexion axis calibrated once -- this is the already-validated")
    print("pathway, unchanged. Now retrying the AB/ADDUCTION swing only.\n")

    math_derived_X = thigh_fc.axes["X"].copy()
    prev_axis = None
    attempt = 0

    while True:
        attempt += 1
        print(f"\n{'=' * 60}")
        print(f"ATTEMPT {attempt}")
        print(f"{'=' * 60}")
        print("Swing your leg OUT to the side and back IN -- a frontal-")
        print("plane motion, not forward/back. Keep your knee fairly")
        print("straight and try not to rotate or lean.")
        input(f"Press Enter, then swing for {a.rotation_s:.0f}s: ")

        gyro_samples = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < a.rotation_s:
            state.poll()
            acc, gyro = state.gravity_gyro(thigh_dev)
            if gyro is not None:
                gyro_samples.append(gyro)
            time.sleep(0.002)

        try:
            axis, svd_ratio = thigh_fc.find_second_axis(gyro_samples)
        except ValueError as e:
            print(f"*** {e}")
            again = input("\nTry again? (y/n): ").strip().lower()
            if again != "y":
                break
            continue

        print(f"\nsvd_ratio: {svd_ratio:.2f} (want > ~3; this is about "
              f"whether THIS swing was clean, not whether the direction "
              f"found is anatomically correct)")

        angle_vs_math = np.degrees(np.arccos(np.clip(abs(np.dot(axis, math_derived_X)), 0, 1)))
        print(f"vs. math-derived X (Y cross Z): {angle_vs_math:.1f} deg "
              f"(informational -- disagreement can be correct anatomy, "
              f"not necessarily an error)")

        if prev_axis is not None:
            angle_vs_prev = np.degrees(np.arccos(np.clip(abs(np.dot(axis, prev_axis)), 0, 1)))
            print(f"vs. previous attempt: {angle_vs_prev:.1f} deg "
                  f"(this IS a meaningful check -- repeated swings finding "
                  f"the same direction is real evidence of reproducibility)")
        prev_axis = axis

        # Quick sanity readout: recover the angle for the swing just performed,
        # zeroed at its own start -- not a real calibration, just a feel-check
        # that projecting onto this axis gives sane-looking numbers.
        print(f"\naxis (sensor frame): [{axis[0]:+.3f}, {axis[1]:+.3f}, {axis[2]:+.3f}]")

        again = input("\nTry the ab/adduction swing again? (y/n): ").strip().lower()
        if again != "y":
            break

    print(f"\n{attempt} attempt(s) this session.")
    print("No single number here means 'done' the way the ankle's 15 deg")
    print("threshold did -- judge by: good svd_ratio, AND repeated attempts")
    print("agreeing with each other. Once you have that, this axis is ready")
    print("to bring into the real calibration pipeline.")

    cb.detach()
    cb.close()


if __name__ == "__main__":
    main()