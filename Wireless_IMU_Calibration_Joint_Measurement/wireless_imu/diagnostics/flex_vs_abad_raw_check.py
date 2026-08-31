#!/usr/bin/env python3
"""
flex_vs_abad_raw_check.py  -  do BOTH swings, compare their raw PCA axes
directly to each other, with NO labeling assumptions.

WHY THIS IS DIFFERENT FROM hip_abad_align_test.py
------------------------------------------------------
That script trusted the FIRST calibration (calibrate_segment's flexion
swing) as already-correct ground truth, and compared the NEW ab/adduction
swing against axes derived from it (Y, X=Y-cross-Z). That comparison is
only as good as the assumption that the first swing was itself clean.

This script makes no such assumption. It does a plain PCA on EACH swing
independently, in the sensor's own raw frame, and reports:
  - the angle BETWEEN the two raw axes directly (the key number: want
    this LARGE -- near 90 deg means the two motions are genuinely
    producing different rotations; near 0 deg means they are landing
    on the same axis regardless of which motion was attempted)
  - each axis's angle to gravity (want each near 90 deg -- a limb swing
    should be roughly horizontal; a small angle here means something
    other than a clean horizontal swing was captured)
  - svd_ratio for each (was the swing itself single-axis, independent of
    which direction it found)

Nothing here is a real calibration and nothing gets saved/zeroed --
purely diagnostic, meant to answer one question before trusting either
swing individually: are these two motions actually distinguishable at
all, in a neutral comparison?

USAGE
    python flex_vs_abad_raw_check.py --thigh-id 6
"""

import argparse
import time

import numpy as np

from imu_receiver import UdpImuCallback
from mujoco_leg_stream import LatestQuatState


def collect_gyro(state, did, duration_s):
    samples = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration_s:
        state.poll()
        _acc, gyro = state.gravity_gyro(did)
        if gyro is not None:
            samples.append(gyro)
        time.sleep(0.002)
    return np.asarray(samples, float)


def collect_gravity(state, did, duration_s):
    samples = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration_s:
        state.poll()
        acc, _gyro = state.gravity_gyro(did)
        if acc is not None:
            samples.append(acc)
        time.sleep(0.002)
    return np.asarray(samples, float)


def raw_pca_axis(gyro_samples):
    """Plain PCA, no orthogonalization against anything, no sign fixing
    beyond a consistent convention. Returns (axis, svd_ratio)."""
    wc = gyro_samples - gyro_samples.mean(axis=0)
    U, S, Vt = np.linalg.svd(wc, full_matrices=False)
    axis = Vt[0]
    svd_ratio = float(S[0] / S[1]) if len(S) > 1 and S[1] > 1e-9 else float("inf")
    # sign convention: same peak-asymmetry heuristic as elsewhere, so
    # repeated runs are at least internally consistent
    proj = wc @ axis
    if np.percentile(proj, 5) + np.percentile(proj, 95) < 0:
        axis = -axis
    return axis, svd_ratio


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), 0, 1))))


def main():
    ap = argparse.ArgumentParser(description="Raw, unbiased flexion-vs-abduction axis check.")
    ap.add_argument("--thigh-id", type=int, required=True)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--gravity-s", type=float, default=4.0)
    ap.add_argument("--swing-s", type=float, default=10.0)
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

    input(f"Stand still for {a.gravity_s:.0f}s to capture gravity. Press Enter: ")
    g_samples = collect_gravity(state, thigh_dev, a.gravity_s)
    g_dir = g_samples.mean(axis=0)
    g_dir /= np.linalg.norm(g_dir)
    print(f"  gravity direction captured from {len(g_samples)} samples.\n")

    while True:
        print(f"\n{'=' * 60}")
        print("FORWARD / BACK swing (flexion/extension)")
        print(f"{'=' * 60}")
        input(f"Press Enter, then swing for {a.swing_s:.0f}s: ")
        flex_gyro = collect_gyro(state, thigh_dev, a.swing_s)
        flex_axis, flex_svd = raw_pca_axis(flex_gyro)
        print(f"  svd_ratio: {flex_svd:.2f}")
        print(f"  raw axis (sensor frame): [{flex_axis[0]:+.3f}, "
              f"{flex_axis[1]:+.3f}, {flex_axis[2]:+.3f}]")
        print(f"  angle to gravity: {angle_deg(flex_axis, g_dir):.1f} deg "
              f"(want near 90 -- a horizontal swing)")

        print(f"\n{'=' * 60}")
        print("SIDE TO SIDE swing (ab/adduction)")
        print(f"{'=' * 60}")
        input(f"Press Enter, then swing for {a.swing_s:.0f}s: ")
        abad_gyro = collect_gyro(state, thigh_dev, a.swing_s)
        abad_axis, abad_svd = raw_pca_axis(abad_gyro)
        print(f"  svd_ratio: {abad_svd:.2f}")
        print(f"  raw axis (sensor frame): [{abad_axis[0]:+.3f}, "
              f"{abad_axis[1]:+.3f}, {abad_axis[2]:+.3f}]")
        print(f"  angle to gravity: {angle_deg(abad_axis, g_dir):.1f} deg "
              f"(want near 90 -- a horizontal swing)")

        sep = angle_deg(flex_axis, abad_axis)
        print(f"\n{'-' * 60}")
        print(f"ANGLE BETWEEN THE TWO RAW AXES: {sep:.1f} deg")
        print(f"{'-' * 60}")
        if sep < 20:
            print("  *** Small separation. These two swings are landing on")
            print("  *** nearly the SAME axis -- whatever the physical")
            print("  *** motions were, they are not producing distinguishable")
            print("  *** rotations yet. This is the thing to fix before")
            print("  *** either swing can be trusted individually.")
        elif sep < 60:
            print("  Moderate separation. Some real difference between the")
            print("  two motions, but likely still some coupling between them.")
        else:
            print("  Good separation -- the two motions are producing")
            print("  genuinely different rotation axes.")

        again = input("\nTry both swings again? (y/n): ").strip().lower()
        if again != "y":
            break

    cb.detach()
    cb.close()


if __name__ == "__main__":
    main()