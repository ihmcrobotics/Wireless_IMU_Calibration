#!/usr/bin/env python3
"""
ankle_align_test.py  -  fast, isolated shank-vs-toe alignment testing.

WHY THIS EXISTS
------------------
Five consecutive full 4-segment sessions were run just to get five data
points on shank-toe alignment (89.7, 41.7, 37.2, 30.7, 33.4 deg -- all
bad). Each full session needs thigh AND pelvis calibrated too, plus the
viewer launched -- several minutes of setup per single ankle data point.

This script calibrates ONLY shank and toe, skips thigh/pelvis/viewer
entirely, and LOOPS so you can retry the toe's Phase 2 swing repeatedly
-- trying the mechanical-constraint approach, a different motion, whatever
-- and see the alignment number immediately after each attempt, without
re-doing shank or restarting the script.

Shank is calibrated ONCE at the start (it has been consistently reliable
paired with thigh across all five sessions). Only the TOE's calibration
repeats in the loop, since that's the only unresolved piece.

Reuses calibrate_segment(), resolve_segment_sign(), and
check_axis_alignment() directly from mujoco_full_leg_stream.py rather
than reimplementing them -- same validated logic, not a fork of it.

USAGE
    python ankle_align_test.py --shank-id 5 --toe-id 7
"""

import argparse
import time

from imu_receiver import UdpImuCallback
from mujoco_leg_stream import (
    LatestQuatState,
    calibrate_segment,
    resolve_segment_sign,
    check_axis_alignment,
)
from rebait_calibration_adapter import _average_rotation, relative_delta


def main():
    ap = argparse.ArgumentParser(description="Fast shank-vs-toe alignment loop.")
    ap.add_argument("--shank-id", type=int, required=True)
    ap.add_argument("--toe-id", type=int, required=True)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--gravity-s", type=float, default=5.0)
    ap.add_argument("--rotation-s", type=float, default=10.0)
    a = ap.parse_args()

    shank_dev = f"imu{a.shank_id:02d}"
    toe_dev = f"imu{a.toe_id:02d}"

    cb = UdpImuCallback(port=a.port, expected_ids=(a.shank_id, a.toe_id))
    cb.enable()
    cb.attach()
    state = LatestQuatState(cb)

    print("Waiting for shank and toe sensors ...")
    while not (state.ready(shank_dev) and state.ready(toe_dev)):
        state.poll()
        time.sleep(0.05)
    print(f"  shank ({shank_dev}) and toe ({toe_dev}) both streaming.\n")

    shank_fc = calibrate_segment("shank", state, shank_dev, a.gravity_s, a.rotation_s)
    resolve_segment_sign("shank", state, shank_dev, shank_fc)
    print("\nShank calibrated once. Now retrying the TOE only -- this")
    print("shouldn't need to be redone unless shank itself feels wrong.\n")

    attempt = 0
    results = []
    while True:
        attempt += 1
        print(f"\n{'=' * 60}")
        print(f"ATTEMPT {attempt}")
        print(f"{'=' * 60}")

        toe_fc = calibrate_segment("toe", state, toe_dev, a.gravity_s, a.rotation_s)
        resolve_segment_sign("toe", state, toe_dev, toe_fc,
                             motion="DORSIFLEXION (toes up)")

        # simultaneous zero, both segments together -- same discipline as
        # the full script, still required even in this isolated version
        print("\n=== SIMULTANEOUS ZERO (shank + toe) ===")
        print("Stand in your normal neutral pose. Press Enter, then hold")
        input(f"neutral for {a.gravity_s:.0f}s: ")
        rots_s, rots_toe = [], []
        t0 = time.monotonic()
        while time.monotonic() - t0 < a.gravity_s:
            state.poll()
            r = state.rotation(shank_dev)
            if r is not None:
                rots_s.append(r)
            r = state.rotation(toe_dev)
            if r is not None:
                rots_toe.append(r)
            time.sleep(0.002)
        shank_fc.set_zero(_average_rotation(rots_s))
        toe_fc.set_zero(_average_rotation(rots_toe))

        check_rel = relative_delta(shank_fc.get_zero(), rots_s[-1],
                                   toe_fc.get_zero(), rots_toe[-1])
        check_deg = toe_fc.project_angle(check_rel, axis=2)
        print(f"ankle angle at the pose just zeroed: {check_deg:+.1f} deg "
              f"(should read ~0)")

        _, align_deg = check_axis_alignment("shank", shank_fc, "toe", toe_fc)
        results.append(align_deg)

        print(f"\n{'-' * 60}")
        print(f"RESULTS SO FAR THIS SESSION: {[f'{r:.1f}' for r in results]}")
        print(f"{'-' * 60}")

        again = input("\nTry the toe swing again? (y/n): ").strip().lower()
        if again != "y":
            break

    print(f"\nFinal attempt: {results[-1]:.1f} deg alignment.")
    if results[-1] < 15.0:
        print("Under 15 deg -- usable. You can now run the full")
        print("mujoco_full_leg_stream.py session; shank/toe should hold.")
    else:
        print("Still above 15 deg. Worth trying a genuinely different")
        print("approach (mechanical constraint, or re-checking the motion")
        print("itself) rather than another unconstrained attempt.")

    cb.detach()
    cb.close()


if __name__ == "__main__":
    main()