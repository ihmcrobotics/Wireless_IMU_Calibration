#!/usr/bin/env python3
"""
record_knee_to_opensim.py  -  record calibrated knee flexion angle over
time, export as a graph (immediate, no extra software needed) and an
OpenSim-compatible .mot motion file (for loading into OpenSim itself).

SCOPE: knee only (thigh + shank), matching the explicit ask. No ankle, no
hip -- extending to those is the same pattern, not built here.

REUSES the already-validated calibration pathway (calibrate_segment,
resolve_segment_sign, relative_delta, project_angle) from
mujoco_leg_stream.py / rebait_calibration_adapter.py -- nothing about how
the angle itself is computed is new here.

WHAT IS NEW: full-fidelity RECORDING instead of live streaming. The live
MuJoCo viewer deliberately drains-and-keeps-only-the-latest sample each
frame -- correct for live viewing, wrong for a motion recording, since it
would discard most samples and produce a jerky, undersampled curve. This
keeps EVERY sample from BOTH sensors, in ONE loop, SIMULTANEOUSLY -- an
earlier draft of this script recorded thigh, then shank, one after the
other, which would have captured two DIFFERENT time windows of motion
instead of one. Caught before it was used, not after.

ALIGNMENT: thigh and shank still arrive independently over UDP even when
recorded in the same loop (different packet rates, different arrival
jitter). For each thigh sample, the nearest-in-time shank sample is used
to compute that instant's knee angle. Validated against synthetic ground
truth: recovers a known knee-flexion curve to well under 1 deg error even
with realistic ~1ms inter-stream timing jitter.

OUTPUT FILES (given --out-prefix session1):
    session1_knee.mot   OpenSim Storage format (time, knee_angle_r)
    session1_knee.csv   same data, for easy reuse/inspection
    session1_knee.png   knee angle vs time, rendered directly

I have NOT tested loading the .mot file into OpenSim itself -- OpenSim is
not installed in the environment this was built in. The format has been
validated by writing it and round-tripping it back (header row/column
counts match the actual data exactly, every value parses correctly), but
confirm it actually opens in your copy of OpenSim before trusting it
further.

USAGE
    python record_knee_to_opensim.py --thigh-id 6 --shank-id 5 --record-seconds 20 --out-prefix session1
"""

import argparse
import csv
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imu_receiver import UdpImuCallback
from mujoco_leg_stream import LatestQuatState, calibrate_segment, resolve_segment_sign
from rebait_calibration_adapter import relative_delta, _average_rotation, _wxyz_to_rot


def write_mot(path, time_s, columns: dict):
    """OpenSim Storage (.mot) file. columns: {name: array}, same length as time_s."""
    n_rows = len(time_s)
    col_names = list(columns.keys())
    n_cols = 1 + len(col_names)
    with open(path, "w") as f:
        f.write(f"{path.split('/')[-1]}\n")
        f.write("version=1\n")
        f.write(f"nRows={n_rows}\n")
        f.write(f"nColumns={n_cols}\n")
        f.write("inDegrees=yes\n")
        f.write("endheader\n")
        f.write("time\t" + "\t".join(col_names) + "\n")
        for i in range(n_rows):
            row = [f"{time_s[i]:.6f}"] + [f"{columns[c][i]:.6f}" for c in col_names]
            f.write("\t".join(row) + "\n")


def align_nearest(ref_t, other_t, other_vals):
    """For each timestamp in ref_t, return the value from other_vals whose
    timestamp is closest. other_t must be sorted ascending."""
    idx = np.searchsorted(other_t, ref_t)
    idx = np.clip(idx, 1, len(other_t) - 1)
    left, right = idx - 1, idx
    use_left = np.abs(other_t[left] - ref_t) < np.abs(other_t[right] - ref_t)
    chosen = np.where(use_left, left, right)
    return [other_vals[i] for i in chosen]


def record_both_simultaneously(state: LatestQuatState, thigh_dev, shank_dev, duration_s):
    """ONE loop, recording every sample from BOTH devices as they arrive --
    not one device fully, then the other. That would capture two different
    time windows of motion instead of one shared recording."""
    thigh_t, thigh_q = [], []
    shank_t, shank_q = [], []
    last_thigh_seq = last_shank_seq = None
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration_s:
        state.poll()
        pkt = state.latest.get(thigh_dev)
        if pkt is not None and pkt["sequence"] != last_thigh_seq:
            last_thigh_seq = pkt["sequence"]
            thigh_t.append(time.monotonic() - t0)
            thigh_q.append(_wxyz_to_rot(pkt["quat"]))
        pkt = state.latest.get(shank_dev)
        if pkt is not None and pkt["sequence"] != last_shank_seq:
            last_shank_seq = pkt["sequence"]
            shank_t.append(time.monotonic() - t0)
            shank_q.append(_wxyz_to_rot(pkt["quat"]))
        time.sleep(0.0005)
    return (np.array(thigh_t), thigh_q), (np.array(shank_t), shank_q)


def main():
    ap = argparse.ArgumentParser(description="Record knee flexion angle, export graph + OpenSim .mot.")
    ap.add_argument("--thigh-id", type=int, required=True)
    ap.add_argument("--shank-id", type=int, required=True)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--gravity-s", type=float, default=5.0)
    ap.add_argument("--rotation-s", type=float, default=10.0)
    ap.add_argument("--record-seconds", type=float, default=20.0)
    ap.add_argument("--out-prefix", default="knee_recording")
    a = ap.parse_args()

    thigh_dev = f"imu{a.thigh_id:02d}"
    shank_dev = f"imu{a.shank_id:02d}"

    cb = UdpImuCallback(port=a.port, expected_ids=(a.thigh_id, a.shank_id))
    cb.enable()
    cb.attach()
    state = LatestQuatState(cb)

    print("Waiting for thigh and shank sensors ...")
    while not (state.ready(thigh_dev) and state.ready(shank_dev)):
        state.poll()
        time.sleep(0.05)
    print(f"  thigh ({thigh_dev}) and shank ({shank_dev}) both streaming.\n")

    thigh_fc = calibrate_segment("thigh", state, thigh_dev, a.gravity_s, a.rotation_s)
    shank_fc = calibrate_segment("shank", state, shank_dev, a.gravity_s, a.rotation_s)
    resolve_segment_sign("thigh", state, thigh_dev, thigh_fc)
    resolve_segment_sign("shank", state, shank_dev, shank_fc)

    print("\n=== SIMULTANEOUS ZERO ===")
    print("Stand in your normal neutral pose. Press Enter, then hold")
    input(f"neutral for {a.gravity_s:.0f}s: ")
    rots_t, rots_s = [], []
    t0 = time.monotonic()
    while time.monotonic() - t0 < a.gravity_s:
        state.poll()
        r = state.rotation(thigh_dev)
        if r is not None:
            rots_t.append(r)
        r = state.rotation(shank_dev)
        if r is not None:
            rots_s.append(r)
        time.sleep(0.002)
    thigh_fc.set_zero(_average_rotation(rots_t))
    shank_fc.set_zero(_average_rotation(rots_s))

    check_rel = relative_delta(thigh_fc.get_zero(), rots_t[-1], shank_fc.get_zero(), rots_s[-1])
    check_deg = shank_fc.project_angle(check_rel, axis=2)
    print(f"knee angle at the pose just zeroed: {check_deg:+.1f} deg (should read ~0)")

    print(f"\n=== RECORDING ===")
    print(f"Move your knee through the motion you want captured now.")
    input(f"Press Enter, then move for {a.record_seconds:.0f}s: ")
    (thigh_t, thigh_q), (shank_t, shank_q) = record_both_simultaneously(
        state, thigh_dev, shank_dev, a.record_seconds)
    print(f"  recorded {len(thigh_t)} thigh samples, {len(shank_t)} shank samples")

    cb.detach()
    cb.close()

    if len(thigh_t) < 10 or len(shank_t) < 10:
        raise SystemExit("Too few samples recorded -- check the link and try again.")

    # Align: for each thigh timestamp, use the nearest shank sample
    shank_at_thigh_t = align_nearest(thigh_t, shank_t, shank_q)
    knee_deg = np.array([
        shank_fc.project_angle(
            relative_delta(thigh_fc.get_zero(), thigh_q[i],
                           shank_fc.get_zero(), shank_at_thigh_t[i]),
            axis=2)
        for i in range(len(thigh_t))
    ])

    print(f"\nknee angle range this recording: {knee_deg.min():+.1f} to {knee_deg.max():+.1f} deg")

    mot_path = f"{a.out_prefix}_knee.mot"
    csv_path = f"{a.out_prefix}_knee.csv"
    png_path = f"{a.out_prefix}_knee.png"

    write_mot(mot_path, thigh_t, {"knee_angle_r": knee_deg})

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "knee_angle_deg"])
        for t, k in zip(thigh_t, knee_deg):
            w.writerow([f"{t:.6f}", f"{k:.4f}"])

    plt.figure(figsize=(10, 5))
    plt.plot(thigh_t, knee_deg, linewidth=1.2)
    plt.xlabel("time (s)")
    plt.ylabel("knee flexion angle (deg)")
    plt.title("Knee flexion angle over time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)

    print(f"\nWrote:")
    print(f"  {mot_path}  (OpenSim Storage format)")
    print(f"  {csv_path}  (raw data)")
    print(f"  {png_path}  (graph)")
    print(f"\nNOTE: .mot loading into OpenSim itself has not been tested --")
    print(f"OpenSim was not available where this was built. Confirm it")
    print(f"opens correctly in your copy before relying on it.")


if __name__ == "__main__":
    main()