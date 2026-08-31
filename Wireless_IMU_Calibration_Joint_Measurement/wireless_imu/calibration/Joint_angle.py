#!/usr/bin/env python3
"""
joint_angle.py  -  multi-sensor calibration and joint angle extraction.

Builds on shank_calibration.py (single-sensor ReBAIT calibration) and adds:

  1. Functional alignment between two adjacent sensors using the shared
     joint axis (PCA on angular velocity during a hinge motion).
  2. Joint flexion/extension angle from the relative orientation of two
     calibrated segments.
  3. Extension to three sensors (toe + shank + thigh) by treating each
     adjacent pair as an independent joint.

CALIBRATION PROTOCOL
--------------------
All sensors record simultaneously in ONE continuous capture.

  Static (5 s):  stand relaxed, all sensors strapped on.
  Dynamic (~25 s):
    - Seated knee swings, large amplitude, 8-10 reps.
      Both shank and thigh rotate about the SAME physical axis (the knee).
      PCA on each sensor's angular velocity recovers that axis in each
      sensor's own frame.  The rotation mapping one onto the other is the
      functional alignment.
    - Toe touches, 5-10 reps (for comparability with ReBAIT).
    - A few walking steps if possible (resolves sign unambiguously).

USAGE
    from joint_angle import calibrate_pair, joint_angle, resolve_joint_sign

    # Load a simultaneous recording
    from imu_receiver import load
    _, prox = load("cal.npz", "imu06")   # thigh (proximal)
    _, dist = load("cal.npz", "imu05")   # shank (distal)

    # Calibrate each sensor and align them
    cal = calibrate_pair(prox, dist, static_s=5.0)

    # Extract knee angle from any recording
    _, prox_trial = load("trial.npz", "imu06")
    _, dist_trial = load("trial.npz", "imu05")
    angle = joint_angle(prox_trial["quat"], dist_trial["quat"], cal)

    # Verify the sign against a known flexion
    cal = resolve_joint_sign(prox_trial["quat"], dist_trial["quat"], cal)
"""

import numpy as np
from shank_calibration import (qmul, qconj, qrot, qnorm, quat_mean,
                                calibrate as calibrate_single,
                                sagittal_angle as segment_angle)


def _pca_axis(gyro, moving_mask):
    """First principal component of angular velocity while moving."""
    g = np.asarray(gyro, float)[moving_mask]
    g = g - g.mean(axis=0)
    _, _, Vt = np.linalg.svd(g, full_matrices=False)
    sv = np.linalg.svd(g, compute_uv=False)
    axis = Vt[0]
    explained = float(sv[0] ** 2 / max(np.sum(sv ** 2), 1e-12))
    return axis, explained


def _align_axes(axis_prox, axis_dist, q_prox, q_dist):
    """
    Given the shared joint axis expressed in each sensor's body frame,
    compute the relative heading offset between the two sensors.

    Both axes should point in the same physical direction (along the joint).
    The angle between them, projected into the plane perpendicular to gravity,
    is the heading alignment that a magnetometer would otherwise provide.
    """
    # Rotate each axis into the world frame using the static calibration
    # At this point q_prox and q_dist are the neutral standing quaternions
    ax_w_prox = qrot(np.atleast_2d(q_prox), np.atleast_2d(axis_prox))[0]
    ax_w_dist = qrot(np.atleast_2d(q_dist), np.atleast_2d(axis_dist))[0]

    # The axes should be parallel (same physical joint). The dot product
    # gives the alignment quality.
    dot = float(np.dot(ax_w_prox, ax_w_dist))
    alignment_deg = float(np.degrees(np.arccos(np.clip(abs(dot), 0, 1))))

    # If they point in opposite directions, flip one
    if dot < 0:
        axis_dist = -axis_dist

    return axis_prox, axis_dist, alignment_deg


def calibrate_pair(prox, dist, static_s=5.0, verbose=True):
    """
    Calibrate a proximal-distal sensor pair (e.g. thigh-shank for knee angle).

    Parameters
    ----------
    prox : dict with keys t, quat, cal_acc, gyro  (from imu_receiver.load)
    dist : dict, same keys, from the same simultaneous recording
    static_s : float, seconds of static standing at the start

    Returns
    -------
    dict with:
        prox_cal, dist_cal : single-sensor calibration dicts
        joint_axis_prox, joint_axis_dist : shared axis in each frame
        alignment_deg : how well the two axes agreed (should be < ~10)
        joint_q_offset : relative quaternion between the two neutral poses
    """
    if verbose:
        print("=" * 60)
        print("PROXIMAL SENSOR CALIBRATION")
        print("=" * 60)
    prox_cal = calibrate_single(prox["t"], prox["quat"], prox["cal_acc"],
                                 prox["gyro"], static_s, verbose=verbose)

    if verbose:
        print("\n" + "=" * 60)
        print("DISTAL SENSOR CALIBRATION")
        print("=" * 60)
    dist_cal = calibrate_single(dist["t"], dist["quat"], dist["cal_acc"],
                                 dist["gyro"], static_s, verbose=verbose)

    # --- functional alignment via shared joint axis -------------------------
    # During seated knee swings, both sensors rotate about the knee axis.
    # PCA on each sensor's angular velocity recovers that axis in each body
    # frame. These should be parallel in the world frame.
    t = np.asarray(dist["t"], float)
    t = t - t[0]
    moving = t >= static_s

    ax_prox, ev_prox = _pca_axis(prox["gyro"], moving)
    ax_dist, ev_dist = _pca_axis(dist["gyro"], moving)

    ax_prox, ax_dist, align = _align_axes(
        ax_prox, ax_dist, prox_cal["q_0"], dist_cal["q_0"])

    if verbose:
        print(f"\n{'=' * 60}")
        print("FUNCTIONAL ALIGNMENT")
        print(f"{'=' * 60}")
        print(f"  joint axis PCA variance  proximal {ev_prox:.3f}  distal {ev_dist:.3f}")
        print(f"  axis alignment error     {align:.2f} deg   (want < 10)")
        if align > 15:
            print("  -> the two sensors' PCA axes do not agree well. Either the")
            print("     motion was not a clean hinge, or the straps slipped.")
        elif align > 5:
            print("  -> acceptable. Some out-of-plane motion or strap play.")
        else:
            print("  -> good. The two sensors see the same rotation axis.")

    # Relative quaternion between the two neutral poses: this is the
    # orientation offset between the two segments in standing.
    q_offset = qmul(qconj([prox_cal["q_0"]]), [dist_cal["q_0"]])[0]

    return {
        "prox_cal": prox_cal,
        "dist_cal": dist_cal,
        "joint_axis_prox": ax_prox,
        "joint_axis_dist": ax_dist,
        "alignment_deg": align,
        "joint_q_offset": q_offset,
    }


def joint_angle(q_prox, q_dist, cal):
    """
    Knee (or ankle) flexion/extension angle from two calibrated sensors.

    Parameters
    ----------
    q_prox : (N, 4) quaternions from the proximal sensor (thigh for knee)
    q_dist : (N, 4) quaternions from the distal sensor (shank for knee)
    cal : dict from calibrate_pair

    Returns
    -------
    angle : (N,) degrees, relative to the neutral standing pose.
            Positive = flexion by convention (configurable via resolve_joint_sign).
    """
    q_p = qnorm(np.asarray(q_prox, float))
    q_d = qnorm(np.asarray(q_dist, float))

    pc = cal["prox_cal"]
    dc = cal["dist_cal"]

    # Each sensor's sagittal angle relative to its own neutral
    ang_prox = segment_angle(q_p, pc["q_f"], pc["q_0"])
    ang_dist = segment_angle(q_d, dc["q_f"], dc["q_0"])

    # Joint angle = distal - proximal (shank angle minus thigh angle)
    # This gives flexion/extension relative to the neutral standing pose.
    return ang_dist - ang_prox


def resolve_joint_sign(q_prox, q_dist, cal, verbose=True):
    """
    Verify that flexion produces a POSITIVE joint angle. If not, flip both
    segment calibrations' signs so the convention is consistent.

    Call this after calibrate_pair, using a recording that contains a
    deliberate, unambiguous flexion movement.
    """
    ang = joint_angle(q_prox, q_dist, cal)
    peak_pos = np.percentile(ang, 95)
    peak_neg = np.percentile(ang, 5)

    if verbose:
        print(f"\n  joint angle range: {ang.min():.1f} to {ang.max():.1f} deg")
        print(f"  peak positive (p95): {peak_pos:.1f}")
        print(f"  peak negative (p05): {peak_neg:.1f}")

    # Flexion should produce the larger excursion
    if abs(peak_neg) > abs(peak_pos):
        if verbose:
            print("  -> flexion appears NEGATIVE. Flipping sign convention.")
        # Flip by negating the functional frame quaternion's vector part
        for key in ("prox_cal", "dist_cal"):
            q = cal[key]["q_f"].copy()
            q[1:] = -q[1:]
            cal[key]["q_f"] = q
    else:
        if verbose:
            print("  -> flexion is positive. Convention is correct.")

    return cal


# --------------------------------------------------------------------------
# Three-sensor extension: toe + shank + thigh
# --------------------------------------------------------------------------

def calibrate_three(toe, shank, thigh, static_s=5.0, verbose=True):
    """
    Calibrate three sensors for ankle + knee joint angles.

    Each adjacent pair is treated independently:
      ankle = calibrate_pair(shank, toe)    # shank is proximal to the toe
      knee  = calibrate_pair(thigh, shank)  # thigh is proximal to the shank

    The functional alignment for each joint comes from a motion that isolates
    that joint's axis:
      - Knee axis: seated knee swings (both thigh and shank rotate about it)
      - Ankle axis: seated ankle dorsi/plantar flexion (shank and toe rotate
        about it, but the thigh does not)

    For a single recording to calibrate both joints, the dynamic take should
    include BOTH motions:
      1. Seated knee swings (8-10 reps)  -> aligns thigh-shank
      2. Seated ankle flexion (8-10 reps) -> aligns shank-toe
      3. Walking steps if possible       -> resolves signs

    Parameters
    ----------
    toe, shank, thigh : dicts from imu_receiver.load, same recording

    Returns
    -------
    dict with knee_cal, ankle_cal (each a calibrate_pair result)
    """
    if verbose:
        print("\n" + "#" * 60)
        print("# KNEE JOINT (thigh + shank)")
        print("#" * 60)
    knee_cal = calibrate_pair(thigh, shank, static_s, verbose=verbose)

    if verbose:
        print("\n" + "#" * 60)
        print("# ANKLE JOINT (shank + toe)")
        print("#" * 60)
    ankle_cal = calibrate_pair(shank, toe, static_s, verbose=verbose)

    return {
        "knee_cal": knee_cal,
        "ankle_cal": ankle_cal,
    }


def three_sensor_angles(q_toe, q_shank, q_thigh, cal):
    """
    Knee and ankle angles from three calibrated sensors.

    Returns (knee_angle, ankle_angle), each in degrees.
    """
    knee = joint_angle(q_thigh, q_shank, cal["knee_cal"])
    ankle = joint_angle(q_shank, q_toe, cal["ankle_cal"])
    return knee, ankle


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    import argparse
    from imu_receiver import load

    ap = argparse.ArgumentParser(
        description="Multi-sensor calibration and joint angle extraction.")
    ap.add_argument("npz", help="recording with all sensors")
    ap.add_argument("--proximal", required=True, help="proximal sensor device name (e.g. imu06)")
    ap.add_argument("--distal", required=True, help="distal sensor device name (e.g. imu05)")
    ap.add_argument("--static", type=float, default=5.0)
    ap.add_argument("--third", default=None, help="third sensor for 3-joint config (e.g. imu07)")
    a = ap.parse_args()

    if a.third:
        _, toe = load(a.npz, a.third)
        _, shank = load(a.npz, a.distal)
        _, thigh = load(a.npz, a.proximal)
        cal = calibrate_three(toe, shank, thigh, a.static)
        knee, ankle = three_sensor_angles(toe["quat"], shank["quat"],
                                           thigh["quat"], cal)
        print(f"\n  knee angle range:  {knee.min():.1f} to {knee.max():.1f} deg")
        print(f"  ankle angle range: {ankle.min():.1f} to {ankle.max():.1f} deg")
    else:
        _, prox = load(a.npz, a.proximal)
        _, dist = load(a.npz, a.distal)
        cal = calibrate_pair(prox, dist, a.static)
        ang = joint_angle(prox["quat"], dist["quat"], cal)
        print(f"\n  joint angle range: {ang.min():.1f} to {ang.max():.1f} deg")

        # Resolve sign against the same recording
        cal = resolve_joint_sign(prox["quat"], dist["quat"], cal)
        ang = joint_angle(prox["quat"], dist["quat"], cal)
        print(f"  after sign check: {ang.min():.1f} to {ang.max():.1f} deg")


if __name__ == "__main__":
    main()