#!/usr/bin/env python3
"""
shank_calibration.py  -  ReBAIT's calibration math, for ONE sensor.

Lifted from DataCollector.dynamicCalibrate, minus the framework. Runs
standalone so a single-IMU test doesn't trip over ReBAIT's hardcoded
devIds[0..3] pelvis/thigh/shank/foot indexing.

WHAT THIS MEASURES
------------------
Shank SEGMENT sagittal angle -- the inclination of the shank -- NOT knee
flexion/extension. Knee angle needs a thigh sensor too (q_SH* . q_TH).

RECORDING PROTOCOL (one continuous capture, do not stop between parts):
  0 - 5 s     static: stand relaxed, weight even, knee unlocked
  5 s onward  dynamic: seated knee swings, then toe touches, then walking

The 5 s static window is recovered by slicing t < STATIC_SEC, exactly as
ReBAIT does. If you record the two parts separately the slice grabs the
wrong data.

Uses plain numpy -- no numpy-quaternion dependency.
"""

import argparse
import numpy as np
from sklearn.decomposition import PCA

STATIC_SEC = 5.0
G_ANATOMICAL = np.array([0.0, 9.81, 0.0])   # ISB Y-up, as ReBAIT hardcodes


# ------------------------------------------------------------------ quaternion
def qconj(q):
    q = np.atleast_2d(q)
    return np.column_stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]])


def qmul(a, b):
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.column_stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def qrot(q, v):
    """Rotate vectors v by quaternions q  ->  q (x) v (x) conj(q)."""
    q, v = np.atleast_2d(q), np.atleast_2d(v)
    w, u = q[:, :1], q[:, 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def qnorm(q):
    q = np.atleast_2d(q)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def axis_angle_quat(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return np.array([np.cos(angle / 2.0), *(np.sin(angle / 2.0) * axis)])


def quat_mean(q):
    """Mean of a tight quaternion cluster, sign-aligned to the first sample."""
    q = qnorm(q)
    ref = q[0]
    s = np.sign(q @ ref)
    s[s == 0] = 1.0
    return qnorm((q * s[:, None]).mean(axis=0))[0]


# ------------------------------------------------------------------ calibration
def _make_q0(quat, static, q_f):
    """Neutral-pose reference so the angle reads ~0 during the static hold."""
    q_init = qmul(qconj(quat[static]), np.tile(qconj([q_f]), (int(static.sum()), 1)))
    return quat_mean(q_init)


def flip_ml_sign(cal, t, quat, static_sec=STATIC_SEC):
    """Reverse the medial-lateral direction and rebuild the neutral reference."""
    t = np.asarray(t, dtype=float); t = t - t[0]
    static = t < static_sec
    q_f = qmul(np.array([0.0, 0.0, 1.0, 0.0]), cal["q_f"])[0]
    out = dict(cal)
    out["q_f"] = q_f
    out["q_0"] = _make_q0(qnorm(np.asarray(quat, dtype=float)), static, q_f)
    out["sign_flipped"] = not cal["sign_flipped"]
    return out


def resolve_sign(cal, t, quat, window, expect_increase=True,
                 static_sec=STATIC_SEC, verbose=True):
    """
    Settle the medial-lateral direction against a movement you KNOW.

    ReBAIT decides the sign by comparing peak positive and peak negative
    angular velocity. That heuristic encodes their own mounting and axis
    convention -- it is not a universal correctness test, and with a different
    sensor orientation it can resolve the wrong way. Verify it once.

    window          (t0, t1) in seconds, spanning a deliberate movement
    expect_increase True if the angle should INCREASE across that window
    """
    t = np.asarray(t, dtype=float); t = t - t[0]
    m = (t >= window[0]) & (t <= window[1])
    if m.sum() < 10:
        raise ValueError("Verification window contains too few samples.")

    ang = sagittal_angle(quat, cal["q_f"], cal["q_0"])
    delta = float(np.mean(ang[m][-max(3, m.sum() // 10):]) -
                  np.mean(ang[m][:max(3, m.sum() // 10)]))
    ok = (delta > 0) == bool(expect_increase)

    if verbose:
        print("--- medial-lateral sign check ---")
        print(f"  net change over {window[0]:.1f}-{window[1]:.1f} s: {delta:+.2f} deg")
        print(f"  expected: {'increase' if expect_increase else 'decrease'}")
        print("  -> sign correct as calibrated." if ok else
              "  -> sign was INVERTED; flipping and rebuilding the reference.")
    return cal if ok else flip_ml_sign(cal, t, quat, static_sec)


def calibrate(t, quat, cal_acc, gyro, static_sec=STATIC_SEC, verbose=True,
              sign_override=None):
    """
    Returns a dict with q_f, q_0 and the diagnostics you need to judge whether
    the calibration is trustworthy.

    t        (N,)    seconds, starting near zero
    quat     (N,4)   w,x,y,z from the sensor's fusion filter
    cal_acc  (N,3)   m/s^2, gravity INCLUDED, sensor frame
    gyro     (N,3)   rad/s, sensor frame
    """
    t = np.asarray(t, dtype=float)
    t = t - t[0]
    quat = qnorm(np.asarray(quat, dtype=float))
    cal_acc = np.asarray(cal_acc, dtype=float)
    gyro = np.asarray(gyro, dtype=float)

    static = t < static_sec
    if static.sum() < 20:
        raise ValueError(f"Only {static.sum()} samples in the static window.")

    # world-frame accel and gyro, using the sensor's own orientation estimate
    a_world = qrot(quat, cal_acc)
    w_world = qrot(quat, gyro)

    # --- step 1: gravity fixes the superior-inferior axis -> q_g -------------
    g_avg = a_world[static].mean(axis=0)
    g_mag = np.linalg.norm(g_avg)
    n = np.cross(g_avg, G_ANATOMICAL)
    n = n / np.linalg.norm(n)
    theta = np.arccos(np.clip(np.dot(g_avg, G_ANATOMICAL) /
                              (g_mag * np.linalg.norm(G_ANATOMICAL)), -1, 1))
    q_g = axis_angle_quat(n, theta)

    # --- step 2: PCA on angular velocity fixes the medial-lateral axis -------
    w_g = qrot(np.tile(q_g, (len(w_world), 1)), w_world)
    dyn = ~static
    if dyn.sum() < 100:
        raise ValueError("Dynamic window too short for PCA.")
    pca = PCA(n_components=3).fit(w_g[dyn])
    z_pca = pca.components_[0]
    evr = pca.explained_variance_ratio_

    rot_ax = np.cross(z_pca, [0, 0, 1])
    ang = np.arccos(np.clip(np.dot(z_pca, [0, 0, 1]), -1, 1))
    sgn = np.sign(rot_ax[1]) if rot_ax[1] != 0 else 1.0
    q_pca = np.array([np.cos(ang / 2.0), 0.0, np.sin(ang / 2.0) * sgn, 0.0])
    q_pca = qnorm(q_pca)[0]

    q_f = qmul(q_pca, q_g)[0]

    # --- step 3: sign disambiguation ----------------------------------------
    # PCA returns an axis, not a direction, so medial-lateral can come out
    # reversed. ReBAIT resolves it by comparing the largest positive and
    # largest negative angular velocity -- which only works if the motion is
    # ASYMMETRIC in rate. Gait is (fast swing one way, slow the other).
    # A symmetric pendulum-like swing is degenerate and the rule becomes a
    # coin flip on noise, so we also report the margin.
    w_f = qrot(np.tile(q_f, (len(w_world), 1)), w_world)
    w_pos = float(w_f[dyn, 2].max())
    w_neg = float(-w_f[dyn, 2].min())
    sign_margin = (w_pos - w_neg) / max(w_pos + w_neg, 1e-9)

    if sign_override is None:
        flipped = w_pos > w_neg
    else:
        flipped = bool(sign_override)
    if flipped:
        q_f = qmul(np.array([0.0, 0.0, 1.0, 0.0]), q_f)[0]   # 180 deg about y

    # --- step 4: neutral-pose reference -------------------------------------
    q_0 = _make_q0(quat, static, q_f)

    # --- diagnostics ---------------------------------------------------------
    gravity_err_deg = np.degrees(np.arccos(np.clip(
        (a_world @ g_avg) / (np.linalg.norm(a_world, axis=1) * g_mag), -1, 1)))
    still = np.linalg.norm(gyro, axis=1) < np.radians(5)

    out = {
        "q_f": q_f, "q_0": q_0, "q_g": q_g, "q_pca": q_pca,
        "pca_ratio": float(evr[0]),
        "pca_ratio_2nd": float(evr[1]),
        "pca_axis": z_pca,
        "sign_flipped": flipped,
        "sign_margin": float(sign_margin),
        "w_pos_max": w_pos, "w_neg_max": w_neg,
        "gravity_magnitude": float(g_mag),
        "n_static": int(static.sum()),
        "n_dynamic": int(dyn.sum()),
        "gravity_wander_deg_p95": float(np.percentile(gravity_err_deg[still], 95))
        if still.sum() > 50 else float("nan"),
    }

    if verbose:
        print("--- calibration ---")
        print(f"  static samples          {out['n_static']}")
        print(f"  dynamic samples         {out['n_dynamic']}")
        print(f"  |gravity| static        {g_mag:.3f} m/s^2   (expect ~9.81)")
        print(f"  PCA explained variance  {evr[0]:.3f}  (2nd: {evr[1]:.3f})")
        print(f"  medial-lateral sign     {'FLIPPED' if flipped else 'as found'}"
              f"   (margin {sign_margin:+.3f})")
        print("     (this heuristic encodes ReBAIT's own axis convention --")
        print("      confirm it with resolve_sign() against a known movement)")
        if abs(sign_margin) < 0.12:
            print("  -> SIGN IS AMBIGUOUS. Peak positive and negative angular")
            print("     velocities are nearly equal, so the direction test is")
            print("     decided by noise. Include WALKING in the dynamic take -")
            print("     gait is asymmetric in rate; a symmetric swing is not.")
            print("     Verify against a deliberate known movement before trusting.")
        print(f"  gravity wander p95      {out['gravity_wander_deg_p95']:.2f} deg")
        _judge_pca(evr[0])
    return out


def _judge_pca(r):
    if r >= 0.95:
        print("  -> PCA well conditioned: rotation is nearly planar.")
    elif r >= 0.85:
        print("  -> PCA acceptable but not clean. Add larger, slower, more")
        print("     purely sagittal motion (seated knee swings) to the dynamic take.")
    else:
        print("  -> PCA POORLY conditioned. The medial-lateral axis is not well")
        print("     determined and every downstream angle inherits that error.")
        print("     Do not trust results from this calibration.")


def sagittal_angle(quat, q_f, q_0):
    """Segment sagittal angle in degrees, relative to the neutral static pose."""
    quat = qnorm(np.asarray(quat, dtype=float))
    n = len(quat)
    q_i = qmul(qmul(np.tile(q_f, (n, 1)), quat), np.tile(q_0, (n, 1)))

    # Resolve the quaternion double cover: q and -q are the same rotation but
    # give different arccos branches. Force each sample into the same
    # hemisphere as its predecessor before extracting the angle.
    if n > 1:
        d = np.sum(q_i[1:] * q_i[:-1], axis=1)
        flip = np.cumprod(np.where(d < 0.0, -1.0, 1.0))
        q_i[1:] = q_i[1:] * flip[:, None]

    w, z = q_i[:, 0], q_i[:, 3]
    denom = np.sqrt(w ** 2 + z ** 2)
    denom[denom == 0] = 1e-12
    ang = 2.0 * np.arccos(np.clip(w / denom, -1, 1)) * np.sign(z)
    return np.degrees(np.unwrap(ang))


def repeatability(records, static_sec=STATIC_SEC):
    """
    Spread across repeated calibrations of the SAME mounting.
    No result can be better than this number -- run it before anything else.
    Pass a list of (t, quat, cal_acc, gyro) tuples.
    """
    neutral, ratios = [], []
    for t, q, a, g in records:
        c = calibrate(t, q, a, g, static_sec, verbose=False)
        ang = sagittal_angle(q, c["q_f"], c["q_0"])
        neutral.append(np.mean(ang[np.asarray(t) - t[0] < static_sec]))
        ratios.append(c["pca_ratio"])
    neutral = np.asarray(neutral)
    print("--- calibration repeatability ---")
    print(f"  runs                 {len(neutral)}")
    print(f"  neutral angle SD     {np.std(neutral, ddof=1):.3f} deg")
    print(f"  neutral angle range  {np.ptp(neutral):.3f} deg")
    print(f"  PCA ratio range      {min(ratios):.3f} - {max(ratios):.3f}")
    print("  This SD is your error floor. Nothing downstream can beat it.")
    return {"sd": float(np.std(neutral, ddof=1)), "range": float(np.ptp(neutral))}


if __name__ == "__main__":
    from imu_receiver import load

    ap = argparse.ArgumentParser(description="Calibrate and extract shank angle.")
    ap.add_argument("npz", help="recording from imu_receiver.py")
    ap.add_argument("--device", default=None)
    ap.add_argument("--static", type=float, default=STATIC_SEC)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    did, d = load(a.npz, a.device)
    print(f"device {did}, {len(d['t'])} samples\n")
    cal = calibrate(d["t"], d["quat"], d["cal_acc"], d["gyro"], a.static)
    ang = sagittal_angle(d["quat"], cal["q_f"], cal["q_0"])

    t = d["t"] - d["t"][0]
    print(f"\n  shank angle range   {ang.min():.1f} to {ang.max():.1f} deg")
    print(f"  neutral (static)    {np.mean(ang[t < a.static]):.2f} deg")
    print("\n  NOTE: this angle is relative to your NEUTRAL STANDING POSE,")
    print("  not to lab vertical. Measure your standing shank inclination")
    print("  with an angle gauge if you need an absolute reference.")

    if a.plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(11, 4))
        plt.plot(t, ang, lw=0.8)
        plt.axvspan(0, a.static, color="0.85", label="static window")
        plt.xlabel("time (s)"); plt.ylabel("shank sagittal angle (deg)")
        plt.legend(); plt.tight_layout(); plt.show()
