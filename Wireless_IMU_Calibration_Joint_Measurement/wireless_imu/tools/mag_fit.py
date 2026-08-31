#!/usr/bin/env python3
"""
mag_fit.py  -  proper magnetometer calibration by ellipsoid fitting.

WHY NOT THE LIBRARY'S METHOD
----------------------------
calibrateMag() uses min/max per axis: it keeps six extreme values and nothing
else. That means

  * the answer depends entirely on whether you reached the true extreme on
    every axis in both directions - miss one and both the centre and the
    radius for that axis are wrong,
  * a single noise spike permanently sets an extreme,
  * it can only scale each axis independently, so it cannot correct soft iron
    (which rotates and shears the ellipsoid, not just stretches it along the
    sensor axes).

That is why three tumbles gave three different answers.

WHAT THIS DOES INSTEAD
----------------------
An ideal magnetometer traced through every orientation sweeps out a SPHERE.
Hard iron shifts the centre; soft iron turns it into a general ellipsoid.
Fit the ellipsoid to ALL the samples at once:

    (m - b)' A (m - b) = 1

then the correction that maps measurements back onto a sphere of radius F is

    m_corrected = F * A^(1/2) (m - b)

b is the hard-iron offset (3x1) and A^(1/2) the soft-iron matrix (3x3, and
crucially NOT diagonal). Every sample contributes, so no single point can
wreck the fit and no extreme has to be hit exactly.

USAGE
    python3 mag_fit.py tumble.npz              # fit and report
    python3 mag_fit.py tumble.npz --plot       # add before/after plots

Record the input by tumbling the sensor continuously for 60 s, in open space,
away from steel. Coverage matters far less than for min/max, but you still
want samples spread over the whole sphere.
"""

import argparse
import numpy as np


def fit_ellipsoid(m):
    """
    Least-squares general ellipsoid fit.

    Solves the quadric  x'Qx + 2p'x + k = 0  in the standard 9-parameter form,
    then converts to centre b and shape matrix A.
    """
    x, y, z = m[:, 0], m[:, 1], m[:, 2]
    D = np.column_stack([x * x, y * y, z * z,
                         2 * x * y, 2 * x * z, 2 * y * z,
                         2 * x, 2 * y, 2 * z])
    # solve D v = 1 in the least-squares sense
    v, *_ = np.linalg.lstsq(D, np.ones(len(m)), rcond=None)

    Q = np.array([[v[0], v[3], v[4]],
                  [v[3], v[1], v[5]],
                  [v[4], v[5], v[2]]])
    p = v[6:9]

    b = -np.linalg.solve(Q, p)                       # hard-iron offset
    k = 1.0 + float(b @ Q @ b)                       # scale so the form = 1
    A = Q / k

    w, V = np.linalg.eigh(A)
    if np.any(w <= 0):
        raise ValueError("Fit did not converge to an ellipsoid. The tumble "
                         "probably did not cover enough of the sphere.")
    A_sqrt = V @ np.diag(np.sqrt(w)) @ V.T
    radii = 1.0 / np.sqrt(w)
    return b, A_sqrt, radii


def report(m, b, A_sqrt, F, label):
    c = F * (m - b) @ A_sqrt.T
    n = np.linalg.norm(c, axis=1)
    return {"mean": float(n.mean()), "sd": float(n.std()),
            "cv": float(100 * n.std() / n.mean()), "label": label, "norms": n}


def main():
    from imu_receiver import load

    ap = argparse.ArgumentParser(description="Ellipsoid-fit magnetometer calibration.")
    ap.add_argument("npz", help="recording of a continuous tumble")
    ap.add_argument("--device", default=None)
    ap.add_argument("--min-move", type=float, default=20.0,
                    help="deg/s below which a sample is dropped as stationary")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    did, d = load(a.npz, a.device)
    m_all = np.asarray(d["mag"], float)
    g = np.degrees(np.linalg.norm(np.asarray(d["gyro"], float), axis=1))

    keep = g > a.min_move
    m = m_all[keep]
    print(f"device {did}   {len(m_all)} samples, {len(m)} while tumbling")
    if len(m) < 500:
        raise SystemExit("Too few moving samples. Tumble continuously while recording.")

    # coverage check: how much of the sphere did the tumble actually visit?
    u = (m - m.mean(axis=0))
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    cov = np.linalg.svd(u, compute_uv=False)
    cov = cov / cov.sum()
    print(f"direction coverage (should be near 0.33 each): "
          f"{cov[0]:.2f} {cov[1]:.2f} {cov[2]:.2f}")
    if cov[2] < 0.15:
        print("  -> one direction is under-sampled. The fit will be weak along it.")

    b, A_sqrt, radii = fit_ellipsoid(m)
    F = float(np.mean(radii))          # target sphere radius = mean semi-axis

    before = np.linalg.norm(m, axis=1)
    after = report(m, b, A_sqrt, F, "after")

    print()
    print(f"  |m| before   {before.mean():7.1f} +/- {before.std():6.1f} mG  "
          f"({100*before.std()/before.mean():5.2f}% CV)")
    print(f"  |m| after    {after['mean']:7.1f} +/- {after['sd']:6.1f} mG  "
          f"({after['cv']:5.2f}% CV)")
    print(f"  ellipsoid semi-axes  {radii[0]:.1f}  {radii[1]:.1f}  {radii[2]:.1f} mG")
    print(f"  axis ratio           {radii.max()/radii.min():.3f}   "
          f"(1.000 = no soft iron)")
    print()
    print(f"  hard-iron offset [mG]  {b[0]:+8.2f} {b[1]:+8.2f} {b[2]:+8.2f}")
    print(f"  |offset|               {np.linalg.norm(b):8.1f} mG")
    print()
    print("  soft-iron matrix (scaled so the sphere radius is 1):")
    M = F * A_sqrt
    for r in M:
        print("      " + "  ".join(f"{q:+9.6f}" for q in r))

    off_diag = np.abs(M - np.diag(np.diag(M))).max() / np.abs(np.diag(M)).mean()
    print(f"  largest off-diagonal term: {100*off_diag:.1f}% of the diagonal")
    if off_diag > 0.02:
        print("  -> real soft iron / axis misalignment present. The library's")
        print("     per-axis scaling CANNOT represent this, which is the main")
        print("     reason its result was unstable.")

    print()
    if after["cv"] < 2.0:
        print("  RESULT: good. Apply this in software on the host.")
    elif after["cv"] < 5.0:
        print("  RESULT: usable but not tight. More tumble coverage would help.")
    else:
        print("  RESULT: poor. Either the tumble was incomplete, or the field")
        print("  where you recorded is itself disturbed. Repeat in open space.")

    print()
    print("  --- paste into your Python processing (full correction) ---")
    print(f"  MAG_B = np.array([{b[0]:.4f}, {b[1]:.4f}, {b[2]:.4f}])")
    print("  MAG_A = np.array([")
    for r in M:
        print(f"      [{r[0]:.6f}, {r[1]:.6f}, {r[2]:.6f}],")
    print("  ])")
    print("  # m_corrected = (m_raw - MAG_B) @ MAG_A.T")

    # what the firmware can actually hold: diagonal only
    diag = np.diag(M)
    print()
    print("  --- firmware constants (DIAGONAL ONLY - loses the cross terms) ---")
    print(f"  const float MAG_BIAS_X  = {b[0]:.2f}f;")
    print(f"  const float MAG_BIAS_Y  = {b[1]:.2f}f;")
    print(f"  const float MAG_BIAS_Z  = {b[2]:.2f}f;")
    print(f"  const float MAG_SCALE_X = {diag[0]/diag.mean():.4f}f;")
    print(f"  const float MAG_SCALE_Y = {diag[1]/diag.mean():.4f}f;")
    print(f"  const float MAG_SCALE_Z = {diag[2]/diag.mean():.4f}f;")

    d_only = F * np.diag(diag) @ np.eye(3)
    cd = (m - b) @ (F * np.diag(diag)).T
    nd = np.linalg.norm(cd, axis=1)
    print(f"  diagonal-only result: {100*nd.std()/nd.mean():.2f}% CV "
          f"(full matrix: {after['cv']:.2f}%)")
    if 100 * nd.std() / nd.mean() > 2 * after["cv"]:
        print("  -> the diagonal form throws away most of the benefit. Prefer")
        print("     correcting on the host with the full matrix.")

    if a.plot:
        import matplotlib.pyplot as plt
        c = F * (m - b) @ A_sqrt.T
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        for k, (data, ttl) in enumerate(((m, "raw"), (c, "corrected"))):
            ax[k].plot(data[:, 0], data[:, 1], ".", ms=1, alpha=0.3, label="XY")
            ax[k].plot(data[:, 0], data[:, 2], ".", ms=1, alpha=0.3, label="XZ")
            ax[k].plot(data[:, 1], data[:, 2], ".", ms=1, alpha=0.3, label="YZ")
            ax[k].set_aspect("equal"); ax[k].set_title(ttl); ax[k].legend(markerscale=8)
            ax[k].grid(alpha=0.3)
        plt.suptitle("magnetometer: corrected should be circular and centred")
        plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
