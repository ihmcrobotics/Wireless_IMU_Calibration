#!/usr/bin/env python3
"""
bench_checks.py  -  accuracy numbers for a single IMU, with NO motion capture.

Gravity is a free, absolute, continuous reference for two of three orientation
degrees of freedom -- and for sagittal shank angle those are the two that
matter. Everything here exploits that, or exploits internal consistency.

    link       packet loss, true sample rate, jitter
    gravity    world-frame gravity must stay put -> direct roll/pitch error
    drift      orientation wander while stationary, split into tilt vs heading
    mag        field magnitude and mag-gravity angle must be orientation-invariant
    gyrofilter Madgwick attenuation and lag, referenced to gyro integration
    closure    accumulated error over a maneuver that returns to a home pose
    sweep      static angle sweep vs an angle gauge: bias, linearity, hysteresis

Usage
    python3 bench_checks.py trial.npz --test all
    python3 bench_checks.py sweep.npz --test sweep --commanded 0 15 30 45 60 75 90
    python3 bench_checks.py closure.npz --test closure --home 2 7 --home 55 60

Depends on numpy and shank_calibration only. matplotlib optional (--plot).
"""

import argparse
import numpy as np

from shank_calibration import qmul, qconj, qrot, qnorm, quat_mean

G = 9.80665
STILL_GYRO = np.radians(3.0)      # rad/s below which a sample counts as static


# ----------------------------------------------------------------- helpers
def _rotvec(q):
    """Quaternion array -> rotation vector (axis * angle), radians."""
    q = qnorm(np.atleast_2d(q))
    q = q * np.where(q[:, :1] < 0, -1.0, 1.0)        # shortest arc
    w = np.clip(q[:, 0], -1.0, 1.0)
    ang = 2.0 * np.arccos(w)
    s = np.sqrt(np.maximum(1.0 - w ** 2, 1e-18))
    return q[:, 1:] * (ang / s)[:, None]


def _still(gyro, thresh=None, smooth=21):
    """
    Static-sample detector.

    The gyro magnitude is SMOOTHED first. At high sample rates broadband gyro
    noise makes the raw magnitude cross any fixed threshold constantly, so a
    per-sample test flickers and breaks continuous holds into fragments --
    which then defeats any settle-time requirement downstream.
    """
    thresh = STILL_GYRO if thresh is None else thresh   # read at CALL time
    g = np.linalg.norm(np.atleast_2d(gyro), axis=1)
    if smooth > 1 and len(g) > smooth:
        k = np.ones(smooth) / smooth
        g = np.convolve(g, k, mode="same")
    return g < thresh


def _plateaus(t, gyro, min_dur=3.0, thresh=None):
    """Index ranges where the sensor is held still for at least min_dur."""
    still = _still(gyro, thresh).astype(int)
    edges = np.diff(np.r_[0, still, 0])
    starts, stops = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [(a, b) for a, b in zip(starts, stops)
            if b > a and (t[b - 1] - t[a]) >= min_dur]


def _fmt(title):
    print(f"\n{'-' * 68}\n{title}\n{'-' * 68}")


# ----------------------------------------------------------------- 1. link
def link(t, seq, verbose=True):
    t = np.asarray(t, float)
    seq = np.asarray(seq, np.int64)
    dt = np.diff(t)
    expected = seq[-1] - seq[0] + 1
    lost = expected - len(seq)
    out = {"n": len(t), "duration_s": float(t[-1] - t[0]),
           "rate_hz": float(1.0 / np.median(dt)),
           "jitter_sd_ms": float(np.std(dt) * 1e3),
           "max_gap_ms": float(np.max(dt) * 1e3),
           "lost": int(lost),
           "loss_pct": float(100.0 * lost / max(expected, 1))}
    if verbose:
        _fmt("LINK")
        print(f"  samples              {out['n']}  over {out['duration_s']:.1f} s")
        print(f"  median rate          {out['rate_hz']:.2f} Hz")
        print(f"  inter-sample jitter  SD {out['jitter_sd_ms']:.2f} ms, "
              f"max gap {out['max_gap_ms']:.1f} ms")
        print(f"  packets lost         {out['lost']}  ({out['loss_pct']:.2f}%)")
        if out["loss_pct"] > 1.0:
            print("  -> loss above 1%. Check WiFi channel, distance, and that the")
            print("     receiver socket buffer is large enough.")
        if out["max_gap_ms"] > 5 * 1e3 / out["rate_hz"]:
            print("  -> a gap of several sample periods. Any gap-filling you do")
            print("     across it is interpolation, not measurement.")
    return out


# ----------------------------------------------------------------- 2. gravity
def gravity(quat, cal_acc, gyro, t=None, settle_s=1.0, verbose=True):
    """
    World-frame gravity must be a CONSTANT vector regardless of how the sensor
    is oriented. However far it wanders is roll/pitch error, measured with no
    external reference at all.
    """
    quat = qnorm(np.asarray(quat, float))
    a_w = qrot(quat, np.asarray(cal_acc, float))
    still = _still(gyro)

    # Exclude samples that are still but not yet SETTLED. The fusion filter
    # needs time to converge after a rotation, so samples right after motion
    # carry filter transient rather than sensor error, and inflate the wander.
    n_raw = int(still.sum())
    if t is not None and settle_s > 0:
        t = np.asarray(t, float); t = t - t[0]
        moving = ~still
        last_move = np.where(moving, t, -np.inf)
        last_move = np.maximum.accumulate(last_move)
        still = still & ((t - last_move) >= settle_s)

    if still.sum() < 50:
        raise ValueError(
            f"Too few settled samples ({int(still.sum())} of {len(quat)}). "
            f"{n_raw} were still before the {settle_s:.1f} s settle filter. "
            "Run gyro_diagnostic() to see whether the holds were long enough "
            "or the still threshold needs raising.")

    ref = a_w[still].mean(axis=0)
    ref_u = ref / np.linalg.norm(ref)
    cosang = np.clip((a_w[still] @ ref_u) / np.linalg.norm(a_w[still], axis=1), -1, 1)
    err = np.degrees(np.arccos(cosang))
    mag = np.linalg.norm(np.asarray(cal_acc, float)[still], axis=1)

    out = {"p50": float(np.percentile(err, 50)), "p95": float(np.percentile(err, 95)),
           "max": float(err.max()), "n_static": int(still.sum()),
           "accel_mag_mean": float(mag.mean()), "accel_mag_sd": float(mag.std()),
           "n_still_raw": n_raw, "settle_s": float(settle_s),
           "implied_bias_mg": float(1000.0 * np.std(mag) * np.sqrt(2.0) / G),
           "implied_scale_err_pct": float(100.0 * (mag.mean() - G) / G),
           "gravity_dir": ref_u}
    if verbose:
        _fmt("GRAVITY CONSTANCY  (roll/pitch error, no reference needed)")
        print(f"  still samples        {out['n_still_raw']}"
              f"   settled ({settle_s:.1f} s): {out['n_static']}")
        print(f"  |accel| when still   {out['accel_mag_mean']:.3f} "
              f"+/- {out['accel_mag_sd']:.3f} m/s^2   (expect 9.81)")
        print(f"    -> implies ~{out['implied_bias_mg']:.0f} mg residual bias / "
              f"axis-scale mismatch")
        print(f"    -> implies ~{out['implied_scale_err_pct']:+.1f}% overall "
              f"accelerometer scale error")
        print(f"  direction wander     p50 {out['p50']:.2f} deg, "
              f"p95 {out['p95']:.2f} deg, max {out['max']:.2f} deg")
        if abs(out["implied_scale_err_pct"]) > 1.0:
            print("  -> overall scale error. calibrateAccelGyro() removes BIAS only,")
            print("     never sensitivity. A uniform scale error costs you almost no")
            print("     ANGLE accuracy (attitude uses the direction of the vector, not")
            print("     its length), so this is cosmetic unless you need |a| itself.")
        if out["implied_bias_mg"] > 30:
            print("  -> the spread, not the mean, is what costs you angle accuracy.")
            print("     A 6-position calibration (each axis pointed down in turn)")
            print("     solves per-axis scale AND bias and would remove most of it.")
        if out["p95"] > 2.0:
            print("  -> more than 2 deg of wander across poses. This is an upper")
            print("     bound on your sagittal angle accuracy; nothing downstream")
            print("     can be better than this.")
        else:
            print("  -> tight. Roll/pitch tracking is sound.")
    return out


def gyro_diagnostic(t, gyro, thresh_deg=(1, 2, 3, 5, 8), verbose=True):
    """
    Why aren't there enough static samples? Shows the gyro noise floor and how
    the still fraction and longest continuous hold vary with the threshold.
    """
    t = np.asarray(t, float); t = t - t[0]
    g = np.degrees(np.linalg.norm(np.asarray(gyro, float), axis=1))
    k = np.ones(21) / 21
    gs = np.convolve(g, k, mode="same")
    if verbose:
        _fmt("GYRO / HOLD DIAGNOSTIC")
        print(f"  |w| raw     median {np.median(g):.2f}, p95 {np.percentile(g,95):.2f} deg/s")
        print(f"  |w| smoothed median {np.median(gs):.2f}, "
              f"p95 {np.percentile(gs,95):.2f} deg/s")
        print(f"\n  {'thresh':>7} {'still %':>9} {'holds':>7} {'longest':>9}")
        for th in thresh_deg:
            m = gs < th
            e = np.diff(np.r_[0, m.astype(int), 0])
            a, b = np.where(e == 1)[0], np.where(e == -1)[0]
            durs = [t[j - 1] - t[i] for i, j in zip(a, b) if j > i]
            long = max(durs) if durs else 0.0
            n_ok = sum(1 for d in durs if d >= 2.0)
            print(f"  {th:7.1f} {100*m.mean():8.1f}% {n_ok:7d} {long:8.2f}s")
        print("\n  'holds' counts stretches of at least 2 s. You want that to match")
        print("  the number of poses you actually held.")

    # Distinguish residual BIAS from NOISE using the quietest stretch.
    q = np.argsort(gs)[:max(200, len(gs) // 50)]
    mean_vec = np.degrees(np.asarray(gyro, float)[q].mean(axis=0))
    sd_vec = np.degrees(np.asarray(gyro, float)[q].std(axis=0))
    bias_mag = float(np.linalg.norm(mean_vec))
    if verbose:
        _fmt("GYRO BIAS vs NOISE  (quietest samples only)")
        print(f"  mean  [{mean_vec[0]:+6.2f} {mean_vec[1]:+6.2f} {mean_vec[2]:+6.2f}] "
              f"deg/s   |mean| {bias_mag:.2f}")
        print(f"  SD    [{sd_vec[0]:6.2f} {sd_vec[1]:6.2f} {sd_vec[2]:6.2f}] deg/s")
        if bias_mag > 1.0:
            print("  -> the MEAN is far from zero, so this is residual BIAS, not noise.")
            print("     The bias correction is not reaching the output. A gyro bias of")
            print("     this size also drives attitude drift inside the fusion filter.")
            print("     Fix: subtract it in firmware before use, or on the host.")
        elif sd_vec.max() > 1.0:
            print("  -> mean near zero but SD large: genuine NOISE. Narrow the gyro")
            print("     DLPF (DLPF_41HZ) rather than chasing bias.")
        else:
            print("  -> clean. Bias removed and noise low.")
    return {"median_deg_s": float(np.median(gs)),
            "bias_deg_s": mean_vec, "noise_sd_deg_s": sd_vec,
            "bias_mag_deg_s": bias_mag}


# ----------------------------------------------------------------- 3. drift
def drift(t, quat, cal_acc, verbose=True):
    """
    Stationary orientation wander, split into TILT (gravity-corrected, should be
    flat) and HEADING (about vertical, unobservable from gravity so it wanders).
    Heading drift rate is how fast your sagittal plane definition rotates.
    """
    t = np.asarray(t, float); t = t - t[0]
    quat = qnorm(np.asarray(quat, float))
    rel = qmul(np.tile(qconj([quat[0]]), (len(quat), 1)), quat)
    rv = np.degrees(_rotvec(rel))

    up = qrot(quat[:1], np.asarray(cal_acc, float)[:1])[0]
    up = up / np.linalg.norm(up)

    heading = rv @ up
    tilt = np.linalg.norm(rv - heading[:, None] * up, axis=1)

    def rate(y):
        return float(np.polyfit(t, y, 1)[0] * 60.0)   # deg per minute

    out = {"duration_min": float(t[-1] / 60.0),
           "heading_total_deg": float(heading[-1] - heading[0]),
           "heading_rate_deg_min": rate(heading),
           "tilt_max_deg": float(tilt.max()),
           "tilt_rate_deg_min": rate(tilt),
           "t": t, "heading": heading, "tilt": tilt}
    if verbose:
        _fmt("STATIONARY DRIFT")
        print(f"  duration             {out['duration_min']:.1f} min")
        print(f"  tilt   max {out['tilt_max_deg']:.2f} deg, "
              f"rate {out['tilt_rate_deg_min']:+.3f} deg/min")
        print(f"  heading total {out['heading_total_deg']:+.2f} deg, "
              f"rate {out['heading_rate_deg_min']:+.3f} deg/min")
        if out["tilt_rate_deg_min"] > 0.05:
            print("  -> tilt should be near flat; gravity corrects it continuously.")
            print("     Drift here suggests accelerometer bias or a filter gain issue.")
        print("  -> heading has no absolute reference beyond the magnetometer.")
        print("     Run this on the bench AND in the capture volume; the")
        print("     difference between the two is your magnetic environment.")
    return out


# ----------------------------------------------------------------- 4. magnetic
def magnetic(mag, cal_acc, gyro, verbose=True):
    """
    Two quantities are orientation-invariant for an undisturbed field and a
    well-calibrated magnetometer: the field MAGNITUDE, and the ANGLE between the
    magnetic vector and gravity. Both computed in the sensor frame, so this
    needs no orientation estimate at all.
    """
    mag = np.asarray(mag, float)
    acc = np.asarray(cal_acc, float)
    still = _still(gyro)
    if still.sum() < 50:
        raise ValueError("Too few static samples. Hold several distinct poses still.")

    m, a = mag[still], acc[still]
    norm = np.linalg.norm(m, axis=1)
    cosang = np.clip(np.sum(m * a, axis=1) /
                     (norm * np.linalg.norm(a, axis=1)), -1, 1)
    dip = 90.0 - np.degrees(np.arccos(cosang))

    out = {"mag_mean": float(norm.mean()), "mag_sd": float(norm.std()),
           "mag_cv_pct": float(100 * norm.std() / max(norm.mean(), 1e-9)),
           "dip_mean": float(dip.mean()), "dip_sd": float(dip.std()),
           "dip_range": float(np.ptp(dip)), "n_static": int(still.sum())}
    if verbose:
        _fmt("MAGNETIC CONSISTENCY")
        print(f"  static samples       {out['n_static']}")
        print(f"  |m|                  {out['mag_mean']:.1f} +/- {out['mag_sd']:.1f} mG "
              f"({out['mag_cv_pct']:.1f}% CV)")
        print(f"  dip angle            {out['dip_mean']:.2f} +/- {out['dip_sd']:.2f} deg "
              f"(range {out['dip_range']:.2f})")
        if out["mag_cv_pct"] > 5 or out["dip_sd"] > 3:
            print("  -> NOT orientation-invariant. Either the hard/soft-iron")
            print("     calibration is poor, or the local field is disturbed.")
            print("     Repeat far from steel to separate the two causes.")
            print("     If it is the environment, consider running mag-free:")
            print("     sagittal angle is largely heading-insensitive.")
        else:
            print("  -> consistent. Magnetometer calibration and local field are sound.")
    return out


# ----------------------------------------------------------------- 5. filter
def gyrofilter(t, quat, gyro, max_lag_ms=120.0, verbose=True, smooth=7):
    """
    Madgwick attenuation and lag, referenced to gyro integration. No external
    instrument: over short horizons the gyro is trustworthy for CHANGE in
    orientation, so differentiating the filter quaternion and comparing gives
    both quantities.

    Two things that a naive version gets wrong, and this one does not:

    1. Regress SIGNED PROJECTIONS onto the dominant rotation axis, not vector
       MAGNITUDES. A norm is non-negative, so any extra noise on the filter
       estimate inflates its mean and pushes the slope above 1 -- an
       attenuation above 1.0 is otherwise impossible for a complementary
       filter, and is the tell that this bias is present.

    2. Zero-phase smooth the quaternion-derived rate. Differentiating a
       quaternion at a few hundred hertz amplifies noise badly. Smoothing is
       necessary, but an ordinary filter would add lag to the very quantity
       being measured, so the smoother is applied forwards and backwards.
    """
    t = np.asarray(t, float)
    quat = qnorm(np.asarray(quat, float))
    gyro = np.asarray(gyro, float)
    dt = float(np.median(np.diff(t)))

    dq = qmul(qconj(quat[:-1]), quat[1:])
    w_filt = _rotvec(dq) / dt
    w_meas = gyro[:-1]

    if smooth > 1:                      # zero-phase: forwards then backwards
        k = np.ones(smooth) / smooth
        for c in range(3):
            f = np.convolve(w_filt[:, c], k, mode="same")
            w_filt[:, c] = np.convolve(f[::-1], k, mode="same")[::-1]

    # dominant rotation axis, from the trustworthy signal
    big = np.linalg.norm(w_meas, axis=1) > np.radians(20)
    if big.sum() < 100:
        raise ValueError("Not enough motion. Swing the sensor through a real range.")
    _, _, Vt = np.linalg.svd(w_meas[big], full_matrices=False)
    axis = Vt[0]
    planar = float(np.linalg.svd(w_meas[big], compute_uv=False)[0] ** 2 /
                   np.sum(np.linalg.svd(w_meas[big], compute_uv=False) ** 2))

    a = w_filt @ axis                   # SIGNED, so noise does not bias the mean
    b = w_meas @ axis

    n = int(round(max_lag_ms / 1e3 / dt))
    x, y = a - a.mean(), b - b.mean()
    lags = np.arange(-n, n + 1)
    cc = np.array([np.dot(x[max(0, k):len(x) + min(0, k)],
                          y[max(0, -k):len(y) + min(0, -k)]) for k in lags])
    k_best = int(lags[np.argmax(cc)])
    best = float(k_best * dt * 1e3)

    if k_best > 0:
        a_al, b_al = a[k_best:], b[:len(b) - k_best]
    elif k_best < 0:
        a_al, b_al = a[:len(a) + k_best], b[-k_best:]
    else:
        a_al, b_al = a, b

    moving = np.abs(b_al) > np.radians(20)
    slope = float(np.dot(a_al[moving], b_al[moving]) /
                  max(np.dot(b_al[moving], b_al[moving]), 1e-12))
    resid = a_al[moving] - slope * b_al[moving]
    hi = np.abs(b_al) > np.percentile(np.abs(b_al[moving]), 90)

    out = {"attenuation_slope": slope, "lag_ms": best, "lag_samples": k_best,
           "peak_rate_deg_s": float(np.degrees(np.abs(b).max())),
           "median_rate_deg_s": float(np.degrees(np.abs(b_al[moving]).mean())),
           "resid_rms_deg_s": float(np.degrees(np.sqrt(np.mean(resid ** 2)))),
           "planarity": planar,
           "slope_high_rate": float(np.dot(a_al[hi], b_al[hi]) /
                                    max(np.dot(b_al[hi], b_al[hi]), 1e-12))
           if hi.sum() > 30 else float("nan")}
    if verbose:
        _fmt("FILTER vs GYRO  (attenuation and lag, no reference needed)")
        print(f"  peak angular rate    {out['peak_rate_deg_s']:.0f} deg/s"
              f"   (mean while moving {out['median_rate_deg_s']:.0f})")
        print(f"  rotation planarity   {out['planarity']:.3f}")
        print(f"  attenuation slope    {out['attenuation_slope']:.4f}   (1.000 = none)")
        print(f"  slope, top decile    {out['slope_high_rate']:.4f}")
        print(f"  residual RMS         {out['resid_rms_deg_s']:.2f} deg/s")
        print(f"  apparent lag         {out['lag_ms']:+.1f} ms "
              f"({out['lag_samples']:+d} samples)")
        if out["attenuation_slope"] > 1.02:
            print("  -> slope above 1 is not physical for a complementary filter.")
            print("     Usually means the rotation was not planar enough for the")
            print("     axis projection, or the quaternion is very noisy.")
        elif out["attenuation_slope"] < 0.97:
            print("  -> the filter under-tracks real rotation. Peak angles will")
            print("     read low during fast movement.")
        else:
            print("  -> tracking is essentially 1:1. No meaningful attenuation.")
        if abs(out["lag_ms"]) > 20:
            print("  -> more than ~20 ms of lag; it will bias gait event timing.")
        elif abs(out["lag_samples"]) <= 1:
            print("  -> lag is at or below one sample. No measurable delay.")
    return out


# ----------------------------------------------------------------- 6. closure
def closure(t, quat, homes, cal_acc=None, verbose=True):
    """
    Return the sensor to a mechanically repeatable home pose after a maneuver.
    Any orientation difference between visits is accumulated error, needing no
    reference instrument. homes = [(t0, t1), (t2, t3), ...] in seconds.

    The error is DECOMPOSED into tilt and heading, because they mean different
    things. Tilt is observable from gravity and is what a sagittal joint angle
    is made of. Heading is not observable from gravity at all, so it absorbs
    both filter perturbation and any yaw slop in the seat -- and it does not
    affect a sagittal angle. A closure error that is nearly all heading is
    therefore much less alarming than the total suggests.
    """
    t = np.asarray(t, float); t = t - t[0]
    quat = qnorm(np.asarray(quat, float))
    means, accs = [], []
    for a_, b_ in homes:
        m = (t >= a_) & (t <= b_)
        if m.sum() < 10:
            raise ValueError(f"Home window {a_}-{b_} s has too few samples.")
        means.append(quat_mean(quat[m]))
        if cal_acc is not None:
            accs.append(np.asarray(cal_acc, float)[m].mean(axis=0))

    up = None
    if accs:
        up = accs[0] / np.linalg.norm(accs[0])

    ref = means[0]
    rows = []
    for k, q in enumerate(means[1:], 1):
        rel = qmul(qconj([ref]), [q])
        rv = np.degrees(_rotvec(rel)[0])
        tot = float(np.linalg.norm(rv))
        if up is not None:
            head = float(rv @ up)
            tilt = float(np.linalg.norm(rv - head * up))
        else:
            head = tilt = float("nan")
        rows.append((tot, tilt, abs(head)))

    out = {"n_visits": len(means),
           "errors_deg": [r[0] for r in rows],
           "tilt_deg": [r[1] for r in rows],
           "heading_deg": [r[2] for r in rows],
           "max_deg": float(max(r[0] for r in rows)) if rows else 0.0,
           "max_tilt_deg": float(max(r[1] for r in rows)) if rows else 0.0,
           "elapsed_s": [homes[k][0] - homes[0][1] for k in range(1, len(homes))]}
    if verbose:
        _fmt("ROTATION CLOSURE")
        print(f"  {'visit':<7} {'total':>8} {'tilt':>8} {'heading':>9} {'away':>8}")
        for k, ((tot, tilt, head), el) in enumerate(zip(rows, out["elapsed_s"]), 2):
            print(f"  {k:<7} {tot:8.2f} {tilt:8.2f} {head:9.2f} {el:7.1f}s")
        print(f"  {'worst':<7} {out['max_deg']:8.2f} {out['max_tilt_deg']:8.2f}")
        if up is None:
            print("  (pass cal_acc to split tilt from heading)")
        elif out["max_tilt_deg"] < 0.3 * out["max_deg"]:
            print("  -> almost all of it is HEADING. Gravity cannot observe heading,")
            print("     so this absorbs both filter perturbation and yaw slop in the")
            print("     seat. It does NOT affect a sagittal joint angle. The TILT")
            print("     column is the number that matters for your measurement.")
        else:
            print("  -> a real share is TILT, which gravity does constrain. That is")
            print("     genuine orientation error and does affect a sagittal angle.")
    return out


def tilt_series(quat, ref_mask, verbose=True):
    """
    Signed rotation angle about the dominant rotation axis, relative to a
    reference pose. This is the right measurand for a bench fixture: it is
    gravity-referenced, exactly like the digital angle gauge you compare it
    against, and it does not involve ReBAIT's PCA/sign machinery (which needs
    walking data and is meaningless on a hinge).
    """
    q = qnorm(np.asarray(quat, float))
    q0 = quat_mean(q[ref_mask])
    rel = qmul(np.tile(qconj([q0]), (len(q), 1)), q)
    rv = _rotvec(rel)

    mag = np.linalg.norm(rv, axis=1)
    big = mag > np.percentile(mag, 70)
    if big.sum() < 20:
        raise ValueError("Not enough rotation away from the reference pose.")
    _, sv, Vt = np.linalg.svd(rv[big], full_matrices=False)
    axis = Vt[0]
    if np.dot(rv[big].mean(axis=0), axis) < 0:
        axis = -axis

    planarity = float(sv[0] ** 2 / max(np.sum(sv ** 2), 1e-12))
    if verbose:
        print(f"  rotation axis planarity {planarity:.4f}  "
              f"(1.000 = pure single-axis hinge)")
        if planarity < 0.995:
            print("  -> the fixture is not rotating about a single clean axis, or the")
            print("     axis is not horizontal. Off-axis motion contaminates the angle.")
    return np.degrees(rv @ axis)


def sweep_from_quat(t, quat, gyro, commanded, min_hold=3.0, verbose=True,
                    hyst_tol=2.0, merge_tol=1.0):
    """
    Detect the holds, build a gravity-referenced tilt angle, then compare.

    Consecutive plateaus whose mean angle differs by less than merge_tol are
    MERGED. A momentary nudge during a hold splits it into fragments, and the
    fragments are all at the same angle -- they are one physical position, not
    several, so treating them separately would wreck the pairing.
    """
    t = np.asarray(t, float); t = t - t[0]
    segs = _plateaus(t, gyro, min_hold)
    if not segs:
        raise ValueError("No holds detected. Hold each angle longer, or raise "
                         "--still-thresh.")
    ref = np.zeros(len(t), bool)
    ref[segs[0][0]:segs[0][1]] = True
    ang = tilt_series(quat, ref, verbose=False)

    merged, raw_n = [segs[0]], len(segs)
    for a_, b_ in segs[1:]:
        pa, pb = merged[-1]
        if abs(ang[a_:b_].mean() - ang[pa:pb].mean()) < merge_tol:
            merged[-1] = (pa, b_)
        else:
            merged.append((a_, b_))

    if verbose:
        _fmt("STATIC ANGLE SWEEP")
        print(f"  plateaus raw {raw_n}  ->  merged {len(merged)}  "
              f"(within {merge_tol} deg)   commanded {len(commanded)}")
    tilt_series(quat, ref, verbose=verbose)

    means = np.array([ang[a_:b_].mean() for a_, b_ in merged])
    sds = np.array([ang[a_:b_].std() for a_, b_ in merged])
    return _sweep_compare(means, sds, np.asarray(commanded, float),
                          verbose=verbose, hyst_tol=hyst_tol)


# ----------------------------------------------------------------- 7. sweep
def sweep(t, angle, gyro, commanded, min_hold=3.0, verbose=True, hyst_tol=2.0):
    """
    Static angle sweep against a digital angle gauge. Detects the held plateaus
    automatically and pairs them with your commanded list, in order.
    """
    t = np.asarray(t, float); t = t - t[0]
    angle = np.asarray(angle, float)
    segs = _plateaus(t, gyro, min_hold)
    meas = np.array([angle[a:b].mean() for a, b in segs])
    sds = np.array([angle[a:b].std() for a, b in segs])
    cmd = np.asarray(commanded, float)

    return _sweep_compare(meas, sds, cmd, verbose=verbose, hyst_tol=hyst_tol)


def _sweep_compare(meas, sds, cmd, verbose=True, hyst_tol=2.0):
    if len(meas) != len(cmd):
        print("  -> COUNT MISMATCH. Detected positions:")
        for i, (m, sd) in enumerate(zip(meas, sds)):
            print(f"       {i:2d}  {m:8.2f} deg  (SD {sd:.3f})")
        print("     Re-run with a commanded list matching these positions.")
        return {"measured": meas, "sd": sds}

    meas = meas - meas[0] + cmd[0]                  # zero on the first hold
    err = meas - cmd
    slope, icpt = np.polyfit(cmd, meas, 1)
    pred = slope * cmd + icpt
    ss = 1.0 - np.sum((meas - pred) ** 2) / max(np.sum((meas - meas.mean()) ** 2), 1e-12)

    # Pair the ascending and descending visits to the SAME physical position.
    # Gauge readings almost never repeat exactly -- 60.6 going up and 60.4
    # coming down is the same block stack -- so cluster within a tolerance
    # instead of requiring exact equality, and compare the ERRORS (which
    # already account for the small difference in commanded value).
    groups = []
    for i in np.argsort(cmd):
        for g in groups:
            if abs(cmd[i] - np.mean(cmd[g])) <= hyst_tol:
                g.append(i)
                break
        else:
            groups.append([i])
    hyst = {round(float(np.mean(cmd[g])), 1): float(np.ptp(err[g]))
            for g in groups if len(g) > 1}

    out = {"commanded": cmd, "measured": meas, "sd": sds, "error": err,
           "bias": float(err.mean()), "max_abs_error": float(np.abs(err).max()),
           "slope": float(slope), "r2": float(ss),
           "hysteresis": hyst,
           "max_hysteresis": float(max(hyst.values())) if hyst else 0.0}
    if verbose:
        print(f"\n  {'cmd':>8} {'meas':>9} {'err':>8} {'SD':>7}")
        for c, m, e, s in zip(cmd, meas, err, sds):
            print(f"  {c:8.1f} {m:9.2f} {e:+8.2f} {s:7.3f}")
        print(f"\n  mean bias            {out['bias']:+.3f} deg")
        print(f"  max abs error        {out['max_abs_error']:.3f} deg")
        print(f"  slope                {out['slope']:.5f}   (1.0 = no scale error)")
        print(f"  linearity R^2        {out['r2']:.6f}")
        if hyst:
            print(f"  max hysteresis       {out['max_hysteresis']:.3f} deg "
                  f"({len(hyst)} up/down pairs)")
        else:
            print("  hysteresis           not computed - no up/down pairs found")
            print("     Either you ran a single-direction sweep, or the up and down")
            print(f"     gauge readings differ by more than {hyst_tol} deg. Raise")
            print("     --hyst-tol if the physical positions really were the same.")
        if abs(out["slope"] - 1.0) > 0.01:
            print("  -> scale error above 1%. Check the gauge, and check that the")
            print("     rotation axis is truly the one the angle is extracted about.")
    return out


# ----------------------------------------------------------------- 8. redonning
def redonning(angles, verbose=True):
    """Spread in reported neutral angle across repeated don-doff cycles."""
    a = np.asarray(angles, float)
    out = {"n": len(a), "mean": float(a.mean()),
           "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
           "range": float(np.ptp(a))}
    if verbose:
        _fmt("RE-DONNING REPEATABILITY")
        print(f"  cycles               {out['n']}")
        print(f"  neutral angle        {out['mean']:.2f} deg")
        print(f"  SD                   {out['sd']:.3f} deg")
        print(f"  range                {out['range']:.3f} deg")
        print("  -> this is frequently LARGER than sensor-vs-mocap error, and it is")
        print("     what decides whether anyone but you can use the system.")
    return out


# ----------------------------------------------------------------- CLI
def main():
    from imu_receiver import load

    ap = argparse.ArgumentParser(description="No-mocap bench checks for one IMU.")
    ap.add_argument("npz")
    ap.add_argument("--device", default=None)
    ap.add_argument("--test", default="all",
                    choices=["all", "link", "gravity", "drift", "mag",
                             "gyrofilter", "closure", "sweep", "holds"])
    ap.add_argument("--commanded", type=float, nargs="+")
    ap.add_argument("--home", type=float, nargs=2, action="append",
                    metavar=("T0", "T1"))
    ap.add_argument("--static", type=float, default=5.0)
    ap.add_argument("--still-thresh", type=float, default=3.0,
                    dest="still_thresh", help="deg/s below which a sample is static")
    ap.add_argument("--merge-tol", type=float, default=1.0, dest="merge_tol",
                    help="deg within which split plateaus are merged")
    ap.add_argument("--hyst-tol", type=float, default=2.0, dest="hyst_tol",
                    help="deg within which up/down holds count as the same position")
    ap.add_argument("--settle", type=float, default=1.0,
                    help="seconds to discard after each movement (filter settling)")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    did, d = load(a.npz, a.device)
    print(f"device {did}   {len(d['t'])} samples")
    global STILL_GYRO
    STILL_GYRO = np.radians(a.still_thresh)
    want = a.test

    if want == "holds":
        gyro_diagnostic(d["t"], d["gyro"])
        return

    if want in ("all", "link"):
        link(d["t"], d["seq"])
    if want in ("all", "gravity"):
        try:
            gravity(d["quat"], d["cal_acc"], d["gyro"], t=d["t"],
                    settle_s=a.settle)
        except ValueError as e:
            print(f"\n  gravity: skipped ({e})")
    if want in ("all", "drift"):
        drift(d["t"], d["quat"], d["cal_acc"])
    if want in ("all", "mag"):
        try:
            magnetic(d["mag"], d["cal_acc"], d["gyro"])
        except ValueError as e:
            print(f"\n  magnetic: skipped ({e})")
    if want in ("all", "gyrofilter"):
        try:
            gyrofilter(d["t"], d["quat"], d["gyro"])
        except ValueError as e:
            print(f"\n  gyrofilter: skipped ({e})")
    if want == "closure":
        if not a.home:
            ap.error("--closure needs at least two --home T0 T1 windows")
        closure(d["t"], d["quat"], [tuple(h) for h in a.home],
                cal_acc=d["cal_acc"])
    if want == "sweep":
        if not a.commanded:
            ap.error("--sweep needs --commanded")
        sweep_from_quat(d["t"], d["quat"], d["gyro"], a.commanded,
                        hyst_tol=a.hyst_tol, merge_tol=a.merge_tol)

    if a.plot:
        import matplotlib.pyplot as plt
        r = drift(d["t"], d["quat"], d["cal_acc"], verbose=False)
        plt.figure(figsize=(10, 4))
        plt.plot(r["t"] / 60, r["tilt"], label="tilt")
        plt.plot(r["t"] / 60, r["heading"], label="heading")
        plt.xlabel("minutes"); plt.ylabel("deg"); plt.legend()
        plt.title("stationary drift"); plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()