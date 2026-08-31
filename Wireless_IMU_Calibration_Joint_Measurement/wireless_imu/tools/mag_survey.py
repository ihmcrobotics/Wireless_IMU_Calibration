#!/usr/bin/env python3
"""
mag_survey.py  -  map the magnetic field across the positions you actually use.

THE KEY IDEA
------------
Hold the sensor in the SAME ORIENTATION at every position. Then a hard-iron
offset is one constant vector in the sensor frame, present identically in every
reading -- so when you subtract one position's mean field vector from another's,
it CANCELS EXACTLY:

    m_i - m_ref = (m_true_i + b) - (m_true_ref + b) = m_true_i - m_true_ref

That means this survey works even though your calibration does not reproduce.
It measures the real difference in the environment, immune to the sensor's own
offset. The absolute |m| is contaminated; the differences are not.

WHAT IT REPORTS
  delta         magnitude of the field difference from the reference position
  angle         angle between the field vector there and at the reference.
                THIS IS THE HEADING ERROR you would incur by moving there.
  dip           angle between the field and gravity, for reference

USAGE
    python3 mag_survey.py
    python3 mag_survey.py --seconds 20 --ref middle
    python3 mag_survey.py --positions seated standing middle --ref middle
"""

import argparse
import time

import numpy as np

from imu_receiver import UdpImuCallback

DEFAULT_POSITIONS = ["seated", "standing", "middle",
                     "walk_start", "walk_mid", "walk_end"]

HINTS = {
    "seated": "sit as you would for a trial; sensor at SHANK height, under the desk",
    "standing": "stand at the desk; sensor at shank height",
    "middle": "middle of the office, away from desk, chair and cabinets",
    "walk_start": "start of your walking path, shank height",
    "walk_mid": "midpoint of the walking path",
    "walk_end": "end of the walking path",
}


def collect(cb, seconds, still_deg_s=5.0):
    """Gather magnetometer and accelerometer means while the sensor is still."""
    mags, accs, moved = [], [], 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        if cb.packetAvailable():
            _, p = cb.getNextRaw()
            if np.degrees(np.linalg.norm(p["gyro"])) > still_deg_s:
                moved += 1
                continue
            mags.append(p["mag"])
            accs.append(p["cal_acc"])
        else:
            time.sleep(0.0005)
    if len(mags) < 200:
        raise RuntimeError(f"only {len(mags)} still samples - hold the sensor "
                           f"steady for the whole {seconds:.0f} s")
    m = np.asarray(mags)
    a = np.asarray(accs)
    return m.mean(axis=0), a.mean(axis=0), len(mags), moved


def dip_deg(m, a):
    """Angle between the field and the horizontal plane, from gravity."""
    c = np.dot(m, a) / (np.linalg.norm(m) * np.linalg.norm(a))
    return 90.0 - np.degrees(np.arccos(np.clip(c, -1, 1)))


def main():
    ap = argparse.ArgumentParser(description="Magnetic survey of your test positions.")
    ap.add_argument("--positions", nargs="+", default=DEFAULT_POSITIONS)
    ap.add_argument("--ref", default="middle", help="reference position name")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--id", type=int, default=5)
    ap.add_argument("--out", default="mag_survey.npz")
    a = ap.parse_args()

    if a.ref not in a.positions:
        ap.error(f"--ref '{a.ref}' is not in the position list")

    print("=" * 66)
    print("MAGNETIC SURVEY")
    print("=" * 66)
    print("CRITICAL: hold the sensor in the SAME ORIENTATION at every position.")
    print("Pick one edge of the enclosure and point it at the same wall each")
    print("time. The whole method depends on this - it is what makes the")
    print("comparison immune to your uncalibrated hard-iron offset.")
    print()
    print("BEST METHOD: set the sensor FLAT ON THE FLOOR at each position, with")
    print("one edge against a straightedge you point at the same wall. The floor")
    print("fixes tilt, the straightedge fixes heading, and neither depends on how")
    print("steady your hand is. Shank height is where the sensor lives in a trial,")
    print("but the floor is close enough and vastly more repeatable.")
    print("=" * 66)

    cb = UdpImuCallback(port=a.port, expected_ids=(a.id,))
    cb.enable()
    cb.attach()

    res = {}
    try:
        for name in a.positions:
            hint = HINTS.get(name, "")
            print(f"\n--- {name} ---")
            if hint:
                print(f"    {hint}")
            input("    in position, sensor aligned and still, press Enter: ")
            while cb.packetAvailable():          # drop anything buffered
                cb.getNextRaw()
            print(f"    recording {a.seconds:.0f} s ... hold still")
            try:
                m, acc, n, moved = collect(cb, a.seconds)
            except RuntimeError as e:
                print(f"    SKIPPED: {e}")
                continue
            res[name] = {"mag": m, "acc": acc, "n": n}
            print(f"    ok  |m| {np.linalg.norm(m):6.1f} mG   dip {dip_deg(m, acc):+6.2f} deg"
                  f"   ({n} still samples, {moved} discarded)")
    finally:
        cb.detach()
        cb.close()

    if a.ref not in res:
        print("\nReference position was not captured. Nothing to compare against.")
        return

    ref = res[a.ref]["mag"]
    ref_a = res[a.ref]["acc"]
    ref_u = ref_a / np.linalg.norm(ref_a)

    print("\n" + "=" * 66)
    print(f"RESULTS   (reference = {a.ref})")
    print("=" * 66)
    print(f"  {'position':<12} {'|m| mG':>8} {'delta mG':>9} {'angle deg':>10} "
          f"{'dip deg':>8} {'tilt err':>9}")
    worst_ang, worst_name, worst_tilt = 0.0, None, 0.0
    for name in a.positions:
        if name not in res:
            continue
        m, acc = res[name]["mag"], res[name]["acc"]
        d = np.linalg.norm(m - ref)
        c = np.dot(m, ref) / (np.linalg.norm(m) * np.linalg.norm(ref))
        ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
        if ang > worst_ang:
            worst_ang, worst_name = ang, name
        # ORIENTATION CHECK. Gravity in the sensor frame fixes the sensor's
        # tilt exactly, so comparing it against the reference measures how
        # badly the orientation actually matched. Without this the whole
        # hard-iron cancellation is an unverified assumption.
        u = acc / np.linalg.norm(acc)
        tilt = np.degrees(np.arccos(np.clip(np.dot(u, ref_u), -1, 1)))
        worst_tilt = max(worst_tilt, tilt)
        if tilt > 5.0:
            worst_ang = worst_ang           # do not let a mismatched pose set the verdict
        elif ang > worst_ang:
            worst_ang, worst_name = ang, name
        flag = "  ORIENTATION MISMATCH" if tilt > 5.0 else ""
        tag = "  <- reference" if name == a.ref else flag
        print(f"  {name:<12} {np.linalg.norm(m):8.1f} {d:9.1f} {ang:10.2f} "
              f"{dip_deg(m, acc):8.2f} {tilt:9.2f}{tag}")

    print()
    print("  'delta' and 'angle' are hard-iron immune ONLY where the orientation")
    print("  truly matched. 'tilt err' is that match, measured from gravity:")
    print("  the angle between the sensor's tilt there and at the reference.")
    print()
    if worst_tilt > 5.0:
        print(f"  *** ORIENTATION NOT HELD ({worst_tilt:.1f} deg worst mismatch).")
        print("  *** With an uncorrected hard-iron offset, tilting the sensor changes")
        print("  *** |m| and dip on its own - so those rows measure your hand, not")
        print("  *** the room. Repeat with the sensor FLAT ON THE FLOOR at each")
        print("  *** position, one edge against a straightedge pointed at the same")
        print("  *** wall. Floor fixes tilt, straightedge fixes heading.")
        print()
    elif worst_tilt > 2.0:
        print(f"  orientation held to {worst_tilt:.1f} deg - acceptable, but tighter is better.")
        print()
    else:
        print(f"  orientation held to {worst_tilt:.1f} deg - good, the comparison is valid.")
        print()
    if worst_name:
        print(f"  worst position: {worst_name}, {worst_ang:.2f} deg from the reference")
        if worst_ang < 3:
            print("  -> the field is uniform across your working area. A magnetometer")
            print("     would be usable here if it were properly calibrated.")
        elif worst_ang < 10:
            print("  -> moderate variation. Heading would wander by a few degrees as")
            print("     you move. Tolerable for sagittal angle, not for heading work.")
        else:
            print("  -> STRONGLY disturbed. Heading is unreliable across this area,")
            print("     and walking through it would drag the estimate around")
            print("     continuously. Run magnetometer-free: sagittal shank angle")
            print("     barely uses heading, and your drift test already showed")
            print("     gravity plus gyro hold attitude on their own.")

    np.savez_compressed(a.out,
                        names=np.array(list(res.keys())),
                        mags=np.array([res[k]["mag"] for k in res]),
                        accs=np.array([res[k]["acc"] for k in res]))
    print(f"\n  saved {a.out}")


if __name__ == "__main__":
    main()