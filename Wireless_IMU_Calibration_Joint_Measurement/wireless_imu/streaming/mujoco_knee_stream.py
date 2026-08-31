#!/usr/bin/env python3
"""
mujoco_knee_stream.py  -  live knee angle in MuJoCo, from two ReBAIT sensors.

SCOPE: exactly what you have hardware for right now -- thigh + shank, knee
flexion/extension only. No ankle, hip, or pelvis code paths, because those
sensors don't exist yet. Extending to the toe (ankle angle) later is the
same pattern again with a third segment; not built here so there's nothing
half-finished sitting unused in the meantime.

WHAT THIS IS BUILT ON, AND WHY
--------------------------------
  RECEIVER      imu_receiver.UdpImuCallback -- the 92-byte quaternion packet,
                dedicated-thread architecture. Verified on real hardware:
                0.00% loss solo on two sensors AND simultaneously on one
                shared socket (multi_link_diag.py). NOT
                wireless_imu_listener.py's 45-byte Euler format -- that would
                force a quaternion->Euler->quaternion round trip for no
                reason, reintroducing exactly the gimbal/convention risk the
                REMAP_TO_FILTER_FRAME investigation spent a long session
                ruling out.

  CALIBRATION   rebait_calibration_adapter.FunctionalCalibration -- ReBAIT's
                validated gravity + gyro-PCA math (shank_calibration.py),
                behind the same public interface Shah's wireless_imu_
                streaming.py already calls. Verified here against synthetic
                ground truth: single-segment angle recovery to <0.001 deg
                error with an asymmetric (gait-like) calibration swing;
                two-segment relative knee angle recovery to <0.001 deg error
                across two DIFFERENTLY MOUNTED sensors (real functional
                alignment, not assumed); and a resolve_sign() method (added
                during this validation -- the original adapter could REPORT
                sign ambiguity but had no way to FIX it) confirmed to detect
                and correct a case where the automatic heuristic picked the
                wrong direction.

  FRAME LOOP    Drains EVERY packet available each frame and keeps only the
                most recent per segment, rather than popping one packet per
                frame. This matters: at ~333 Hz arrival against MuJoCo's
                render rate, popping one-per-frame reproduces the exact 65%
                "loss" artifact link_diag.py's --load test produced earlier
                in this project -- that was a consumer-design bug, not a
                link problem, and the fix is the same here.

CALIBRATION PROTOCOL (per segment)
-------------------------------------
  Phase 1 -- gravity     : stand/sit still, ~5 s
  Phase 2 -- rotation     : seated knee swings, large amplitude, THEN a few
                           gait-like steps if you can manage them un-tethered
                           -- symmetric swings alone leave the sign a coin
                           flip (confirmed on real donning data earlier:
                           identical calibration flags produced opposite
                           angle conventions across two re-donnings)
  Phase 3 -- zero         : neutral standing pose, ~5 s
  Phase 4 -- sign check   : one deliberate, slow, KNOWN-direction flexion.
                           resolve_sign() uses this to fix the sign if the
                           automatic heuristic got it wrong -- do not skip
                           this even if sign_ambiguous reads False; the
                           validated failure mode is silent, not flagged.

USAGE
    python mujoco_knee_stream.py --thigh-id 6 --shank-id 5
    python mujoco_knee_stream.py --thigh-id 6 --shank-id 5 --model-dir /path/to/myo_sim/body
"""

import argparse
import importlib.util
import os
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from imu_receiver import UdpImuCallback
from rebait_calibration_adapter import FunctionalCalibration, relative_delta, _wxyz_to_rot


# =============================================================================
# MODEL LOCATION  (same pattern as the original wireless_imu_streaming.py)
# =============================================================================
_MYOBODY_RELATIVE = Path("simhive") / "myo_sim" / "body"
MYOBODY_XML = "myobody.xml"
KNEE_JOINT = "knee_angle_r"


def _find_myobody_dir():
    spec = importlib.util.find_spec("myosuite")
    if spec is None or spec.origin is None:
        return None
    candidate = Path(spec.origin).parent / _MYOBODY_RELATIVE
    return candidate if (candidate / MYOBODY_XML).exists() else None


def _build_xml():
    return "\n".join([
        '<mujoco model="knee_imu_stream">',
        '  <option gravity="0 0 0"/>',
        '  <visual><headlight ambient="0.50 0.50 0.50"/>'
        '<global offwidth="1280" offheight="720"/></visual>',
        f'  <include file="{MYOBODY_XML}"/>',
        '  <worldbody>',
        '    <light pos="0 2 6" diffuse="0.9 0.9 0.9" dir="0 -0.3 -1"/>',
        '    <geom name="lab_floor" type="plane" size="6 6 0.1" '
        'rgba="0.35 0.35 0.35 1" contype="0" conaffinity="0"/>',
        '  </worldbody>',
        '</mujoco>',
    ])


# =============================================================================
# RECEIVER ADAPTER: drain everything, keep the latest per segment
# =============================================================================

class LatestQuatState:
    """
    Wraps UdpImuCallback for live rendering rather than full-fidelity
    recording. imu_receiver.record() keeps every sample for later analysis;
    a live viewer only ever needs the newest one. Draining fully each poll
    and discarding all but the last packet per device is the fix for the
    one-pop-per-frame consumer pattern that produced 65% "loss" under
    link_diag.py --load -- that was never a link problem, it was arrival
    rate (~333 Hz) outrunning a consumer that only took one sample per call.
    """

    def __init__(self, cb: UdpImuCallback):
        self.cb = cb
        self.latest = {}       # imu_id (str, e.g. "imu05") -> raw packet dict
        self.updated = {}      # imu_id -> bool, has at least one packet ever arrived
        self.drained_this_poll = {}

    def poll(self):
        """Call once per frame. Drains the callback's buffer completely."""
        counts = {}
        while self.cb.packetAvailable():
            did, pkt = self.cb.getNextRaw()
            self.latest[did] = pkt
            self.updated[did] = True
            counts[did] = counts.get(did, 0) + 1
        self.drained_this_poll = counts
        return counts

    def rotation(self, did):
        """Latest orientation for a device, as a scipy Rotation, or None."""
        pkt = self.latest.get(did)
        if pkt is None:
            return None
        return _wxyz_to_rot(pkt["quat"])   # packet quat is (w,x,y,z)

    def gravity_gyro(self, did):
        """Latest (cal_acc, gyro) for calibration sampling, or (None, None)."""
        pkt = self.latest.get(did)
        if pkt is None:
            return None, None
        return pkt["cal_acc"], pkt["gyro"]

    def ready(self, did):
        return self.updated.get(did, False)


# =============================================================================
# CALIBRATION  (mirrors the 4-phase protocol in the module docstring)
# =============================================================================

def _collect(state: LatestQuatState, did, duration_s, want="gravity"):
    """Poll and accumulate samples for duration_s seconds."""
    out = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration_s:
        state.poll()
        acc, gyro = state.gravity_gyro(did)
        if acc is not None:
            out.append(acc if want == "gravity" else gyro)
        time.sleep(0.002)
    return out


def calibrate_segment(name, state: LatestQuatState, did, gravity_s=5.0, rotation_s=10.0):
    """Interactive 3-phase build + zero. Phase 4 (sign check) is separate --
    call resolve_segment_sign() after this, using a fresh deliberate motion."""
    print(f"\n=== Calibrating: {name.upper()} (device {did}) ===")
    fc = FunctionalCalibration()

    input(f"  [Phase 1] Stand/sit still for {gravity_s:.0f}s. Press Enter: ")
    for a in _collect(state, did, gravity_s, "gravity"):
        fc.add_gravity_frame(*a)

    print(f"  [Phase 2] Swing the joint through its range for {rotation_s:.0f}s")
    print(f"            (large seated swings; add a few steps if you can --")
    print(f"            symmetric motion alone leaves the sign ambiguous).")
    input("  Press Enter to begin: ")
    for g in _collect(state, did, rotation_s, "gyro"):
        fc.add_rotation_frame(*g)

    fc.build()
    print(f"  axes built. sign_margin={fc.sign_margin:+.3f} "
          f"ambiguous={fc.sign_ambiguous}")
    print(f"  svd_ratio={fc.svd_ratio:.2f} (want > ~3; a low ratio means the")
    print(f"    swing was NOT cleanly single-axis -- other rotation, not just")
    print(f"    the joint's own axis, was excited, e.g. hip rotation/abduction")
    print(f"    mixed into what should be pure flexion/extension)")
    print(f"  gravity_spread={fc.gravity_spread:.4f}")
    if fc.svd_ratio < 3.0:
        print(f"  *** LOW svd_ratio ({fc.svd_ratio:.2f}). This segment's swing")
        print(f"  *** likely contaminated the joint axis with off-axis motion --")
        print(f"  *** a probable direct cause of downstream alignment mismatch,")
        print(f"  *** independent of the sign question below.")
    if fc.sign_ambiguous:
        print("  *** sign is ambiguous from this swing alone -- the sign")
        print("  *** check in Phase 4 is not optional for this segment.")

    input(f"  [Phase 3] Return to neutral, hold {gravity_s:.0f}s. Press Enter: ")
    rots = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < gravity_s:
        state.poll()
        r = state.rotation(did)
        if r is not None:
            rots.append(r)
        time.sleep(0.002)
    from rebait_calibration_adapter import _average_rotation
    fc.set_zero(_average_rotation(rots))
    print(f"  [{name}] zeroed from {len(rots)} samples.")
    return fc


def resolve_segment_sign(name, state: LatestQuatState, did, fc: FunctionalCalibration,
                         hold_s=4.0, axis=2):
    """
    Phase 4. One deliberate, slow, KNOWN-direction flexion, held at the end.
    Do this even if sign_ambiguous read False after build() -- the validated
    failure mode (D1 donning data, and the synthetic seed-0 case found while
    testing this adapter) is a WRONG sign with no automatic warning, not a
    flagged one.
    """
    print(f"\n  [{name}] SIGN CHECK. Perform one slow, deliberate FLEXION")
    print(f"  and hold it for the last {hold_s:.0f}s of the window.")
    input("  Press Enter, then flex and hold: ")
    rots = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < hold_s:
        state.poll()
        r = state.rotation(did)
        if r is not None:
            rots.append(r)
        time.sleep(0.002)
    if not rots:
        print("  *** no samples captured -- sign NOT verified. Re-run this step.")
        return
    from rebait_calibration_adapter import _average_rotation
    held = _average_rotation(rots[-max(1, len(rots) // 3):])   # last third = the held pose
    flipped = fc.resolve_sign(held, expect_positive=True, axis=axis)
    check_angle = fc.get_primary_angle(held, axis=axis)
    print(f"  held-pose angle now reads {check_angle:+.1f} deg "
          f"({'FLIPPED' if flipped else 'was already correct'})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Live knee angle in MuJoCo from two IMUs.")
    ap.add_argument("--thigh-id", type=int, required=True)
    ap.add_argument("--shank-id", type=int, required=True)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--gravity-s", type=float, default=5.0)
    ap.add_argument("--rotation-s", type=float, default=10.0)
    a = ap.parse_args()

    thigh_dev = f"imu{a.thigh_id:02d}"
    shank_dev = f"imu{a.shank_id:02d}"

    cb = UdpImuCallback(port=a.port, expected_ids=(a.thigh_id, a.shank_id))
    cb.enable()
    cb.attach()
    state = LatestQuatState(cb)

    print("Waiting for both sensors ...")
    while not (state.ready(thigh_dev) and state.ready(shank_dev)):
        state.poll()
        time.sleep(0.05)
    print(f"  thigh ({thigh_dev}) and shank ({shank_dev}) both streaming.")

    thigh_fc = calibrate_segment("thigh", state, thigh_dev, a.gravity_s, a.rotation_s)
    shank_fc = calibrate_segment("shank", state, shank_dev, a.gravity_s, a.rotation_s)
    resolve_segment_sign("thigh", state, thigh_dev, thigh_fc)
    resolve_segment_sign("shank", state, shank_dev, shank_fc)

    # --- AXIS ALIGNMENT CHECK -------------------------------------------
    # NOT proven to be the cause of any specific observed offset -- an
    # attempt to reproduce a large misalignment from imperfect (off-axis)
    # swing motion synthetically did NOT actually produce one, PCA turned
    # out more robust to that than expected. This is a DIAGNOSTIC, not a
    # confirmed fix: it measures directly, on YOUR real calibration data,
    # whether the thigh's and shank's independently-derived joint axes
    # actually agree on the same physical direction -- mirroring the
    # equivalent check in joint_angle.py's calibrate_pair(), which this
    # script's simpler two-segment path never replicated.
    # --- SIMULTANEOUS ZERO ---------------------------------------------
    # calibrate_segment() zeros each segment separately, minutes apart. The
    # knee angle is a RELATIVE quantity (relative_delta = thigh^-1 * shank),
    # so it is only correct if both zeros describe the SAME physical pose.
    # If your posture differed even slightly between the thigh's neutral
    # hold and the shank's neutral hold, that mismatch becomes a constant
    # offset on every reading from then on -- a stable ~40 deg "knee angle"
    # while visibly standing straight is exactly that signature, not a live
    # tracking error. This re-zeros BOTH segments from ONE shared moment,
    # which removes that offset regardless of what happened earlier.
    print("\n=== FINAL SIMULTANEOUS ZERO ===")
    print("Stand in your normal neutral pose -- the SAME posture you want")
    print("to read as 0 deg knee angle. Both segments will be zeroed")
    print("together from this one moment, which is what actually matters")
    print("for a RELATIVE angle -- not whether each looked right alone.")
    input("Press Enter, then hold neutral for 5s: ")
    rots_t, rots_s = [], []
    t0 = time.monotonic()
    while time.monotonic() - t0 < 5.0:
        state.poll()
        r = state.rotation(thigh_dev)
        if r is not None:
            rots_t.append(r)
        r = state.rotation(shank_dev)
        if r is not None:
            rots_s.append(r)
        time.sleep(0.002)
    from rebait_calibration_adapter import _average_rotation, relative_delta as _rd
    thigh_fc.set_zero(_average_rotation(rots_t))
    shank_fc.set_zero(_average_rotation(rots_s))

    # Verify immediately, rather than discovering an offset by watching the
    # viewer: report the knee angle AT the pose that was just called zero.
    check_rel = _rd(thigh_fc.get_zero(), rots_t[-1], shank_fc.get_zero(), rots_s[-1])
    check_deg = shank_fc.project_angle(check_rel, axis=2)
    print(f"knee angle at the pose just zeroed: {check_deg:+.1f} deg "
          f"(should read ~0)")
    if abs(check_deg) > 5.0:
        print("*** still nonzero at neutral -- something else needs fixing")
        print("*** before trusting the live angle. Do not treat the run")
        print("*** below as validated if this is large.")

    # --- AXIS ALIGNMENT CHECK -------------------------------------------
    # Moved here deliberately: this MUST use the shared simultaneous zero,
    # not each segment's own individually-captured Phase-3 zero. An earlier
    # version computed this using get_zero() BEFORE the simultaneous-zero
    # fix ran -- the exact same inconsistent-reference-frame bug the
    # simultaneous zero exists to fix, corrupting this diagnostic itself.
    # (Caught because two independent sessions reproduced 72.9 deg and
    # 73.4 deg almost exactly -- real swing-quality noise would not repeat
    # that precisely; a structural bug using the same flawed reference
    # frame every time would.)
    # Gravity (Y) axis alignment -- a SEPARATE check from the joint axis.
    # Gravity is trivial: "down" is the same direction for both sensors
    # regardless of segment, swing quality, or anatomy. If this ALSO shows
    # large disagreement, the problem is in the shared zero/orientation
    # pipeline itself, not anything specific to the joint-axis (PCA) step --
    # that would rule out swing quality and anatomy as explanations too,
    # and point back at something in how orientation is being handled in
    # common for both segments.
    world_grav_thigh = thigh_fc.get_zero().apply(thigh_fc.axes["Y"])
    world_grav_shank = shank_fc.get_zero().apply(shank_fc.axes["Y"])
    grav_dot = float(np.clip(abs(np.dot(world_grav_thigh, world_grav_shank)), 0, 1))
    grav_align_deg = float(np.degrees(np.arccos(grav_dot)))
    print(f"\ngravity axis alignment (thigh vs shank): {grav_align_deg:.1f} deg "
          f"(expect near 0 -- 'down' is the same direction for both)")

    world_axis_thigh = thigh_fc.get_zero().apply(thigh_fc.axes["Z"])
    world_axis_shank = shank_fc.get_zero().apply(shank_fc.axes["Z"])
    align_dot = float(np.clip(abs(np.dot(world_axis_thigh, world_axis_shank)), 0, 1))
    align_deg = float(np.degrees(np.arccos(align_dot)))
    print(f"joint axis alignment (thigh vs shank): {align_deg:.1f} deg")
    if grav_align_deg > 15.0:
        print("*** Gravity axes ALSO disagree by a large amount. This is NOT")
        print("*** specific to the joint-axis/PCA step -- something in the")
        print("*** shared zero/orientation handling is wrong for one or both")
        print("*** segments. Swing quality and anatomy are ruled out by this.")
    elif align_deg > 15.0:
        print("*** Gravity agrees well but the JOINT axis does not. This is")
        print("*** specific to the PCA/rotation-axis step, not general frame")
        print("*** handling -- points at either a genuine biomechanical")
        print("*** difference in how hip vs knee motion was performed, or")
        print("*** something in how the joint axis itself is determined.")
    if align_deg > 15.0:
        print("*** LARGE misalignment. The two segments' calibration swings")
        print("*** did not agree on the same physical rotation axis -- this")
        print("*** alone can produce errors of a similar magnitude in every")
        print("*** knee-angle reading. Re-run Phase 2 for both segments with")
        print("*** larger, cleaner, more purely single-axis motion before")
        print("*** trusting the live angle.")
    elif align_deg > 5.0:
        print("(moderate -- worth a cleaner recalibration if the live angle")
        print(" still looks wrong)")

    print("\nCalibration complete. Launching viewer ...\n")

    # --- MuJoCo import deferred to here: everything above is fully testable
    # without MuJoCo installed. ---
    import mujoco
    import mujoco.viewer

    if a.model_dir is not None:
        model_dir = Path(a.model_dir)
    else:
        model_dir = _find_myobody_dir()
        if model_dir is None:
            raise SystemExit("Could not auto-detect myobody. Pass --model-dir.")

    os.chdir(model_dir)
    model = mujoco.MjModel.from_xml_string(_build_xml())
    data = mujoco.MjData(model)

    knee_addr = None
    for i in range(model.njnt):
        if model.joint(i).name == KNEE_JOINT:
            knee_addr = model.jnt_qposadr[i]
    if knee_addr is None:
        print(f"  [WARN] joint '{KNEE_JOINT}' not found in the model.")

    mujoco.mj_forward(model, data)

    frame = 0
    drop_warn_shown = False
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 150

        while viewer.is_running():
            counts = state.poll()   # drains EVERYTHING waiting, every frame

            if not drop_warn_shown and any(c > 20 for c in counts.values()):
                print(f"  [note] drained {counts} packets in one frame -- "
                      f"the render loop is falling behind arrival rate. "
                      f"Fine occasionally; a persistent pattern means the "
                      f"frame rate itself needs attention, not this drain.")
                drop_warn_shown = True

            rot_t = state.rotation(thigh_dev)
            rot_s = state.rotation(shank_dev)
            knee_deg = 0.0
            if rot_t is not None and rot_s is not None:
                hip_rel = relative_delta(thigh_fc.get_zero(), rot_t,
                                         shank_fc.get_zero(), rot_s)
                knee_deg = shank_fc.project_angle(hip_rel, axis=2)
                if knee_addr is not None:
                    data.qpos[knee_addr] = np.radians(knee_deg)

            frame += 1
            if frame % 60 == 0:
                print(f"  frame {frame}   knee = {knee_deg:+6.1f} deg", end="\r")

            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.002)

    cb.detach()
    cb.close()


if __name__ == "__main__":
    main()