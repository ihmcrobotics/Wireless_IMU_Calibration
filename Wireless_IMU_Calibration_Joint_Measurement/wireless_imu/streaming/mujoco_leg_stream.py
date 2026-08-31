#!/usr/bin/env python3
"""
mujoco_leg_stream.py  -  live knee + ankle + hip-flexion angle in MuJoCo,
from up to four ReBAIT sensors (thigh, shank, toe, pelvis).

PELVIS IS STRUCTURALLY DIFFERENT FROM THE OTHER THREE SEGMENTS. The pelvis
does not flex about its own joint axis -- there is no swing to run PCA on.
So pelvis calibration is GRAVITY + ZERO ONLY: no Phase 2 swing, no build(),
no axes of its own. Hip angle instead reuses the THIGH's already-built axes
(from its knee-swing calibration) applied to a pelvis-relative rotation --
the same "one segment calibrated once, reused for two joints" principle
already used for the shank (knee + ankle), just with thigh doing double
duty for knee + hip instead.

SCOPE: HIP FLEXION + AB/ADDUCTION + ROTATION (drift-prone, see below).
The pelvis's own 3-DOF is still NOT implemented -- deliberately out of
scope.

HIP ROTATION IS STRUCTURALLY DIFFERENT FROM THE OTHER TWO DOFs, not just
harder to isolate. Flexion and ab/adduction axes are roughly HORIZONTAL,
so gravity-orthogonalization gives real, independent information to find
them. Rotation's axis is roughly VERTICAL -- nearly the same direction as
gravity itself -- so that trick cannot apply; there is no accelerometer-
based correction possible for rotation about gravity's own direction,
matching McGrath & Stirling 2022's "practical non-identifiability #2".

Consequence: hip rotation is computed by projecting onto thigh_fc.axes["Y"]
(gravity, already established -- NO new calibration swing needed for this
DOF), which is mathematically IDENTICAL to raw, uncorrected gyro
integration about the vertical axis (validated synthetically: 0.000 deg
difference). It WILL drift, with nothing to correct it. This is not a
stable absolute angle -- it is a short-burst RELATIVE measurement, reset
manually by pressing 'z' in the viewer while standing at neutral. Expect
roughly sub-degree to a few degrees of drift over 5-15s on a well-nulled
board, growing to tens of degrees over a full minute on a bad session --
see the drift table computed from this project's own measured gyro
bias/noise before this was built.

SIGN for hip rotation is UNVERIFIED as of this writing -- written to
qpos with no flip applied, unlike hip_abad (which needed one, found only
after real-hardware testing). Do not assume the same convention; verify
with a known rotation the same way.

AB/ADDUCTION uses FunctionalCalibration.find_second_axis(): a SECOND,
independent swing (leg out to the side and back) on the SAME thigh
sensor already calibrated for flexion. Deliberately orthogonalized only
against gravity, NOT against the flexion axis -- see find_second_axis()'s
own docstring for why forcing that second orthogonality would silently
collapse the "independently measured" axis into the mathematically
trivial cross(gravity, flexion) result, defeating the point of measuring
it at all. Validated against synthetic ground truth (0.0000 deg error
recovering known commanded ab/adduction angles) AND confirmed on real
hardware with two independently-designed diagnostic tools that made no
shared assumptions (hip_abad_align_test.py and flex_vs_abad_raw_check.py)
before being wired in here.

The ab/adduction axis gets its OWN sign check (a deliberate abduction,
held) -- unlike hip flexion, which could reuse thigh_fc's existing
flexion-context sign check because both read the identical axis. This is
a genuinely new axis, so it needs its own verification against a known
direction, the same as every other axis in this whole pipeline.

Internal/external rotation is NOT implemented. Do not derive it from
axes["X"] or any other leftover/derived direction -- see the module
history above for why an unverified sign on a derived axis is exactly
the kind of thing that ships a plausible-looking but wrong number.

Extends mujoco_knee_stream.py (thigh+shank, knee only) by one segment.
Still no hip/pelvis code -- that sensor doesn't exist yet, and hip
calibration is structurally different (no swing to run PCA on; would need
its own zero-only path, not built here).

THE ONE THING THAT IS NOT "JUST REPEAT THE PATTERN": the shank is SHARED
between both joints. If it were calibrated twice -- once paired with the
thigh swing, once paired with an ankle swing -- you would get two
DIFFERENT, independently-derived shank axes with no guarantee they agree,
which is exactly the ~70 degree thigh/shank misalignment problem that took
five sessions to resolve for the two-segment case, reintroduced on
purpose. So the shank is calibrated ONCE, and that single FunctionalCal-
ibration object is reused for BOTH the knee (paired with thigh) and the
ankle (paired with toe).

--toe-id is OPTIONAL. Omit it and this behaves exactly like
mujoco_knee_stream.py (thigh+shank, knee only) -- useful for continuing
to validate the 2-sensor path while the toe sensor is being brought up
separately.

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
from rebait_calibration_adapter import FunctionalCalibration, relative_delta, _wxyz_to_rot, project_onto_axis, _average_rotation


# =============================================================================
# MODEL LOCATION  (same pattern as the original wireless_imu_streaming.py)
# =============================================================================
_MYOBODY_RELATIVE = Path("simhive") / "myo_sim" / "body"
MYOBODY_XML = "myobody.xml"
KNEE_JOINT = "knee_angle_r"
# NOT YET CONFIRMED for this model, unlike knee_angle_r (verified via
# mujoco_sanity_check.py). List real joint names before trusting this:
#   python -c "
#   import mujoco
#   m = mujoco.MjModel.from_xml_path('myobody.xml')
#   for i in range(m.njnt): print(m.joint(i).name)"
# (run from inside the model directory). Override with --ankle-joint if
# the real name differs.
ANKLE_JOINT = "ankle_angle_r"
# Confirmed present in the myosuite model's full joint list, and
# confirmed to drive the RIGHT motion via mujoco_hip_3dof_sanity_check.py
# (sequential sweep, no IMU) before any calibration was built around them.
HIP_FLEXION_JOINT = "hip_flexion_r"
HIP_ABDUCTION_JOINT = "hip_adduction_r"
HIP_ROTATION_JOINT = "hip_rotation_r"
# hip_rotation_r intentionally not driven -- see module header.


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

def check_axis_alignment(name_a, fc_a: FunctionalCalibration,
                         name_b, fc_b: FunctionalCalibration):
    """
    Compares two ALREADY-ZEROED segments' Y (gravity) and Z (joint) axes in
    world frame. MUST be called after both segments' zero is the shared,
    simultaneous one -- calling this with each segment's own individually-
    captured zero reproduces the exact bug that made this diagnostic report
    a false ~70 deg misalignment across four sessions before the fix.

    Gravity should agree closely regardless of segment/swing/anatomy ("down"
    is down for everyone). If gravity ALSO disagrees, the problem is in the
    shared zero/orientation pipeline, not this pair's joint axis specifically.
    """
    world_grav_a = fc_a.get_zero().apply(fc_a.axes["Y"])
    world_grav_b = fc_b.get_zero().apply(fc_b.axes["Y"])
    grav_dot = float(np.clip(abs(np.dot(world_grav_a, world_grav_b)), 0, 1))
    grav_deg = float(np.degrees(np.arccos(grav_dot)))
    print(f"\ngravity axis alignment ({name_a} vs {name_b}): {grav_deg:.1f} deg "
          f"(expect near 0)")

    world_axis_a = fc_a.get_zero().apply(fc_a.axes["Z"])
    world_axis_b = fc_b.get_zero().apply(fc_b.axes["Z"])
    align_dot = float(np.clip(abs(np.dot(world_axis_a, world_axis_b)), 0, 1))
    align_deg = float(np.degrees(np.arccos(align_dot)))
    print(f"joint axis alignment ({name_a} vs {name_b}): {align_deg:.1f} deg")

    if grav_deg > 15.0:
        print(f"*** Gravity axes ALSO disagree by a large amount for this")
        print(f"*** pair -- not specific to the joint axis, something in")
        print(f"*** the shared zero/orientation handling is wrong.")
    elif align_deg > 15.0:
        print(f"*** LARGE joint-axis misalignment for {name_a}/{name_b}.")
        print(f"*** Gravity agrees but the joint axis does not -- redo")
        print(f"*** Phase 2 for one or both with larger, cleaner motion.")
    elif align_deg > 5.0:
        print(f"(moderate -- worth a cleaner recalibration if the live")
        print(f" angle for this joint still looks wrong)")
    return grav_deg, align_deg


def calibrate_pelvis(state: LatestQuatState, did, gravity_s=5.0):
    """
    GRAVITY + ZERO ONLY. No Phase 2 swing (the pelvis has no joint of its
    own to swing about), no build(), no PCA axes, no sign check. Returns
    just an average gravity direction (for the alignment check below) --
    the pelvis's actual ZERO ROTATION gets set later, in the shared
    simultaneous-zero step, same as every other segment.
    """
    print(f"\n=== Calibrating: PELVIS (device {did}) ===")
    print(f"  Pelvis has no swing of its own -- gravity only.")
    input(f"  Stand/sit still for {gravity_s:.0f}s. Press Enter: ")
    accs = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < gravity_s:
        state.poll()
        acc, _ = state.gravity_gyro(did)
        if acc is not None:
            accs.append(acc)
        time.sleep(0.002)
    if not accs:
        raise RuntimeError(f"No accelerometer samples captured for pelvis "
                           f"({did}). Check it is streaming.")
    g = np.mean(accs, axis=0)
    g /= np.linalg.norm(g)
    print(f"  [pelvis] gravity direction captured from {len(accs)} samples.")
    return g   # unit vector, in the PELVIS SENSOR's own body frame


def check_pelvis_alignment(thigh_fc: FunctionalCalibration, pelvis_gravity_body,
                           pelvis_zero: Rotation):
    """
    Gravity-only alignment check for the thigh/pelvis pair -- the pelvis has
    no joint axis to compare, only gravity. Same principle as
    check_axis_alignment: "down" should agree closely regardless of segment,
    and if it doesn't, the shared zero/orientation step needs attention.
    """
    world_grav_thigh = thigh_fc.get_zero().apply(thigh_fc.axes["Y"])
    world_grav_pelvis = pelvis_zero.apply(pelvis_gravity_body)
    dot = float(np.clip(abs(np.dot(world_grav_thigh, world_grav_pelvis)), 0, 1))
    deg = float(np.degrees(np.arccos(dot)))
    print(f"\ngravity axis alignment (thigh vs pelvis): {deg:.1f} deg "
          f"(expect near 0)")
    if deg > 15.0:
        print("*** LARGE. Something in the shared zero/orientation step is")
        print("*** off for this pair -- redo the simultaneous zero, standing")
        print("*** and pausing before pressing Enter.")
    return deg


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
    fc.set_zero(_average_rotation(rots))
    print(f"  [{name}] zeroed from {len(rots)} samples.")
    return fc


def resolve_segment_sign(name, state: LatestQuatState, did, fc: FunctionalCalibration,
                         hold_s=4.0, axis=2, motion="FLEXION"):
    """
    Phase 4. One deliberate, slow, KNOWN-direction motion, held at the end.
    Do this even if sign_ambiguous read False after build() -- the validated
    failure mode (D1 donning data, and the synthetic seed-0 case found while
    testing this adapter) is a WRONG sign with no automatic warning, not a
    flagged one. `motion` names the movement so the prompt is anatomically
    correct (FLEXION for thigh/shank, DORSIFLEXION -- toes up -- for toe).
    """
    print(f"\n  [{name}] SIGN CHECK. Perform one slow, deliberate {motion}")
    print(f"  and hold it for the last {hold_s:.0f}s of the window.")
    input(f"  Press Enter, then perform the {motion.lower()} and hold: ")
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
    held = _average_rotation(rots[-max(1, len(rots) // 3):])   # last third = the held pose
    flipped = fc.resolve_sign(held, expect_positive=True, axis=axis)
    check_angle = fc.get_primary_angle(held, axis=axis)
    print(f"  held-pose angle now reads {check_angle:+.1f} deg "
          f"({'FLIPPED' if flipped else 'was already correct'})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Live knee (+ optional ankle) angle in MuJoCo from ReBAIT IMUs.")
    ap.add_argument("--thigh-id", type=int, required=True)
    ap.add_argument("--shank-id", type=int, required=True)
    ap.add_argument("--toe-id", type=int, default=None,
                    help="optional. Omit to run thigh+shank/knee-only, "
                         "exactly like mujoco_knee_stream.py.")
    ap.add_argument("--pelvis-id", type=int, default=None,
                    help="optional. Adds hip FLEXION only (not adduction/"
                         "rotation -- see module docstring). Independent "
                         "of --toe-id; use either, both, or neither.")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--ankle-joint", default=ANKLE_JOINT,
                    help="override if the model's real ankle joint name "
                         "differs from the unverified default")
    ap.add_argument("--hip-flexion-joint", default=HIP_FLEXION_JOINT,
                    help="override if the model's real hip flexion joint "
                         "name differs from the unverified default")
    ap.add_argument("--hip-abad", action="store_true",
                    help="optional, requires --pelvis-id. Adds hip ab/"
                         "adduction: a SECOND swing on the thigh sensor "
                         "(leg out to the side), independent of --toe-id.")
    ap.add_argument("--hip-abduction-joint", default=HIP_ABDUCTION_JOINT,
                    help="override if the model's real hip adduction "
                         "joint name differs from the unverified default")
    ap.add_argument("--hip-rotation", action="store_true",
                    help="optional, requires --hip-abad. Adds hip internal/"
                         "external rotation. UNLIKE flexion/ab-adduction, "
                         "this drifts over time (no magnetometer, no way to "
                         "correct rotation about the vertical axis) -- press "
                         "'z' in the viewer to re-zero to 0 while standing "
                         "at neutral. Not a stable absolute angle; treat as "
                         "short-burst relative motion only.")
    ap.add_argument("--hip-rotation-joint", default=HIP_ROTATION_JOINT,
                    help="override if the model's real hip rotation joint "
                         "name differs from the unverified default")
    ap.add_argument("--gravity-s", type=float, default=5.0)
    ap.add_argument("--rotation-s", type=float, default=10.0)
    a = ap.parse_args()

    if a.hip_abad and a.pelvis_id is None:
        raise SystemExit("--hip-abad requires --pelvis-id (ab/adduction "
                         "is a pelvis-relative angle, same as hip flexion).")
    if a.hip_rotation and not a.hip_abad:
        raise SystemExit("--hip-rotation requires --hip-abad (reuses that "
                         "phase's thigh calibration; no separate swing "
                         "needed for rotation -- see module docstring).")

    have_toe = a.toe_id is not None
    have_pelvis = a.pelvis_id is not None
    thigh_dev = f"imu{a.thigh_id:02d}"
    shank_dev = f"imu{a.shank_id:02d}"
    toe_dev = f"imu{a.toe_id:02d}" if have_toe else None
    pelvis_dev = f"imu{a.pelvis_id:02d}" if have_pelvis else None

    ids = ((a.thigh_id, a.shank_id) + ((a.toe_id,) if have_toe else ())
          + ((a.pelvis_id,) if have_pelvis else ()))
    n_sensors = len(ids)
    cb = UdpImuCallback(port=a.port, expected_ids=ids)
    cb.enable()
    cb.attach()
    state = LatestQuatState(cb)

    print(f"Waiting for {n_sensors} sensors ...")
    while not (state.ready(thigh_dev) and state.ready(shank_dev)
              and (not have_toe or state.ready(toe_dev))
              and (not have_pelvis or state.ready(pelvis_dev))):
        state.poll()
        time.sleep(0.05)
    label = f"thigh ({thigh_dev}), shank ({shank_dev})"
    if have_toe:
        label += f", toe ({toe_dev})"
    if have_pelvis:
        label += f", pelvis ({pelvis_dev})"
    print(f"  {label} all streaming.")

    thigh_fc = calibrate_segment("thigh", state, thigh_dev, a.gravity_s, a.rotation_s)
    # SHANK CALIBRATED ONCE. This one FunctionalCalibration object is reused
    # below for BOTH the knee (paired with thigh_fc) and the ankle (paired
    # with toe_fc, if present). Do not calibrate the shank a second time
    # against the toe -- that would reintroduce, on purpose, the exact
    # independent-axis-disagreement problem that took five sessions to sort
    # out for thigh/shank alone.
    shank_fc = calibrate_segment("shank", state, shank_dev, a.gravity_s, a.rotation_s)
    resolve_segment_sign("thigh", state, thigh_dev, thigh_fc)
    resolve_segment_sign("shank", state, shank_dev, shank_fc)

    # --- HIP AB/ADDUCTION: a SECOND, independent swing on the thigh -----
    # sensor already calibrated above. Deliberately gravity-only
    # orthogonalized -- see find_second_axis()'s docstring. Validated
    # against synthetic ground truth AND confirmed on real hardware with
    # two independently-designed diagnostic tools before being wired in
    # here (hip_abad_align_test.py, flex_vs_abad_raw_check.py).
    thigh_abad_axis = None
    if a.hip_abad:
        print(f"\n=== THIGH AB/ADDUCTION (device {thigh_dev}) ===")
        print("Swing your leg OUT to the side and back IN -- a frontal-")
        print("plane motion, not forward/back. Keep your knee fairly")
        print("straight and try not to rotate or lean.")
        input(f"Press Enter, then swing for {a.rotation_s:.0f}s: ")
        abad_gyro = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < a.rotation_s:
            state.poll()
            _acc, gyro = state.gravity_gyro(thigh_dev)
            if gyro is not None:
                abad_gyro.append(gyro)
            time.sleep(0.002)
        thigh_abad_axis, abad_svd = thigh_fc.find_second_axis(abad_gyro)
        print(f"  svd_ratio: {abad_svd:.2f} (want > ~3)")
        if abad_svd < 3.0:
            print(f"  *** LOW svd_ratio ({abad_svd:.2f}). This swing likely")
            print(f"  *** was not cleanly single-axis -- consider redoing it")
            print(f"  *** before trusting the ab/adduction angle this session.")
        angle_vs_X = np.degrees(np.arccos(np.clip(
            abs(np.dot(thigh_abad_axis, thigh_fc.axes["X"])), 0, 1)))
        print(f"  vs. math-derived X (informational): {angle_vs_X:.1f} deg")

        # Its OWN sign check -- this is a genuinely new axis, unlike hip
        # flexion (which reuses thigh_fc's existing flexion-context sign
        # check because both read the identical axis). Cannot borrow that
        # check here; this axis has never been verified against a known
        # direction until now.
        print("\n  AB/ADDUCTION SIGN CHECK. Perform one slow, deliberate")
        print("  ABDUCTION (leg out to the side) and hold it for the")
        print("  last 4s of the window.")
        input("  Press Enter, then abduct and hold: ")
        rots = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 4.0:
            state.poll()
            r = state.rotation(thigh_dev)
            if r is not None:
                rots.append(r)
            time.sleep(0.002)
        held = _average_rotation(rots[-max(1, len(rots)//3):])
        held_rel = thigh_fc.get_delta(held)
        check_deg = project_onto_axis(held_rel, thigh_abad_axis)
        flipped = False
        if check_deg < 0:
            thigh_abad_axis = -thigh_abad_axis
            flipped = True
            check_deg = -check_deg
        print(f"  held-pose abduction now reads {check_deg:+.1f} deg "
              f"({'FLIPPED' if flipped else 'was already correct'})")

    toe_fc = None
    if have_toe:
        toe_fc = calibrate_segment("toe", state, toe_dev, a.gravity_s, a.rotation_s)
        resolve_segment_sign("toe", state, toe_dev, toe_fc,
                             motion="DORSIFLEXION (toes up)")

    pelvis_gravity_body = None
    if have_pelvis:
        pelvis_gravity_body = calibrate_pelvis(state, pelvis_dev, a.gravity_s)
        # No resolve_segment_sign() call here -- hip flexion reuses thigh_fc's
        # Z axis, already sign-verified by the knee's OWN sign check above.
        # Confirmed in testing: that check correctly sets the hip sign too.

    # --- SIMULTANEOUS ZERO, ALL SEGMENTS TOGETHER -----------------------
    # calibrate_segment() zeros each segment separately, minutes apart. Any
    # RELATIVE angle (knee OR ankle) is only correct if all its segments'
    # zeros describe the SAME physical pose. This captures thigh, shank,
    # and (if present) toe from ONE shared moment.
    print("\n=== FINAL SIMULTANEOUS ZERO ===")
    print("Stand in your normal neutral pose -- the SAME posture you want")
    print("to read as 0 deg for every joint. All segments will be zeroed")
    print("together from this one moment.")
    input("Press Enter, then hold neutral for 5s: ")
    rots_t, rots_s, rots_toe, rots_pelvis = [], [], [], []
    t0 = time.monotonic()
    while time.monotonic() - t0 < 5.0:
        state.poll()
        r = state.rotation(thigh_dev)
        if r is not None:
            rots_t.append(r)
        r = state.rotation(shank_dev)
        if r is not None:
            rots_s.append(r)
        if have_toe:
            r = state.rotation(toe_dev)
            if r is not None:
                rots_toe.append(r)
        if have_pelvis:
            r = state.rotation(pelvis_dev)
            if r is not None:
                rots_pelvis.append(r)
        time.sleep(0.002)
    thigh_fc.set_zero(_average_rotation(rots_t))
    shank_fc.set_zero(_average_rotation(rots_s))
    if have_toe:
        toe_fc.set_zero(_average_rotation(rots_toe))
    # Pelvis has no FunctionalCalibration object (no axes of its own), so its
    # zero is a bare Rotation, not stored via .set_zero() on anything.
    pelvis_zero = _average_rotation(rots_pelvis) if have_pelvis else None

    # Verify immediately: report the knee (and ankle) angle AT the pose
    # that was just called zero, rather than discovering an offset later
    # by watching the viewer.
    check_rel = relative_delta(thigh_fc.get_zero(), rots_t[-1], shank_fc.get_zero(), rots_s[-1])
    check_deg = shank_fc.project_angle(check_rel, axis=2)
    print(f"knee angle at the pose just zeroed: {check_deg:+.1f} deg "
          f"(should read ~0)")
    if abs(check_deg) > 5.0:
        print("*** still nonzero at neutral -- something else needs fixing")
        print("*** before trusting the live angle.")

    if have_toe:
        check_rel_a = relative_delta(shank_fc.get_zero(), rots_s[-1], toe_fc.get_zero(), rots_toe[-1])
        check_deg_a = toe_fc.project_angle(check_rel_a, axis=2)
        print(f"ankle angle at the pose just zeroed: {check_deg_a:+.1f} deg "
              f"(should read ~0)")
        if abs(check_deg_a) > 5.0:
            print("*** still nonzero at neutral for the ankle -- same caution")
            print("*** as above applies.")

    # --- AXIS ALIGNMENT CHECKS, EACH PAIR THAT SHARES A JOINT -----------
    # Must run AFTER the simultaneous zero above -- see check_axis_alignment's
    # docstring for why (this was itself a real, found-and-fixed bug).
    check_axis_alignment("thigh", thigh_fc, "shank", shank_fc)
    if have_toe:
        check_axis_alignment("shank", shank_fc, "toe", toe_fc)
    if have_pelvis:
        check_pelvis_alignment(thigh_fc, pelvis_gravity_body, pelvis_zero)
        check_rel_h = relative_delta(pelvis_zero, rots_pelvis[-1], thigh_fc.get_zero(), rots_t[-1])
        check_deg_h = thigh_fc.project_angle(check_rel_h, axis=2)
        print(f"hip flexion at the pose just zeroed: {check_deg_h:+.1f} deg "
              f"(should read ~0)")
        if abs(check_deg_h) > 5.0:
            print("*** still nonzero at neutral for the hip -- same caution")
            print("*** as above applies.")
        if a.hip_abad and thigh_abad_axis is not None:
            check_deg_ha = project_onto_axis(check_rel_h, thigh_abad_axis)
            print(f"hip ab/adduction at the pose just zeroed: {check_deg_ha:+.1f} deg "
                  f"(should read ~0)")
            if abs(check_deg_ha) > 5.0:
                print("*** still nonzero at neutral for hip ab/adduction -- same")
                print("*** caution as above applies.")

    print("\nCalibration complete. Launching viewer ...\n")

# --- MuJoCo import deferred to here: everything above is fully testable
    # without MuJoCo installed. ---
    import mujoco
    import mujoco.viewer
    try:
        import msvcrt   # Windows-only stdlib; non-blocking keypress check
        have_msvcrt = True
    except ImportError:
        have_msvcrt = False
        if a.hip_rotation:
            print("  [WARN] msvcrt not available (non-Windows) -- the 'z' "
                  "re-zero key won't work. Hip rotation will only show the "
                  "zero-pose baseline, with no way to reset drift live.")

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

    ankle_addr = None
    if have_toe:
        for i in range(model.njnt):
            if model.joint(i).name == a.ankle_joint:
                ankle_addr = model.jnt_qposadr[i]
        if ankle_addr is None:
            print(f"  [WARN] joint '{a.ankle_joint}' not found in the model.")
            print(f"  [WARN] --toe-id was given but ankle will NOT be driven.")
            print(f"  [WARN] List real joint names and pass --ankle-joint if")
            print(f"  [WARN] the name differs (see ANKLE_JOINT comment above).")

    hip_addr = None
    if have_pelvis:
        for i in range(model.njnt):
            if model.joint(i).name == a.hip_flexion_joint:
                hip_addr = model.jnt_qposadr[i]
        if hip_addr is None:
            print(f"  [WARN] joint '{a.hip_flexion_joint}' not found in the model.")
            print(f"  [WARN] --pelvis-id was given but hip will NOT be driven.")
            print(f"  [WARN] Pass --hip-flexion-joint if the name differs.")

    hip_abad_addr = None
    if a.hip_abad:
        for i in range(model.njnt):
            if model.joint(i).name == a.hip_abduction_joint:
                hip_abad_addr = model.jnt_qposadr[i]
        if hip_abad_addr is None:
            print(f"  [WARN] joint '{a.hip_abduction_joint}' not found in the model.")
            print(f"  [WARN] --hip-abad was given but ab/adduction will NOT be driven.")
            print(f"  [WARN] Pass --hip-abduction-joint if the name differs.")

    hip_rot_addr = None
    if a.hip_rotation:
        for i in range(model.njnt):
            if model.joint(i).name == a.hip_rotation_joint:
                hip_rot_addr = model.jnt_qposadr[i]
        if hip_rot_addr is None:
            print(f"  [WARN] joint '{a.hip_rotation_joint}' not found in the model.")
            print(f"  [WARN] --hip-rotation was given but rotation will NOT be driven.")
            print(f"  [WARN] Pass --hip-rotation-joint if the name differs.")

    mujoco.mj_forward(model, data)

    frame = 0
    drop_warn_shown = False
    hip_rot_offset_deg = 0.0
    hip_rot_zero_time = time.monotonic()
    hip_rot_rezero_pending = False   # set by keypress, consumed once the
                                      # frame's rotation is actually computed
    if a.hip_rotation:
        print("Hip rotation: press 'z' at any time while standing at")
        print("neutral to re-zero it. It WILL drift between re-zeros --")
        print("trust it less the longer since the last 'z'.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 150

        while viewer.is_running():
            counts = state.poll()   # drains EVERYTHING waiting, every frame

            if a.hip_rotation and have_msvcrt and msvcrt.kbhit():
                if msvcrt.getch() in (b"z", b"Z"):
                    hip_rot_rezero_pending = True

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

            ankle_deg = 0.0
            if have_toe:
                rot_toe = state.rotation(toe_dev)
                # NOTE: reuses the SAME shank_fc built above for the knee --
                # this is the shared-shank principle the whole file exists
                # to enforce, not a second independent shank calibration.
                if rot_s is not None and rot_toe is not None:
                    ankle_rel = relative_delta(shank_fc.get_zero(), rot_s,
                                               toe_fc.get_zero(), rot_toe)
                    ankle_deg = toe_fc.project_angle(ankle_rel, axis=2)
                    if ankle_addr is not None:
                        data.qpos[ankle_addr] = np.radians(ankle_deg)

            hip_deg = 0.0
            hip_abad_deg = 0.0
            if have_pelvis:
                rot_pelvis = state.rotation(pelvis_dev)
                # NOTE: reuses the SAME thigh_fc built above for the knee --
                # the pelvis contributes only its zero rotation, nothing
                # else. thigh_fc's axes were never rebuilt for hip.
                if rot_pelvis is not None and rot_t is not None:
                    hip_rel = relative_delta(pelvis_zero, rot_pelvis,
                                             thigh_fc.get_zero(), rot_t)
                    hip_deg = thigh_fc.project_angle(hip_rel, axis=2)
                    if hip_addr is not None:
                        data.qpos[hip_addr] = np.radians(hip_deg)

                    # AB/ADDUCTION: the SAME hip_rel just computed above,
                    # projected onto the SECOND axis instead of Z -- one
                    # relative rotation, two independent projections. Uses
                    # project_onto_axis (NOT thigh_fc.project_second_axis,
                    # which assumes an un-delta'd rotation and would double-
                    # apply the zero correction on an already-relative
                    # rotation like hip_rel).
                    if a.hip_abad and thigh_abad_axis is not None:
                        hip_abad_deg = project_onto_axis(hip_rel, thigh_abad_axis)
                        if hip_abad_addr is not None:
                            # SIGN FLIP HERE ONLY -- do not touch hip_abad_deg
                            # itself. The calibration's own sign check confirmed
                            # +hip_abad_deg = abduction, correctly and self-
                            # consistently (verified: a held abduction reads
                            # positive). But the MuJoCo joint is literally named
                            # hip_adduction_r, and standard OpenSim/gait2392
                            # convention defines POSITIVE on that joint as
                            # ADDUCTION -- the opposite of this calibration's
                            # convention. Confirmed on real hardware: the printed
                            # angle was correct (positive during a real abduction)
                            # while the MuJoCo model visibly moved the opposite
                            # way. The printed hip_abad_deg is left unchanged --
                            # only the value handed to MuJoCo is negated, so the
                            # number on screen keeps meaning what the sign check
                            # established, independent of which convention any
                            # particular skeletal model happens to use.
                            data.qpos[hip_abad_addr] = -np.radians(hip_abad_deg)

                    # HIP ROTATION: the SAME hip_rel again, projected onto
                    # thigh_fc.axes["Y"] (gravity/vertical) -- NOT a newly
                    # calibrated axis, since gravity is already established
                    # from Phase 1 and IS (approximately) the rotation axis
                    # for this DOF. See module docstring for why this DOF
                    # is structurally different: no gravity-orthogonalization
                    # trick is possible when the axis being sought basically
                    # IS the gravity direction. Validated synthetically:
                    # this projection is exactly equivalent to raw gyro
                    # integration about the vertical axis (0.000 deg
                    # difference) -- meaning it drifts the same way raw
                    # integration would, with nothing to correct it. DO NOT
                    # trust this as a stable absolute angle; it is a
                    # relative, drifting measurement, reset only by 'z'.
                    #
                    # SIGN: UNVERIFIED. Written as-is, no flip applied --
                    # unlike hip_abad above, there is no real-hardware
                    # evidence yet for which direction is positive on
                    # hip_rotation_r. Do not assume it matches the
                    # ab/adduction pattern; verify with a real, known
                    # rotation the same way ab/adduction's sign was caught
                    # and fixed.
                    if a.hip_rotation:
                        hip_rot_raw_deg = project_onto_axis(hip_rel, thigh_fc.axes["Y"])
                        if hip_rot_rezero_pending:
                            hip_rot_offset_deg = hip_rot_raw_deg
                            hip_rot_zero_time = time.monotonic()
                            hip_rot_rezero_pending = False
                            print(f"\n  [hip rotation re-zeroed]" + " " * 40)
                        hip_rot_deg = hip_rot_raw_deg - hip_rot_offset_deg
                        if hip_rot_addr is not None:
                            data.qpos[hip_rot_addr] = np.radians(hip_rot_deg)

            frame += 1
            if frame % 60 == 0:
                line = f"  frame {frame}   knee = {knee_deg:+6.1f} deg"
                if have_toe:
                    line += f"   ankle = {ankle_deg:+6.1f} deg"
                if have_pelvis:
                    line += f"   hip = {hip_deg:+6.1f} deg"
                    if a.hip_abad:
                        line += f"   hip_abad = {hip_abad_deg:+6.1f} deg"
                    if a.hip_rotation:
                        since_zero = time.monotonic() - hip_rot_zero_time
                        # DISPLAY-ONLY negation. Confirmed on real hardware:
                        # the visual MuJoCo motion (driven by the UNFLIPPED
                        # hip_rot_deg below, in the qpos write) already
                        # matches real physical rotation correctly -- so
                        # qpos must NOT change. It was the PRINTED number's
                        # sign that was backwards. Flipping hip_rot_deg
                        # itself would fix the number but break the visual
                        # match that's already confirmed correct -- so the
                        # negation lives ONLY here, in the display string,
                        # mirroring hip_abad's fix but at the opposite end
                        # of the pipeline (that one needed the flip at the
                        # qpos write instead, because ITS printed number was
                        # already correct and the visual was backwards).
                        line += (f"   hip_rot = {-hip_rot_deg:+6.1f} deg "
                                f"({since_zero:4.0f}s since 'z')")
                print(line, end="\r")

            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.002)

    cb.detach()
    cb.close()


if __name__ == "__main__":
    main()