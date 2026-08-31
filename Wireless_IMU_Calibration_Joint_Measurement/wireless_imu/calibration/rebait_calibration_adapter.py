"""
rebait_calibration_adapter.py  -  ReBAIT calibration theory, Shah's interface.

WHY THIS FILE EXISTS
---------------------
wireless_imu_streaming.py calls a FunctionalCalibration class (plus
_average_rotation and relative_delta) from calibration.py. I have never seen
calibration.py's actual internals -- only how the streaming script calls it.
So "combining" ReBAIT's theory into the MuJoCo pipeline cannot mean editing
code I haven't seen; it means providing something that satisfies the SAME
public interface, built on math that IS fully known and already validated:
shank_calibration.py's gravity + gyro-PCA calibration (Hoegberg, Donahue &
Major 2025 / ReBAIT), the exact code this whole project has been testing.

This is a bridge, not a copy of Shah's file. Import it as a drop-in
replacement:

    # in wireless_imu_streaming.py, change ONLY this line:
    from rebait_calibration_adapter import (FunctionalCalibration,
                                             _average_rotation, relative_delta)

Nothing else in wireless_imu_streaming.py needs to change -- every method
call it already makes (add_gravity_frame, add_rotation_frame, build, axes,
svd_ratio, gravity_spread, reset_calibration, set_zero, get_zero, get_delta,
get_angles, get_primary_angle, project_angle) is implemented here.

WHAT IS PORTED FROM shank_calibration.py, EXACTLY, NOT REIMPLEMENTED
----------------------------------------------------------------------
  - gravity axis:     mean direction of stationary accel samples
  - joint axis:        first principal component of gyro samples during the
                        flex/extend phase (same PCA shank_calibration.py uses)
  - sign check:        ReBAIT's peak-positive-vs-negative heuristic, WITH THE
                        SAME AMBIGUITY WARNING -- this project spent a long
                        session establishing that a symmetric seated swing
                        makes this heuristic a coin flip (D1 donning data:
                        identical "as found" flags produced OPPOSITE angle
                        conventions across two of five re-donnings). That
                        finding is carried over here rather than silently
                        dropped. The sign this module picks is a DEFAULT,
                        not a verified truth -- resolve_joint_sign-style
                        verification against a known movement is still your
                        job before trusting the sign, exactly as before.

TWO THINGS I HAD TO INFER RATHER THAN CONFIRM, BECAUSE I HAVE NOT SEEN
calibration.py's SOURCE -- VERIFY THESE AGAINST HIS REAL NUMBERS IF YOU
STILL HAVE A CALIBRATION LOG FROM HIS VERSION
-------------------------------------------------------------------------
  1. gravity_spread's exact formula/units. His docstring: "mean angular
     scatter of Phase 1 accel samples from their mean direction... want <
     ~0.05". No units stated. I have implemented it as the mean angle (in
     RADIANS) between each sample and the mean direction, which is the most
     natural reading of "angular scatter" and lands in the same ~0.0x order
     of magnitude as his threshold -- but I cannot rule out he meant degrees,
     or 1-cos(angle), or something else. If a saved calib_axes_*.json from a
     run under his original calibration.py exists, compare its
     gravity_spread values against this implementation's on the SAME
     recording to confirm the scale matches before trusting the 0.05
     threshold.

  2. relative_delta()'s rotation composition order for the hip. Reconstructed
     from the call site (`relative_delta(pelvis_zero, rot_p, thigh_fc.get_zero(),
     rot_t)` then `thigh_fc.project_angle(hip_rel, axis=...)`) as: each
     segment's own zeroed delta, then the thigh's delta expressed relative to
     the pelvis's delta. This is the physically sensible definition of one
     segment's rotation relative to another, but it is inferred, not copied
     from his code.

Everything else here (axis construction, angle projection, get_delta) follows
directly and unambiguously from how the streaming script uses the returned
values, so there is much less room for a silent mismatch there.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from shank_calibration import qmul, qconj, qnorm, quat_mean


# --------------------------------------------------------------------------
# quaternion <-> scipy Rotation conversion
#
# shank_calibration.py's quaternion arrays are [w, x, y, z].
# scipy.spatial.transform.Rotation.as_quat() / from_quat() use [x, y, z, w].
# Getting this order wrong silently reintroduces exactly the kind of frame
# bug the REMAP_TO_FILTER_FRAME saga was about. Both directions are wrapped
# here so nothing downstream has to remember the order.
# --------------------------------------------------------------------------

def _rot_to_wxyz(rot: Rotation) -> np.ndarray:
    x, y, z, w = rot.as_quat()
    return np.array([w, x, y, z])


def _wxyz_to_rot(q: np.ndarray) -> Rotation:
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w])


def _rotvec_deg(rot: Rotation) -> np.ndarray:
    return np.degrees(rot.as_rotvec())


# --------------------------------------------------------------------------
# module-level helpers matching calibration.py's exports
# --------------------------------------------------------------------------

def _average_rotation(frames) -> Rotation:
    """Mean of a list of scipy Rotations (quaternion averaging, sign-safe)."""
    if not frames:
        return Rotation.identity()
    q = np.array([_rot_to_wxyz(r) for r in frames])
    return _wxyz_to_rot(quat_mean(q))


def relative_delta(zero1: Rotation, rot1: Rotation,
                    zero2: Rotation, rot2: Rotation) -> Rotation:
    """
    Rotation of segment 2 relative to segment 1, each already referenced to
    its own zero pose. See the module docstring, point 2, for why this
    composition order is inferred rather than confirmed.
    """
    delta1 = zero1.inv() * rot1
    delta2 = zero2.inv() * rot2
    return delta1.inv() * delta2


def project_onto_axis(rel_rotation: Rotation, axis_vec) -> float:
    """
    Project an ALREADY-COMPUTED relative rotation (e.g. from
    relative_delta(), such as the hip's pelvis-vs-thigh rotation) onto an
    arbitrary axis vector -- for the hip's ab/adduction angle, where the
    axis comes from FunctionalCalibration.find_second_axis() rather than
    one of the fixed self.axes["X"/"Y"/"Z"] slots that project_angle()
    reads from directly.

    DELIBERATELY separate from FunctionalCalibration.project_second_axis(),
    which is for a DIFFERENT case: a segment's OWN rotation relative to ITS
    OWN zero (calls self.get_delta() internally). rel_rotation here is
    already zero-adjusted by relative_delta() -- calling get_delta() on it
    again would double-apply the zero correction and silently produce a
    wrong angle. This function does the projection only, matching exactly
    how project_angle() itself handles an already-relative rotation.
    """
    rv_deg = np.degrees(rel_rotation.as_rotvec())
    return float(np.dot(rv_deg, axis_vec))


# --------------------------------------------------------------------------
# FunctionalCalibration
# --------------------------------------------------------------------------

class FunctionalCalibration:
    """
    Drop-in replacement for calibration.py's FunctionalCalibration, built on
    ReBAIT's gravity + gyro-PCA calibration instead of an independent
    implementation. See the module docstring for exactly what is ported vs.
    inferred.
    """

    def __init__(self):
        self.reset_calibration()
        self._zero: Rotation = Rotation.identity()

    def reset_calibration(self) -> None:
        self._grav_samples = []      # list of (ax, ay, az), Phase 1
        self._gyro_samples = []      # list of (gx, gy, gz), Phase 2
        self.axes = None             # filled by build(): {'X','Y','Z' -> unit vec}
        self.svd_ratio = float("nan")
        self.gravity_spread = float("nan")
        self.sign_margin = float("nan")
        self.sign_ambiguous = True
        self._built = False

    # ---- data collection, mirrors add_gravity_frame / add_rotation_frame --

    def add_gravity_frame(self, ax: float, ay: float, az: float) -> None:
        self._grav_samples.append((ax, ay, az))

    def add_rotation_frame(self, gx: float, gy: float, gz: float) -> None:
        self._gyro_samples.append((gx, gy, gz))

    # ---- build(): gravity axis + gyro-PCA axis, ReBAIT-style ---------------

    def build(self) -> None:
        if len(self._grav_samples) < 10:
            raise ValueError("Too few gravity-phase samples. Re-run Phase 1.")
        if len(self._gyro_samples) < 10:
            raise ValueError("Too few rotation-phase samples. Re-run Phase 2.")

        g = np.asarray(self._grav_samples, float)
        w = np.asarray(self._gyro_samples, float)

        # --- gravity axis: mean direction, and its scatter -------------
        g_dirs = g / np.linalg.norm(g, axis=1, keepdims=True)
        g_mean = g_dirs.mean(axis=0)
        g_mean /= np.linalg.norm(g_mean)
        cosang = np.clip(g_dirs @ g_mean, -1.0, 1.0)
        # See module docstring point 1: radians, best-effort match to his
        # "want < ~0.05" threshold. Verify against his real numbers if you can.
        self.gravity_spread = float(np.mean(np.arccos(cosang)))

        # --- joint axis: first principal component of gyro during motion,
        #     identical in spirit to shank_calibration.py's PCA step ------
        wc = w - w.mean(axis=0)
        U, S, Vt = np.linalg.svd(wc, full_matrices=False)
        joint_axis = Vt[0]
        self.svd_ratio = float(S[0] / S[1]) if len(S) > 1 and S[1] > 1e-9 else float("inf")

        # --- ReBAIT's sign heuristic, WITH the ambiguity it actually has ---
        proj = wc @ joint_axis
        peak_pos = float(np.percentile(proj, 95))
        peak_neg = float(np.percentile(proj, 5))
        denom = abs(peak_pos) + abs(peak_neg)
        self.sign_margin = ((abs(peak_pos) - abs(peak_neg)) / denom
                            if denom > 1e-9 else 0.0)
        self.sign_ambiguous = abs(self.sign_margin) < 0.12   # same threshold as shank_calibration.py
        if peak_neg + peak_pos < 0:
            joint_axis = -joint_axis

        # --- orthonormal frame: gravity fixes Y, joint axis fixes Z,
        #     Gram-Schmidt so they are exactly perpendicular, X completes it
        y_axis = g_mean
        z_axis = joint_axis - np.dot(joint_axis, y_axis) * y_axis
        n = np.linalg.norm(z_axis)
        if n < 1e-6:
            raise ValueError(
                "Joint axis is parallel to gravity -- the swing did not "
                "isolate a mediolateral rotation. Re-run Phase 2.")
        z_axis /= n
        x_axis = np.cross(y_axis, z_axis)

        self.axes = {"X": x_axis, "Y": y_axis, "Z": z_axis}
        self._built = True

    # ---- zeroing -----------------------------------------------------------

    def set_zero(self, zero_rot: Rotation) -> None:
        self._zero = zero_rot

    def get_zero(self) -> Rotation:
        return self._zero

    def get_delta(self, rot: Rotation) -> Rotation:
        return self._zero.inv() * rot

    # ---- second axis (e.g. hip ab/adduction) --------------------------------

    def find_second_axis(self, gyro_samples) -> tuple:
        """
        Find a SECOND, independent rotation axis from a separate swing --
        e.g. hip ab/adduction, alongside the primary flexion axis already
        in self.axes["Z"]. Requires build() to have already run (uses the
        gravity axis it found).

        DELIBERATELY orthogonalized against gravity ONLY, NOT against the
        existing Z (flexion) axis. An earlier version of this forced
        orthogonality against both -- but gravity's perpendicular plane is
        2-D and Z already lives inside it, so forcing orthogonality against
        Z too leaves only one possible direction, which is mathematically
        IDENTICAL to axes["X"] (=Y cross Z) regardless of what the actual
        swing data contains. That collapses "independently measured" into
        "mathematically derived" and defeats the point of measuring it at
        all. Validated against synthetic ground truth: when the true
        second axis is NOT perfectly orthogonal to flexion (the realistic
        case -- real hip anatomy isn't perfectly orthogonal, matching the
        ankle/subtalar lesson), this gravity-only version recovers the
        true axis exactly, while axes["X"] is measurably off.

        Returns (axis_unit_vector, svd_ratio). Does NOT store anything on
        self or mutate self.axes -- the caller decides what to do with it,
        keeping this fully additive and non-disruptive to the existing
        single-axis (Z) pathway everything else depends on.
        """
        if not self._built:
            raise RuntimeError("build() has not been called yet.")
        w = np.asarray(gyro_samples, float)
        if len(w) < 10:
            raise ValueError("Too few samples for a second axis.")
        wc = w - w.mean(axis=0)
        U, S, Vt = np.linalg.svd(wc, full_matrices=False)
        raw_axis = Vt[0]
        svd_ratio = float(S[0] / S[1]) if len(S) > 1 and S[1] > 1e-9 else float("inf")

        proj = wc @ raw_axis
        peak_pos, peak_neg = np.percentile(proj, 95), np.percentile(proj, 5)
        if peak_neg + peak_pos < 0:
            raw_axis = -raw_axis

        y_axis = self.axes["Y"]
        ortho = raw_axis - np.dot(raw_axis, y_axis) * y_axis
        n = np.linalg.norm(ortho)
        if n < 1e-6:
            raise ValueError(
                "Second axis collapsed onto gravity -- the swing did not "
                "isolate a horizontal rotation. Re-run it.")
        return ortho / n, svd_ratio

    def project_second_axis(self, rot: Rotation, axis_vec) -> float:
        """Same projection formula as project_angle/get_primary_angle, for
        a second axis found via find_second_axis() rather than self.axes["Z"]."""
        if not self._built:
            raise RuntimeError("build() has not been called yet.")
        rv_deg = np.degrees(self.get_delta(rot).as_rotvec())
        return float(np.dot(rv_deg, axis_vec))

    # ---- angle extraction ---------------------------------------------------

    def get_angles(self, rot: Rotation) -> tuple:
        """
        Decompose the delta rotation's rotation-vector onto all three
        anatomical axes. Diagnostic, not the primary joint angle -- matches
        how wireless_imu_streaming.py's debug printout uses it (X/Y/Z, deg).
        """
        if not self._built:
            raise RuntimeError("build() has not been called yet.")
        rv = self.get_delta(rot).as_rotvec()
        rv_deg = np.degrees(rv)
        return (float(np.dot(rv_deg, self.axes["X"])),
                float(np.dot(rv_deg, self.axes["Y"])),
                float(np.dot(rv_deg, self.axes["Z"])))

    def get_primary_angle(self, rot: Rotation, axis: int = None) -> float:
        """
        Signed rotation angle about the joint axis (Z by default), matching
        ReBAIT's sagittal-angle extraction. `axis` follows Shah's index
        convention: 0=X, 1=Y, 2=Z (default). Called with axis=2/0/1 for the
        thigh's flex/adduct/rotate decomposition.
        """
        if not self._built:
            raise RuntimeError("build() has not been called yet.")
        key = {0: "X", 1: "Y", 2: "Z", None: "Z"}[axis]
        rv_deg = np.degrees(self.get_delta(rot).as_rotvec())
        return float(np.dot(rv_deg, self.axes[key]))

    def resolve_sign(self, rot_known_direction: Rotation, expect_positive: bool = True,
                     axis: int = None) -> bool:
        """
        Verify (and if needed, fix) the sign chosen by build()'s peak-
        asymmetry heuristic, against ONE deliberate movement whose direction
        you actually know -- e.g. "this is a slow, deliberate flexion, and
        flexion should read positive."

        This exists because sign_ambiguous only tells you the heuristic
        COULD have been wrong; it does not fix anything. For a seated,
        near-symmetric swing (the calibration motion recommended for this
        sensor placement) sign_margin regularly comes out near zero, and the
        automatic choice is then decided by noise -- confirmed on real
        donning data earlier in this project: the SAME "as found" flag
        produced OPPOSITE angle conventions on two separate re-donnings.

        Call this once after build(), using a recording of one unambiguous,
        deliberate motion (not the calibration swing itself).

        Returns True if the axis was flipped, False if it was already correct.
        """
        if not self._built:
            raise RuntimeError("build() has not been called yet.")
        angle = self.get_primary_angle(rot_known_direction, axis=axis)
        flipped = False
        if (angle > 0) != expect_positive:
            key = {0: "X", 1: "Y", 2: "Z", None: "Z"}[axis]
            self.axes[key] = -self.axes[key]
            flipped = True
        return flipped

    def project_angle(self, rel_rotation: Rotation, axis: int) -> float:
        """
        Same projection as get_primary_angle, but on an already-computed
        relative rotation (used for the hip, via relative_delta()) rather
        than one this instance's own zero. axis: 0=X, 1=Y, 2=Z.
        """
        if not self._built:
            raise RuntimeError("build() has not been called yet.")
        key = {0: "X", 1: "Y", 2: "Z"}[axis]
        rv_deg = np.degrees(rel_rotation.as_rotvec())
        return float(np.dot(rv_deg, self.axes[key]))