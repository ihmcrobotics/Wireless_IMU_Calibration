#!/usr/bin/env python3
"""
mujoco_hip_3dof_sanity_check.py  -  confirm hip_adduction_r and
hip_rotation_r actually drive the model correctly, with NO IMU code
involved. Same idea as mujoco_sanity_check.py (knee) and
mujoco_hip_sanity_check.py (hip flexion), extended to the two additional
hip DOFs needed for a full 3-DOF hip.

Both joint names were already confirmed PRESENT in this model's joint
list (pulled earlier: hip_flexion_r, hip_adduction_r, hip_rotation_r all
exist). What is NOT yet confirmed is that they each drive the RIGHT
motion -- this is that confirmation, done before any IMU-based
calibration work is built around them.

SEQUENTIAL, NOT SIMULTANEOUS. Each joint is swept alone, for its own
labeled phase, so the motion you see is unambiguously attributable to
one joint. Sweeping all three at once would produce combined motion
that's hard to visually check against any one joint name.

WHAT TO WATCH FOR, PER PHASE
  ABDUCTION/ADDUCTION : leg swings OUT to the side and back IN --
                        a frontal-plane motion, NOT forward/back.
  ROTATION             : leg/foot TWISTS about its own long axis --
                        toes turning in and out, NOT swinging in any
                        direction. This is the subtlest of the three
                        to see clearly; watch the FOOT orientation,
                        not just general leg position.

If a phase produces no visible motion, or the WRONG kind of motion (e.g.
"rotation" phase makes the leg swing sideways instead of twisting), the
joint name or sign convention needs attention before building calibration
around it -- exactly the kind of mismatch this script exists to catch
early, cheaply, with no sensors involved.

Amplitude is kept modest (+/-20 deg) for these two, smaller than the
knee/hip-flexion sweep (+/-30 deg) -- adduction and rotation have smaller
real physiological ranges, and this viewer does not enforce joint limits
(no contacts/actuation, pure kinematic display), so an oversized sweep
would just show an unrealistic, hard-to-interpret pose rather than
clamping sensibly.
"""
import argparse
import importlib.util
import os
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

_MYOBODY_RELATIVE = Path("simhive") / "myo_sim" / "body"
MYOBODY_XML = "myobody.xml"
ADDUCTION_JOINT = "hip_adduction_r"
ROTATION_JOINT = "hip_rotation_r"
FLEXION_JOINT = "hip_flexion_r"   # included for reference/comparison only


def _find_myobody_dir():
    spec = importlib.util.find_spec("myosuite")
    if spec is None or spec.origin is None:
        return None
    candidate = Path(spec.origin).parent / _MYOBODY_RELATIVE
    return candidate if (candidate / MYOBODY_XML).exists() else None


def _build_xml():
    return "\n".join([
        '<mujoco model="hip_3dof_sanity_check">',
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


def _find_joint(model, name):
    for i in range(model.njnt):
        if model.joint(i).name == name:
            return model.jnt_qposadr[i]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--adduction-joint", default=ADDUCTION_JOINT)
    ap.add_argument("--rotation-joint", default=ROTATION_JOINT)
    ap.add_argument("--phase-seconds", type=float, default=8.0)
    ap.add_argument("--amplitude-deg", type=float, default=20.0)
    a = ap.parse_args()

    print(f"mujoco version: {mujoco.__version__}")

    if a.model_dir is not None:
        model_dir = Path(a.model_dir)
    else:
        model_dir = _find_myobody_dir()
        if model_dir is None:
            raise SystemExit(
                "Could not auto-detect myobody.xml under the myosuite "
                "package. Pass --model-dir explicitly."
            )
    print(f"model dir: {model_dir}")

    os.chdir(model_dir)
    model = mujoco.MjModel.from_xml_string(_build_xml())
    data = mujoco.MjData(model)
    print(f"loaded OK: bodies={model.nbody} joints={model.njnt} nq={model.nq}")

    add_addr = _find_joint(model, a.adduction_joint)
    rot_addr = _find_joint(model, a.rotation_joint)
    flex_addr = _find_joint(model, FLEXION_JOINT)

    for label, addr, name in [("adduction", add_addr, a.adduction_joint),
                              ("rotation", rot_addr, a.rotation_joint)]:
        if addr is None:
            print(f"[WARN] joint '{name}' not found -- {label} phase will "
                  f"be skipped. List real joint names with model.joint(i).name "
                  f"for i in range(model.njnt) if this is unexpected.")
        else:
            print(f"{label} joint '{name}' found at qpos[{addr}]")

    mujoco.mj_forward(model, data)

    phases = []
    if add_addr is not None:
        phases.append(("ABDUCTION / ADDUCTION", add_addr,
                       "leg should swing OUT to the side and back IN "
                       "-- a frontal-plane motion, not forward/back."))
    if rot_addr is not None:
        phases.append(("INTERNAL / EXTERNAL ROTATION", rot_addr,
                       "leg/foot should TWIST about its own long axis "
                       "-- watch the FOOT turning in and out, not the "
                       "leg swinging anywhere."))

    if not phases:
        raise SystemExit("Neither joint was found -- nothing to test. "
                         "Check joint names first.")

    print(f"\nLaunching viewer. {len(phases)} phase(s), "
          f"{a.phase_seconds:.0f}s each, +/-{a.amplitude_deg:.0f} deg.")
    print("Close the viewer window, or wait, to end early.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 150

        for label, addr, watch_for in phases:
            if not viewer.is_running():
                break
            print(f"--- {label} ---")
            print(f"    watch for: {watch_for}")
            t0 = time.time()
            while viewer.is_running() and time.time() - t0 < a.phase_seconds:
                data.qpos[addr] = np.radians(
                    a.amplitude_deg * np.sin(2 * (time.time() - t0)))
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.01)
            data.qpos[addr] = 0.0   # reset to neutral before the next phase
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.5)

    print("Sanity check complete. For each phase, confirm the motion you")
    print("saw matched the 'watch for' description printed above it. If")
    print("either phase showed no motion, or the WRONG kind of motion,")
    print("report which one -- the joint name or sign needs attention")
    print("before any calibration work is built around it.")


if __name__ == "__main__":
    main()