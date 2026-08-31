#!/usr/bin/env python3
"""
mujoco_hip_sanity_check.py  -  confirm hip_flexion_r actually drives the
model correctly, with NO IMU code involved. Same idea as
mujoco_sanity_check.py did for knee_angle_r, applied to the hip.

Isolates one question: does the THIGH visibly swing from the hip when this
joint is driven? If it doesn't move, or if the wrong thing moves (e.g. only
the lower leg), the joint name is wrong for this model and
mujoco_full_leg_stream.py needs --hip-flexion-joint pointed at the correct
one -- confirm this BEFORE trusting any live hip angle from real sensors.
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
HIP_JOINT = "hip_flexion_r"


def _find_myobody_dir():
    spec = importlib.util.find_spec("myosuite")
    if spec is None or spec.origin is None:
        return None
    candidate = Path(spec.origin).parent / _MYOBODY_RELATIVE
    return candidate if (candidate / MYOBODY_XML).exists() else None


def _build_xml():
    return "\n".join([
        '<mujoco model="sanity_check">',
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--hip-joint", default=HIP_JOINT,
                    help="override if the real joint name differs")
    a = ap.parse_args()

    print(f"mujoco version: {mujoco.__version__}")

    if a.model_dir is not None:
        model_dir = Path(a.model_dir)
    else:
        model_dir = _find_myobody_dir()
        if model_dir is None:
            raise SystemExit(
                "Could not auto-detect myobody.xml under the myosuite "
                "package. Pass --model-dir explicitly, e.g.:\n"
                "  python mujoco_sanity_check.py --model-dir "
                "C:\\path\\to\\myosuite\\simhive\\myo_sim\\body"
            )
    print(f"model dir: {model_dir}")
    if not (model_dir / MYOBODY_XML).exists():
        raise SystemExit(f"{MYOBODY_XML} not found in {model_dir}")

    os.chdir(model_dir)
    model = mujoco.MjModel.from_xml_string(_build_xml())
    data = mujoco.MjData(model)
    print(f"loaded OK: bodies={model.nbody} joints={model.njnt} nq={model.nq}")

    hip_addr = None
    for i in range(model.njnt):
        if model.joint(i).name == a.hip_joint:
            hip_addr = model.jnt_qposadr[i]
    if hip_addr is None:
        print(f"[WARN] joint '{a.hip_joint}' not found -- check the joint name "
              f"in this myosuite version (model.joint(i).name for "
              f"i in range(model.njnt) to list them all -- you already have "
              f"the full list from the earlier check).")
        raise SystemExit(1)
    else:
        print(f"hip joint '{a.hip_joint}' found at qpos[{hip_addr}]")

    mujoco.mj_forward(model, data)

    print("\nLaunching viewer. The WHOLE LEG should swing back and forth")
    print("from the hip for 10 seconds -- no IMU, no sensors, just a")
    print("scripted test. Watch specifically whether the thigh itself")
    print("swings (correct) or only the lower leg moves (wrong joint).")
    print("Close the viewer window, or wait, to end.\n")

    t0 = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 150
        while viewer.is_running() and time.time() - t0 < 10.0:
            data.qpos[hip_addr] = np.radians(30.0 * np.sin(2 * (time.time() - t0)))
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)

    print("Sanity check complete. If the WHOLE LEG (thigh included) visibly")
    print("swung back and forth from the hip, the joint name is correct and")
    print("mujoco_full_leg_stream.py's default will work as-is. If nothing")
    print("moved, or only part of the leg moved oddly, pass --hip-flexion-joint")
    print("with the correct name to mujoco_full_leg_stream.py.")


if __name__ == "__main__":
    main()