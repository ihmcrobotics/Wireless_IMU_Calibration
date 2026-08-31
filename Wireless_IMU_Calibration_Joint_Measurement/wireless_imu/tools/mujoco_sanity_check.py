#!/usr/bin/env python3
"""
mujoco_sanity_check.py  -  confirm MuJoCo + myosuite work at all, with NO
IMU code involved. Run this BEFORE mujoco_knee_stream.py.

Isolates one question: does the viewer launch and show the model, with the
knee joint moving on a simple scripted sweep? If this fails, the problem is
in the MuJoCo/myosuite install or model path -- not in any IMU wiring, so
don't go looking there.
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
KNEE_JOINT = "knee_angle_r"


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

    knee_addr = None
    for i in range(model.njnt):
        if model.joint(i).name == KNEE_JOINT:
            knee_addr = model.jnt_qposadr[i]
    if knee_addr is None:
        print(f"[WARN] joint '{KNEE_JOINT}' not found -- check the joint name "
              f"in this myosuite version (mj_id2name / model.joint(i).name "
              f"for i in range(model.njnt) to list them all).")
    else:
        print(f"knee joint found at qpos[{knee_addr}]")

    mujoco.mj_forward(model, data)

    print("\nLaunching viewer. The knee should sweep back and forth on its")
    print("own for 10 seconds -- no IMU, no sensors, just a scripted test.")
    print("Close the viewer window, or wait, to end.\n")

    t0 = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 150
        while viewer.is_running() and time.time() - t0 < 10.0:
            if knee_addr is not None:
                data.qpos[knee_addr] = np.radians(30.0 * np.sin(2 * (time.time() - t0)))
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)

    print("Sanity check complete. If the knee visibly swept back and forth,")
    print("MuJoCo + myosuite are working correctly -- any problem in the full")
    print("script is in the IMU wiring, not here.")


if __name__ == "__main__":
    main()