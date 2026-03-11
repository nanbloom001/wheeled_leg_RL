import argparse
import itertools
import numpy as np
import pandas as pd
import mujoco
from scipy.spatial.transform import Rotation as R


def pitch_deg_from_quat_wxyz(quat_wxyz: np.ndarray) -> float:
    r = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    roll, pitch, yaw = r.as_euler("xyz", degrees=True)
    return float(pitch)


def run_one(model_path: str, target_q: np.ndarray, dt: float, decimation: int, params: dict) -> dict:
    model = mujoco.MjModel.from_xml_path(model_path)
    model.opt.timestep = dt
    d = mujoco.MjData(model)

    joint_slice = slice(6, 18)
    model.dof_damping[joint_slice] = params["joint_damping"]
    model.dof_frictionloss[joint_slice] = params["joint_frictionloss"]
    model.dof_armature[joint_slice] = params["joint_armature"]

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        model.geom_friction[floor_id, 0] = params["floor_friction"]

    standing = np.array([0.1, -0.65, 0.6, -0.1, 0.65, -0.6, -0.1, -0.65, 0.6, 0.1, 0.65, -0.6], dtype=np.float64)
    d.qpos[:] = 0.0
    d.qvel[:] = 0.0
    d.qpos[2] = 0.25
    d.qpos[3] = 1.0
    d.qpos[7:] = standing
    d.ctrl[:] = standing
    mujoco.mj_forward(model, d)

    effort_limit = float(np.abs(model.actuator_forcerange[0, 1]))

    qerr = []
    sat = []
    velx = []
    fall = []

    for i in range(len(target_q)):
        d.ctrl[:] = target_q[i]
        for _ in range(decimation):
            mujoco.mj_step(model, d)

        q = d.qpos[7:]
        tq = d.qfrc_actuator[6:18]
        qe = float(np.mean(np.abs(q - target_q[i])))
        sr = float(np.mean(np.abs(tq) >= (0.98 * effort_limit)))
        vx = float(d.qvel[0])

        pitch = abs(pitch_deg_from_quat_wxyz(d.qpos[3:7]))
        h = float(d.qpos[2])
        is_fall = 1.0 if (h < 0.11 or pitch > 70.0) else 0.0

        qerr.append(qe)
        sat.append(sr)
        velx.append(abs(vx))
        fall.append(is_fall)

    qerr_m = float(np.mean(qerr))
    sat_m = float(np.mean(sat))
    velx_m = float(np.mean(velx))
    fall_m = float(np.mean(fall))

    score = 4.0 * sat_m + 2.0 * qerr_m + 3.0 * fall_m + 0.5 * velx_m

    return {
        **params,
        "qerr_mean": qerr_m,
        "sat_mean": sat_m,
        "velx_abs_mean": velx_m,
        "fall_ratio": fall_m,
        "score": score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="isaac_openloop_targets.npz")
    parser.add_argument("--mjcf", default="WAVEGO_mujoco/scene.xml")
    parser.add_argument("--out", default="mujoco_openloop_sweep.csv")
    args = parser.parse_args()

    data = np.load(args.targets)
    target_q = data["target_q"]
    dt = float(data["dt"][0])
    decimation = int(data["decimation"][0])

    grid = {
        "joint_damping": [0.0, 0.1, 0.2],
        "joint_frictionloss": [0.2, 0.4, 0.6],
        "joint_armature": [0.005, 0.01, 0.02],
        "floor_friction": [0.8, 1.0, 1.2],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    print(f"[INFO] Running open-loop dynamics sweep: {len(combos)} configs")
    rows = []
    for i, vals in enumerate(combos, start=1):
        params = {k: v for k, v in zip(keys, vals)}
        result = run_one(args.mjcf, target_q, dt, decimation, params)
        rows.append(result)
        print(
            f"[{i:02d}/{len(combos)}] score={result['score']:.4f} "
            f"sat={result['sat_mean']:.3f} qerr={result['qerr_mean']:.3f} "
            f"fall={result['fall_ratio']:.3f} params={params}"
        )

    df = pd.DataFrame(rows).sort_values("score")
    df.to_csv(args.out, index=False)

    print("\n=== Top-10 configs ===")
    print(df.head(10).to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
