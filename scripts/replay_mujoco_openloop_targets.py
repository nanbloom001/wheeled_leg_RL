import argparse
import numpy as np
import mujoco


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="isaac_openloop_targets.npz")
    parser.add_argument("--mjcf", default="WAVEGO_mujoco/scene.xml")
    parser.add_argument("--out", default="mujoco_openloop_results.npz")
    args = parser.parse_args()

    data_npz = np.load(args.targets)
    target_q = data_npz["target_q"]
    isaac_qerr = data_npz["isaac_qerr"]
    isaac_sat = data_npz["isaac_sat"]
    isaac_velx = data_npz["isaac_velx"]
    dt = float(data_npz["dt"][0])
    decimation = int(data_npz["decimation"][0])

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    model.opt.timestep = dt
    d = mujoco.MjData(model)

    effort_limit = float(np.abs(model.actuator_forcerange[0, 1]))

    # 与 sim2sim 相同初始化
    standing = np.array([0.1, -0.65, 0.6, -0.1, 0.65, -0.6, -0.1, -0.65, 0.6, 0.1, 0.65, -0.6], dtype=np.float64)
    d.qpos[:] = 0.0
    d.qvel[:] = 0.0
    d.qpos[2] = 0.25
    d.qpos[3] = 1.0
    d.qpos[7:] = standing
    d.ctrl[:] = standing
    mujoco.mj_forward(model, d)

    muj_qerr = []
    muj_sat = []
    muj_velx = []

    print(f"[INFO] Replay MuJoCo open-loop targets: steps={len(target_q)}, effort_limit={effort_limit:.2f}")
    for i in range(len(target_q)):
        d.ctrl[:] = target_q[i]
        for _ in range(decimation):
            mujoco.mj_step(model, d)

        q = d.qpos[7:]
        tq = d.qfrc_actuator[6:18]
        qerr = np.mean(np.abs(q - target_q[i]))
        sat = np.mean(np.abs(tq) >= (0.98 * effort_limit))
        velx = d.qvel[0]

        muj_qerr.append(qerr)
        muj_sat.append(sat)
        muj_velx.append(velx)

        if (i + 1) % 100 == 0:
            print(f"[Step {i+1:4d}] vel_x={velx:+.3f}, qerr_mean={qerr:.3f}, sat={sat*100:.0f}%")

    muj_qerr = np.asarray(muj_qerr)
    muj_sat = np.asarray(muj_sat)
    muj_velx = np.asarray(muj_velx)

    np.savez(
        args.out,
        mujoco_qerr=muj_qerr,
        mujoco_sat=muj_sat,
        mujoco_velx=muj_velx,
        isaac_qerr=isaac_qerr,
        isaac_sat=isaac_sat,
        isaac_velx=isaac_velx,
    )

    print("\n=== Open-loop Cross-Engine Comparison ===")
    print(f"Isaac mean qerr:  {np.mean(isaac_qerr):.4f}")
    print(f"MuJoCo mean qerr: {np.mean(muj_qerr):.4f}")
    print(f"Isaac mean sat:   {np.mean(isaac_sat):.4f}")
    print(f"MuJoCo mean sat:  {np.mean(muj_sat):.4f}")
    print(f"Isaac mean |velx|:{np.mean(np.abs(isaac_velx)):.4f}")
    print(f"MuJoCo mean |velx|:{np.mean(np.abs(muj_velx)):.4f}")

    ratio_qerr = np.mean(muj_qerr) / max(np.mean(isaac_qerr), 1e-8)
    ratio_sat = np.mean(muj_sat) / max(np.mean(isaac_sat), 1e-8)
    print(f"qerr ratio (muj/isaac): {ratio_qerr:.2f}x")
    print(f"sat ratio  (muj/isaac): {ratio_sat:.2f}x")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
