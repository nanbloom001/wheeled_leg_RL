import argparse
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/model_2999.pt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--cmd-x", type=float, default=0.0)
    parser.add_argument("--cmd-y", type=float, default=0.0)
    parser.add_argument("--cmd-wz", type=float, default=0.0)
    parser.add_argument("--out", default="isaac_openloop_targets.npz")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher({"headless": args.headless})
    simulation_app = app_launcher.app

    import gymnasium as gym
    from isaaclab_tasks.manager_based.locomotion.velocity.config.wavego.flat_env_cfg import WavegoFlatEnvCfg

    cfg = WavegoFlatEnvCfg()
    cfg.scene.num_envs = 1
    cfg.observations.policy.enable_corruption = False
    cfg.events.push_robot = None
    cfg.events.physics_material = None
    cfg.events.add_base_mass = None
    cfg.events.base_com = None

    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.ranges.lin_vel_x = (args.cmd_x, args.cmd_x)
    cfg.commands.base_velocity.ranges.lin_vel_y = (args.cmd_y, args.cmd_y)
    cfg.commands.base_velocity.ranges.ang_vel_z = (args.cmd_wz, args.cmd_wz)

    env = gym.make("Isaac-Velocity-Flat-WAVEGO-v0", cfg=cfg)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sd = ckpt["model_state_dict"]

    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(48, 512), torch.nn.ELU(),
                torch.nn.Linear(512, 256), torch.nn.ELU(),
                torch.nn.Linear(256, 128), torch.nn.ELU(),
                torch.nn.Linear(128, 12),
            )

        def forward(self, x):
            return self.net(x)

    policy = Actor()
    clean_dict = {
        k.replace("actor.", "net."): v
        for k, v in sd.items()
        if any(t in k for t in ["actor.0", "actor.2", "actor.4", "actor.6"])
    }
    policy.load_state_dict(clean_dict)
    policy.eval()

    obs, _ = env.reset()
    dev = obs["policy"].device
    policy = policy.to(dev)
    obs_mean = sd["actor_obs_normalizer._mean"].to(dev)
    obs_std = sd["actor_obs_normalizer._std"].to(dev)

    robot = env.unwrapped.scene["robot"]
    effort_limit = 1.96

    target_log = []
    qpos_log = []
    torque_log = []
    sat_log = []
    velx_log = []
    qerr_log = []

    print(f"[INFO] Export Isaac open-loop targets: steps={args.steps}, cmd={[args.cmd_x, args.cmd_y, args.cmd_wz]}")

    for i in range(args.steps):
        o = obs["policy"]
        o_n = torch.clamp((o - obs_mean) / (obs_std + 1e-8), -5, 5)

        with torch.no_grad():
            a = policy(o_n)

        # Isaac 动作链路：target = default + action*scale（scale=0.25）
        default_q = robot.data.default_joint_pos[0]
        target_q = (a[0] * 0.25 + default_q).detach().cpu().numpy()

        obs, _, _, _, _ = env.step(a)

        q = robot.data.joint_pos[0].detach().cpu().numpy()
        tq = robot.data.applied_torque[0].detach().cpu().numpy()
        velx = robot.data.root_lin_vel_b[0, 0].item()

        qerr = np.abs(q - target_q)
        sat = np.mean(np.abs(tq) >= (0.98 * effort_limit))

        target_log.append(target_q)
        qpos_log.append(q)
        torque_log.append(tq)
        sat_log.append(sat)
        velx_log.append(velx)
        qerr_log.append(np.mean(qerr))

        if (i + 1) % 100 == 0:
            print(
                f"[Step {i+1:4d}] vel_x={velx:+.3f}, qerr_mean={np.mean(qerr):.3f}, "
                f"sat={sat*100:.0f}%"
            )

    target_log = np.asarray(target_log)
    qpos_log = np.asarray(qpos_log)
    torque_log = np.asarray(torque_log)
    sat_log = np.asarray(sat_log)
    velx_log = np.asarray(velx_log)
    qerr_log = np.asarray(qerr_log)

    np.savez(
        args.out,
        target_q=target_log,
        isaac_q=qpos_log,
        isaac_torque=torque_log,
        isaac_sat=sat_log,
        isaac_velx=velx_log,
        isaac_qerr=qerr_log,
        cmd=np.array([args.cmd_x, args.cmd_y, args.cmd_wz], dtype=np.float64),
        dt=np.array([0.005], dtype=np.float64),
        decimation=np.array([4], dtype=np.int32),
    )

    print("\n=== Isaac Open-loop Export Summary ===")
    print(f"mean |vel_x|: {np.mean(np.abs(velx_log)):.4f}")
    print(f"mean qerr: {np.mean(qerr_log):.4f}")
    print(f"mean sat: {np.mean(sat_log):.4f}")
    print(f"saved: {args.out}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
