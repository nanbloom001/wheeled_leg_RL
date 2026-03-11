import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/model_2999.pt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher({"headless": args.headless})
    simulation_app = app_launcher.app

    import gymnasium as gym
    from isaaclab_tasks.manager_based.locomotion.velocity.config.wavego.flat_env_cfg import WavegoFlatEnvCfg

    env_cfg = WavegoFlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.push_robot = None
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.rel_standing_envs = 1.0
    env_cfg.commands.base_velocity.heading_command = False

    env = gym.make("Isaac-Velocity-Flat-WAVEGO-v0", cfg=env_cfg)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    sd = checkpoint["model_state_dict"]

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
    clean_dict = {k.replace("actor.", "net."): v for k, v in sd.items() if any(t in k for t in ["actor.0", "actor.2", "actor.4", "actor.6"])}
    policy.load_state_dict(clean_dict)
    policy.eval()

    obs, _ = env.reset()
    device = obs["policy"].device
    policy = policy.to(device)
    obs_mean = sd["actor_obs_normalizer._mean"].to(device)
    obs_std = sd["actor_obs_normalizer._std"].to(device)

    sat_hist = []
    vel_hist = []
    err_hist = []

    print("[INFO] Running zero-command check in Isaac...")
    for i in range(1, args.steps + 1):
        o = obs["policy"]
        o_n = torch.clamp((o - obs_mean) / (obs_std + 1e-8), -5, 5)
        with torch.no_grad():
            a = policy(o_n)

        obs, _, _, _, _ = env.step(a)

        robot = env.unwrapped.scene["robot"]
        q = robot.data.joint_pos[0]
        qd = robot.data.default_joint_pos[0]
        qerr = (q - qd).abs().mean().item()
        velx = robot.data.root_lin_vel_b[0, 0].item()
        tq = robot.data.applied_torque[0].abs()
        # use effort limit from config (scalar per joint in this env)
        sat = (tq >= (1.96 * 0.98)).float().mean().item()

        sat_hist.append(sat)
        vel_hist.append(abs(velx))
        err_hist.append(qerr)

        if i % 100 == 0:
            print(f"[Step {i:4d}] |vel_x|={abs(velx):.3f}, joint_err_mean={qerr:.3f}, sat={sat*100:.0f}%")

    print("\n=== Isaac zero-command summary ===")
    print(f"mean |vel_x|: {sum(vel_hist)/len(vel_hist):.4f}")
    print(f"mean joint_err: {sum(err_hist)/len(err_hist):.4f}")
    print(f"mean sat_ratio: {sum(sat_hist)/len(sat_hist):.4f}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
