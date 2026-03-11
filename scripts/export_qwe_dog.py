
"""
Export QWE_DOG RSL-RL policy to ONNX format.
"""

import argparse
import os
import sys
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Export RSL-RL policy to ONNX.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-QweDog-Linear-Fast-v0", help="Name of the task.")
parser.add_argument("--run_dir", type=str, required=True, help="Path to the run directory.")
parser.add_argument("--filename", type=str, default="model_1749.pt", help="Name of the checkpoint file.")
parser.add_argument("--output_name", type=str, default="policy.onnx", help="Name of the output ONNX file.")
# append AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -----------------------------------------------------------
# import isaaclab modules after launching app
from rsl_rl.runners import OnPolicyRunner

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import export_policy_as_onnx, export_policy_as_jit

def main():
    # resolve task and agent config
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    
    # override number of environments to 1 for export
    env_cfg.scene.num_envs = 1
    # disable serialization to avoid pickle issues during export (if any)
    # env_cfg.scene.replicate_physics = False

    # create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # prepare agent config
    agent_cfg["device"] = env.device
    agent_cfg["num_envs"] = env.num_envs
    agent_cfg["obs_dim"] = env.num_observations
    agent_cfg["critic_obs_dim"] = env.num_states
    agent_cfg["action_dim"] = env.num_actions

    # create runner
    # RSL_RL runner handles loading the policy and normalizer from checkpoint
    log_dir = os.path.dirname(args.run_dir) # parent dir of specific run
    run_name = os.path.basename(args.run_dir) # specific run name
    
    # HACK: OnPolicyRunner needs a log_dir that contains the runs. 
    # If args.run_dir is ".../qwe_dog_flat/2026-01-21...", we pass ".../qwe_dog_flat" and run_name="2026-01-21..."
    
    runner = OnPolicyRunner(env, agent_cfg, log_dir=log_dir, run_name=run_name)
    
    # load checkpoint
    resume_path = os.path.join(args.run_dir, args.filename)
    print(f"[INFO] Loading checkpoint from: {resume_path}")
    runner.load(resume_path)

    # set evaluation mode
    runner.policy.eval()

    # get policy and normalizer
    policy = runner.alg.actor_critic.actor
    
    # check if normalizer exists in the algorithm state
    normalizer = None
    if hasattr(runner.alg.actor_critic, "obs_std") and hasattr(runner.alg.actor_critic, "obs_mean"):
        # construct a mock normalizer if RSL_RL uses internal mean/std
        # But RSL_RL usually uses an EmpiricalNormalization module if configured.
        # Let's check agent_cfg to see if empirical_normalization is true.
        if agent_cfg.get("empirical_normalization", False):
            print("[INFO] Empirical normalization is enabled. Extracting running mean and var.")
            # RSL_RL stores running mean/std in the runner.alg.actor_critic if it's part of the model, 
            # OR in the runner.obs_normalizer if it's separate.
            # Checking RSL_RL source implies it might be handled differently versions to versions.
            # The Isaac Lab exporter expects a module that has .forward(x) -> norm_x.
            
            # For simplicity, if we can't find a module, we might need to manually construct one or trust the policy has it built-in.
            # However, standard RSL_RL usually keeps raw policy and handles normalization outside.
            pass

    # Actually, OnPolicyRunner doesn't expose 'normalizer' module easily if it's the internal one.
    # IsaacLab's exporter expects a module.
    # Let's try to pass the whole actor_critic as policy if it contains the normalizer?
    # No, exporter expects 'policy' to be actor.
    
    # Correct approach for RSL_RL in IsaacLab:
    # RSL_RL runner usually has 'obs_normalizer' if configured.
    if hasattr(runner, "obs_normalizer"):
        normalizer = runner.obs_normalizer
        print(f"[INFO] Found observation normalizer: {normalizer}")

    # export
    export_path = os.path.join(args.run_dir, "exported")
    print(f"[INFO] Exporting to: {export_path}")
    
    # Export ONNX
    export_policy_as_onnx(runner.alg.actor_critic, export_path, normalizer=normalizer, filename=args.output_name)
    print("[INFO] ONNX export complete.")
    
    # Export JIT (as backup)
    export_policy_as_jit(runner.alg.actor_critic, normalizer, export_path, filename="policy.pt")
    print("[INFO] JIT export complete.")

    # close environment
    env.close()

if __name__ == "__main__":
    main()
