# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL (Advanced DDP: Render on specific Rank)."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# 添加 cli_args 路径
sys.path.append(os.path.join(os.path.dirname(__file__), "reinforcement_learning", "rsl_rl"))

try:
    import cli_args
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("cli_args", os.path.join(os.path.dirname(__file__), "reinforcement_learning/rsl_rl/cli_args.py"))
    if spec:
        cli_args = importlib.util.module_from_spec(spec)
        sys.modules["cli_args"] = cli_args
        spec.loader.exec_module(cli_args)
        cli_args.add_rsl_rl_args(parser)

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL (Advanced DDP).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument("--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None.")

# Custom arg: Render Rank
parser.add_argument("--render_rank", type=int, default=0, help="Which local rank should enable rendering/livestream (default: 0).")

if 'cli_args' in sys.modules:
    cli_args.add_rsl_rl_args(parser)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# --- Advanced DDP Logic ---
if args_cli.distributed:
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    
    # 逻辑：所有 Rank 必须 Headless (A10 不支持 GUI)
    # 指定 Rank 开启 Livestream，其他关闭
    args_cli.headless = True
    
    if local_rank == args_cli.render_rank:
        print(f"[INFO] Rank {local_rank} selected for LIVESTREAM (Headless).")
        # 强制开启 Livestream=2 (WebRTC)
        args_cli.livestream = 1
    else:
        # 其他 Rank 强制关闭直播
        args_cli.livestream = 0
        
else:
    print("[INFO] Single GPU mode.")

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""
# ... (Standard Check) ...
import importlib.metadata as metadata
from packaging import version
RSL_RL_VERSION = "3.0.1"
try:
    installed_version = metadata.version("rsl-rl-lib")
    if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
        print(f"Please update rsl-rl-lib to {RSL_RL_VERSION}")
        exit(1)
except Exception:
    pass

"""Rest everything follows."""

import logging
import time
from datetime import datetime
import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
import isaaclab_tasks
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError("Distributed training requires GPU.")

    if args_cli.distributed:
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        env_cfg.sim.device = f"cuda:{local_rank}"
        agent_cfg.device = f"cuda:{local_rank}"
        seed = agent_cfg.seed + local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    # 只有 Rank 0 负责创建日志目录和打印信息，防止冲突
    # 但每个 Rank 都需要知道路径
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # 录像逻辑：通常只让 Rank 0 录像
    # 修改：增加 Rank 检查，防止多进程写同一个文件
    if args_cli.video:
        is_rank0 = not args_cli.distributed or int(os.getenv("LOCAL_RANK", "0")) == 0
        if is_rank0:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    
    # 只有 Rank 0 记录 git info (避免文件冲突)
    if not args_cli.distributed or int(os.getenv("LOCAL_RANK", "0")) == 0:
        runner.add_git_repo_to_log(__file__)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        runner.load(resume_path)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
