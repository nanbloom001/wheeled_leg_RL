from __future__ import annotations

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL (DDP with single-rank video)."""

import argparse
import importlib.metadata as metadata
import importlib.util
import logging
import os
import platform
import sys
import time
from datetime import datetime

from packaging import version

from isaaclab.app import AppLauncher


def _load_cli_args_module():
    cli_args_path = os.path.join(
        os.path.dirname(__file__), "reinforcement_learning", "rsl_rl", "cli_args.py"
    )
    cli_args_dir = os.path.dirname(cli_args_path)
    if cli_args_dir not in sys.path:
        sys.path.append(cli_args_dir)

    try:
        import cli_args as cli_args_module
    except ImportError:
        spec = importlib.util.spec_from_file_location("cli_args", cli_args_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load cli_args from {cli_args_path}")
        cli_args_module = importlib.util.module_from_spec(spec)
        sys.modules["cli_args"] = cli_args_module
        spec.loader.exec_module(cli_args_module)

    return cli_args_module


def _get_local_rank(distributed: bool) -> int:
    return int(os.getenv("LOCAL_RANK", "0")) if distributed else 0


def _is_rank_zero(distributed: bool, local_rank: int) -> bool:
    return (not distributed) or local_rank == 0


def _read_proc_int(path: str) -> int | None:
    try:
        with open(path, encoding="utf-8") as file:
            return int(file.read().strip())
    except (OSError, ValueError):
        return None


def _video_step_trigger(step: int, interval: int) -> bool:
    return step > 0 and step % interval == 0


def _resolve_resume_path(log_root_path: str, agent_cfg: RslRlBaseRunnerCfg) -> str:
    checkpoint_arg = agent_cfg.load_checkpoint
    if checkpoint_arg and os.path.isfile(checkpoint_arg):
        return os.path.abspath(checkpoint_arg)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _prime_video_rendering(env, enable_video: bool, local_rank: int) -> None:
    env.reset()
    if enable_video:
        print(f"[INFO] Priming render pipeline on rank {local_rank} before enabling video capture.")
        env.render()
        env.render()
    env.reset()


def _configure_video_view(env_cfg, enable_video: bool, local_rank: int) -> None:
    if not enable_video or not hasattr(env_cfg, "viewer"):
        return

    # We keep the default world view as requested by the user
    if hasattr(env_cfg, "num_rerenders_on_reset"):
        env_cfg.num_rerenders_on_reset = max(int(env_cfg.num_rerenders_on_reset), 1)

    print(f"[INFO] Video capture viewer configured on rank {local_rank} (using default world coordinates).")


cli_args = _load_cli_args_module()

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL (Advanced DDP).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument(
    "--render_rank", type=int, default=0, help="Which local rank should enable cameras/video in distributed mode."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

local_rank = _get_local_rank(args_cli.distributed)
user_requested_livestream = any(
    arg == "--livestream" or arg.startswith("--livestream=") for arg in sys.argv[1:]
)

args_cli._video_requested = bool(args_cli.video)
args_cli._video_rank = (not args_cli.distributed) or local_rank == args_cli.render_rank
args_cli.video = args_cli._video_requested and args_cli._video_rank
args_cli.enable_cameras = bool(args_cli.video)

if args_cli.distributed:
    args_cli.headless = True
    if args_cli._video_requested:
        if args_cli._video_rank:
            if not (user_requested_livestream and args_cli.livestream in (1, 2)):
                args_cli.livestream = 2
        else:
            args_cli.livestream = 0
    elif user_requested_livestream and args_cli.livestream in (1, 2):
        if not args_cli._video_rank:
            args_cli.livestream = 0
    else:
        args_cli.livestream = 0

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    raise SystemExit(1)

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
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
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. Please use GPU device for distributed training."
        )

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    if isinstance(env_cfg, ManagerBasedRLEnvCfg) and args_cli._video_requested:
        commands_cfg = getattr(env_cfg, "commands", None)
        base_velocity_cfg = getattr(commands_cfg, "base_velocity", None)
        if base_velocity_cfg is not None and hasattr(base_velocity_cfg, "debug_vis"):
            base_velocity_cfg.debug_vis = False

    _configure_video_view(env_cfg, args_cli.video, local_rank)

    if _is_rank_zero(args_cli.distributed, local_rank) and args_cli._video_requested:
        max_user_watches = _read_proc_int("/proc/sys/fs/inotify/max_user_watches")
        max_user_instances = _read_proc_int("/proc/sys/fs/inotify/max_user_instances")
        if (max_user_watches is not None and max_user_watches < 262144) or (
            max_user_instances is not None and max_user_instances < 512
        ):
            print("[WARN] Low inotify limits detected. DDP + video may fail with 'No space left on device'.")

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if _is_rank_zero(args_cli.distributed, local_rank):
        print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if _is_rank_zero(args_cli.distributed, local_rank):
        print(f"Exact experiment name requested from command line: {log_dir_name}")
    if agent_cfg.run_name:
        log_dir_name += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir_name)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    env_cfg.log_dir = log_dir
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    _prime_video_rendering(env, args_cli.video, local_rank)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = _resolve_resume_path(log_root_path, agent_cfg)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: _video_step_trigger(step, args_cli.video_interval),
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Rank {local_rank} recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    if _is_rank_zero(args_cli.distributed, local_rank):
        runner.add_git_repo_to_log(__file__)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        if _is_rank_zero(args_cli.distributed, local_rank):
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    if _is_rank_zero(args_cli.distributed, local_rank):
        print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()