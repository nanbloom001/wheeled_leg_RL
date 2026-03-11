#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare IsaacLab and MuJoCo rollout traces step-by-step.")
    parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-WAVEGO-v0-Play")
    parser.add_argument("--scene", type=str, default="WAVEGO_mujoco/scene.xml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/model_2999.pt",
    )
    parser.add_argument(
        "--env-yaml",
        type=str,
        default="logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/params/env.yaml",
    )
    parser.add_argument(
        "--io-descriptor",
        type=str,
        default="tmp/io_descriptors_wavego/isaac_velocity_flat_wavego_v0_IO_descriptors.yaml",
    )
    parser.add_argument("--cmd-x", type=float, default=0.5)
    parser.add_argument("--cmd-y", type=float, default=0.0)
    parser.add_argument("--cmd-wz", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-csv", type=str, default="tmp/isaac_mujoco_step_compare.csv")
    parser.add_argument("--output-npz", type=str, default="tmp/isaac_mujoco_step_compare.npz")
    parser.add_argument("--joint-damping", type=float, default=None)
    parser.add_argument("--joint-frictionloss-scale", type=float, default=1.0)
    parser.add_argument("--armature-scale", type=float, default=1.0)
    parser.add_argument("--floor-friction-scale", type=float, default=1.0)
    parser.add_argument("--solver-iterations", type=int, default=None)
    parser.add_argument("--noslip-iterations", type=int, default=None)
    return parser.parse_args()


def _disable_randomization(env_cfg):
    if hasattr(env_cfg, "events"):
        for name in [
            "physics_material",
            "add_base_mass",
            "base_com",
            "base_external_force_torque",
            "reset_base",
            "reset_robot_joints",
            "push_robot",
        ]:
            if hasattr(env_cfg.events, name):
                setattr(env_cfg.events, name, None)


def _configure_command(env_cfg):
    cmd = env_cfg.commands.base_velocity
    cmd.heading_command = False
    cmd.rel_heading_envs = 0.0
    cmd.rel_standing_envs = 0.0
    cmd.resampling_time_range = (1.0e9, 1.0e9)


def _force_fixed_command(env, command: np.ndarray):
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    cmd_term.vel_command_b[:, 0] = float(command[0])
    cmd_term.vel_command_b[:, 1] = float(command[1])
    cmd_term.vel_command_b[:, 2] = float(command[2])
    if hasattr(cmd_term, "is_standing_env"):
        cmd_term.is_standing_env[:] = False
    if hasattr(cmd_term, "is_heading_env"):
        cmd_term.is_heading_env[:] = False


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _yaw_from_wxyz(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _apply_mujoco_runtime_overrides(model: mujoco.MjModel, args: argparse.Namespace) -> dict[str, float]:
    overrides: dict[str, float] = {}

    # DOF-level overrides for all non-root joints.
    dof_ids = [int(model.jnt_dofadr[jid]) for jid in range(1, model.njnt)]
    dof_ids_arr = np.asarray(dof_ids, dtype=np.int64)
    if args.joint_damping is not None:
        model.dof_damping[dof_ids_arr] = float(args.joint_damping)
        overrides["joint_damping"] = float(args.joint_damping)

    if args.joint_frictionloss_scale != 1.0:
        model.dof_frictionloss[dof_ids_arr] *= float(args.joint_frictionloss_scale)
        overrides["joint_frictionloss_scale"] = float(args.joint_frictionloss_scale)

    if args.armature_scale != 1.0:
        model.dof_armature[dof_ids_arr] *= float(args.armature_scale)
        overrides["armature_scale"] = float(args.armature_scale)

    # Global solver/contact overrides.
    if args.solver_iterations is not None:
        model.opt.iterations = int(args.solver_iterations)
        overrides["solver_iterations"] = float(args.solver_iterations)

    if args.noslip_iterations is not None:
        model.opt.noslip_iterations = int(args.noslip_iterations)
        overrides["noslip_iterations"] = float(args.noslip_iterations)

    if args.floor_friction_scale != 1.0:
        floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_geom_id >= 0:
            model.geom_friction[floor_geom_id, :] *= float(args.floor_friction_scale)
            overrides["floor_friction_scale"] = float(args.floor_friction_scale)

    return overrides


def main():
    args = parse_args()

    app_launcher = AppLauncher({"headless": args.headless, "device": args.device})
    simulation_app = app_launcher.app

    workspace_root = Path(__file__).resolve().parents[1]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    from scripts.sim2sim_test import (
        ActorMLP,
        apply_obs_normalization,
        extract_policy_observation_from_mujoco,
        load_actor_obs_normalizer_from_checkpoint,
        load_isaac_params_from_env_yaml,
        load_policy_io_config,
        reorder_policy_to_mj,
        sync_mujoco_timing,
        validate_mujoco_joint_actuator_order,
        verify_and_sync_actuator_params,
        _policy_to_mj_index,
    )

    command = np.array([args.cmd_x, args.cmd_y, args.cmd_wz], dtype=np.float64)

    io_cfg = load_policy_io_config(args.io_descriptor)

    # MuJoCo setup
    mj_model = mujoco.MjModel.from_xml_path(args.scene)
    mj_joint_names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, mj_model.njnt)]
    policy_to_mj_idx = _policy_to_mj_index(io_cfg.policy_joint_names, mj_joint_names)
    default_joint_pos_mj = reorder_policy_to_mj(io_cfg.default_joint_pos_policy, policy_to_mj_idx)

    isaac_params = load_isaac_params_from_env_yaml(args.env_yaml, default_joint_pos_mj=default_joint_pos_mj)
    policy_update_interval = sync_mujoco_timing(mj_model, isaac_params)
    verify_and_sync_actuator_params(mj_model, isaac_params)
    overrides = _apply_mujoco_runtime_overrides(mj_model, args)
    mj_data = mujoco.MjData(mj_model)
    validate_mujoco_joint_actuator_order(mj_model)

    mj_data.qpos[2] = 0.25
    mj_data.qpos[7:] = default_joint_pos_mj
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    # Isaac setup
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
    _disable_randomization(env_cfg)
    _configure_command(env_cfg)
    env_cfg.observations.policy.enable_corruption = False
    env = gym.make(args.task, cfg=env_cfg)
    obs_dict, _ = env.reset()

    # force command after reset
    _force_fixed_command(env, command)
    zero_action = torch.zeros((1, 12), dtype=torch.float32, device=env.unwrapped.device)
    obs_dict, _, _, _, _ = env.step(zero_action)

    policy = ActorMLP(args.checkpoint)
    normalizer = load_actor_obs_normalizer_from_checkpoint(args.checkpoint)

    prev_action_policy_isaac = np.zeros(12, dtype=np.float64)
    prev_action_policy_mj = np.zeros(12, dtype=np.float64)

    isaac_x0 = float(env.unwrapped.scene["robot"].data.root_pos_w[0, 0].item())
    isaac_y0 = float(env.unwrapped.scene["robot"].data.root_pos_w[0, 1].item())
    isaac_quat0 = env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
    isaac_yaw0 = _yaw_from_wxyz(isaac_quat0)
    mj_x0 = float(mj_data.qpos[0])
    mj_y0 = float(mj_data.qpos[1])
    mj_quat0 = np.asarray(mj_data.qpos[3:7], dtype=np.float64)
    mj_yaw0 = _yaw_from_wxyz(mj_quat0)

    isaac_fwd = np.array([np.cos(isaac_yaw0), np.sin(isaac_yaw0)], dtype=np.float64)
    isaac_lat = np.array([-np.sin(isaac_yaw0), np.cos(isaac_yaw0)], dtype=np.float64)
    mj_fwd = np.array([np.cos(mj_yaw0), np.sin(mj_yaw0)], dtype=np.float64)
    mj_lat = np.array([-np.sin(mj_yaw0), np.cos(mj_yaw0)], dtype=np.float64)

    rows = []
    obs_isaac_log = []
    obs_mj_log = []
    action_isaac_log = []
    action_mj_log = []

    for step in range(args.steps):
        # Isaac side
        _force_fixed_command(env, command)
        obs_isaac = obs_dict["policy"][0].detach().cpu().numpy().astype(np.float64)
        obs_isaac_norm = apply_obs_normalization(obs_isaac, normalizer)
        action_isaac = policy(obs_isaac_norm)

        action_tensor = torch.from_numpy(action_isaac.astype(np.float32)).unsqueeze(0).to(env.unwrapped.device)
        obs_dict, _, _, _, _ = env.step(action_tensor)
        prev_action_policy_isaac = action_isaac

        # MuJoCo side
        obs_mj = extract_policy_observation_from_mujoco(
            model=mj_model,
            data=mj_data,
            default_joint_pos_policy=io_cfg.default_joint_pos_policy,
            previous_action_policy=prev_action_policy_mj,
            command=command,
            policy_to_mj_idx=policy_to_mj_idx,
        )
        obs_mj_norm = apply_obs_normalization(obs_mj, normalizer)
        action_mj = policy(obs_mj_norm)

        action_mj_order = reorder_policy_to_mj(action_mj, policy_to_mj_idx)
        ctrl_target = action_mj_order * io_cfg.action_scale + default_joint_pos_mj
        mj_data.ctrl[:] = ctrl_target
        prev_action_policy_mj = action_mj

        for _ in range(policy_update_interval):
            mujoco.mj_step(mj_model, mj_data)

        obs_isaac_log.append(obs_isaac)
        obs_mj_log.append(obs_mj)
        action_isaac_log.append(action_isaac)
        action_mj_log.append(action_mj)

        row = {
            "step": step,
            "isaac_vx": float(obs_isaac[0]),
            "isaac_vy": float(obs_isaac[1]),
            "isaac_wz": float(obs_isaac[5]),
            "mujoco_vx": float(obs_mj[0]),
            "mujoco_vy": float(obs_mj[1]),
            "mujoco_wz": float(obs_mj[5]),
            "isaac_base_x": float(env.unwrapped.scene["robot"].data.root_pos_w[0, 0].item()),
            "isaac_base_y": float(env.unwrapped.scene["robot"].data.root_pos_w[0, 1].item()),
            "isaac_base_z": float(env.unwrapped.scene["robot"].data.root_pos_w[0, 2].item()),
            "mujoco_base_x": float(mj_data.qpos[0]),
            "mujoco_base_y": float(mj_data.qpos[1]),
            "mujoco_base_z": float(mj_data.qpos[2]),
            "obs_l2_diff": float(np.linalg.norm(obs_isaac - obs_mj)),
            "action_l2_diff": float(np.linalg.norm(action_isaac - action_mj)),
        }
        row["isaac_dx"] = row["isaac_base_x"] - isaac_x0
        row["isaac_dy"] = row["isaac_base_y"] - isaac_y0
        row["mujoco_dx"] = row["mujoco_base_x"] - mj_x0
        row["mujoco_dy"] = row["mujoco_base_y"] - mj_y0
        row["pos_err_norm"] = float(
            np.linalg.norm([row["isaac_dx"] - row["mujoco_dx"], row["isaac_dy"] - row["mujoco_dy"]])
        )
        rows.append(row)

    obs_isaac_arr = np.asarray(obs_isaac_log)
    obs_mj_arr = np.asarray(obs_mj_log)
    action_isaac_arr = np.asarray(action_isaac_log)
    action_mj_arr = np.asarray(action_mj_log)

    obs_diff = obs_isaac_arr - obs_mj_arr
    action_diff = action_isaac_arr - action_mj_arr

    obs_group_rmse = {
        "base_lin_vel": float(np.sqrt(np.mean(np.square(obs_diff[:, 0:3])))),
        "base_ang_vel": float(np.sqrt(np.mean(np.square(obs_diff[:, 3:6])))),
        "projected_gravity": float(np.sqrt(np.mean(np.square(obs_diff[:, 6:9])))),
        "velocity_commands": float(np.sqrt(np.mean(np.square(obs_diff[:, 9:12])))),
        "joint_pos_rel": float(np.sqrt(np.mean(np.square(obs_diff[:, 12:24])))),
        "joint_vel_rel": float(np.sqrt(np.mean(np.square(obs_diff[:, 24:36])))),
        "last_action": float(np.sqrt(np.mean(np.square(obs_diff[:, 36:48])))),
    }

    # macro metrics (trajectory / position)
    isaac_dx = np.array([r["isaac_dx"] for r in rows], dtype=np.float64)
    isaac_dy = np.array([r["isaac_dy"] for r in rows], dtype=np.float64)
    mujoco_dx = np.array([r["mujoco_dx"] for r in rows], dtype=np.float64)
    mujoco_dy = np.array([r["mujoco_dy"] for r in rows], dtype=np.float64)
    pos_err = np.sqrt((isaac_dx - mujoco_dx) ** 2 + (isaac_dy - mujoco_dy) ** 2)
    macro_metrics = {
        "final_pos_err_norm": float(pos_err[-1]),
        "mean_pos_err_norm": float(np.mean(pos_err)),
        "max_pos_err_norm": float(np.max(pos_err)),
        "isaac_final_dx": float(isaac_dx[-1]),
        "isaac_final_dy": float(isaac_dy[-1]),
        "mujoco_final_dx": float(mujoco_dx[-1]),
        "mujoco_final_dy": float(mujoco_dy[-1]),
    }

    isaac_final_disp_world = np.array([isaac_dx[-1], isaac_dy[-1]], dtype=np.float64)
    mj_final_disp_world = np.array([mujoco_dx[-1], mujoco_dy[-1]], dtype=np.float64)
    macro_aligned_metrics = {
        "isaac_forward_disp": float(np.dot(isaac_final_disp_world, isaac_fwd)),
        "isaac_lateral_disp": float(np.dot(isaac_final_disp_world, isaac_lat)),
        "mujoco_forward_disp": float(np.dot(mj_final_disp_world, mj_fwd)),
        "mujoco_lateral_disp": float(np.dot(mj_final_disp_world, mj_lat)),
    }
    macro_aligned_metrics["forward_disp_err"] = abs(
        macro_aligned_metrics["isaac_forward_disp"] - macro_aligned_metrics["mujoco_forward_disp"]
    )
    macro_aligned_metrics["lateral_disp_err"] = abs(
        macro_aligned_metrics["isaac_lateral_disp"] - macro_aligned_metrics["mujoco_lateral_disp"]
    )

    # micro metrics (per-joint)
    joint_pos_diff = obs_diff[:, 12:24]
    joint_vel_diff = obs_diff[:, 24:36]
    per_joint_rmse = {}
    for j, name in enumerate(io_cfg.policy_joint_names):
        per_joint_rmse[name] = {
            "joint_pos_rel_rmse": float(np.sqrt(np.mean(np.square(joint_pos_diff[:, j])))),
            "joint_vel_rel_rmse": float(np.sqrt(np.mean(np.square(joint_vel_diff[:, j])))),
        }

    summary = {
        "steps": args.steps,
        "cmd": command.tolist(),
        "obs_total_rmse": float(np.sqrt(np.mean(np.square(obs_diff)))),
        "action_total_rmse": float(np.sqrt(np.mean(np.square(action_diff)))),
        "obs_group_rmse": obs_group_rmse,
        "macro_metrics": macro_metrics,
        "macro_aligned_metrics": macro_aligned_metrics,
        "micro_joint_rmse": per_joint_rmse,
        "isaac_vx_mean": float(np.mean(obs_isaac_arr[:, 0])),
        "mujoco_vx_mean": float(np.mean(obs_mj_arr[:, 0])),
        "isaac_vy_mean": float(np.mean(obs_isaac_arr[:, 1])),
        "mujoco_vy_mean": float(np.mean(obs_mj_arr[:, 1])),
        "isaac_wz_mean": float(np.mean(obs_isaac_arr[:, 5])),
        "mujoco_wz_mean": float(np.mean(obs_mj_arr[:, 5])),
        "mujoco_overrides": overrides,
    }

    # Combined optimization objective (smaller is better): macro + micro
    summary["score_macro_forward_lateral"] = float(
        summary["macro_aligned_metrics"]["forward_disp_err"]
        + summary["macro_aligned_metrics"]["lateral_disp_err"]
    )
    summary["score_micro_jointvel_action"] = float(
        summary["obs_group_rmse"]["joint_vel_rel"] + summary["action_total_rmse"]
    )
    summary["score_total"] = float(
        summary["score_macro_forward_lateral"] + summary["score_micro_jointvel_action"]
    )

    out_csv = Path(args.output_csv)
    out_npz = Path(args.output_npz)
    _write_csv(out_csv, rows)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        obs_isaac=obs_isaac_arr,
        obs_mujoco=obs_mj_arr,
        action_isaac=action_isaac_arr,
        action_mujoco=action_mj_arr,
        obs_diff=obs_diff,
        action_diff=action_diff,
        summary=np.array([summary], dtype=object),
    )

    print("COMPARE_DONE")
    print(f"steps={summary['steps']}, cmd={summary['cmd']}")
    print(f"obs_total_rmse={summary['obs_total_rmse']:.6f}, action_total_rmse={summary['action_total_rmse']:.6f}")
    print(
        f"mean_vx isaac={summary['isaac_vx_mean']:.4f} vs mujoco={summary['mujoco_vx_mean']:.4f}; "
        f"mean_vy isaac={summary['isaac_vy_mean']:.4f} vs mujoco={summary['mujoco_vy_mean']:.4f}; "
        f"mean_wz isaac={summary['isaac_wz_mean']:.4f} vs mujoco={summary['mujoco_wz_mean']:.4f}"
    )
    print("obs_group_rmse:", summary["obs_group_rmse"])
    print("macro_metrics:", summary["macro_metrics"])
    print("macro_aligned_metrics:", summary["macro_aligned_metrics"])
    print("mujoco_overrides:", summary["mujoco_overrides"])
    print(
        "scores:",
        {
            "macro_forward_lateral": summary["score_macro_forward_lateral"],
            "micro_jointvel_action": summary["score_micro_jointvel_action"],
            "total": summary["score_total"],
        },
    )
    top_vel = sorted(
        summary["micro_joint_rmse"].items(),
        key=lambda kv: kv[1]["joint_vel_rel_rmse"],
        reverse=True,
    )[:4]
    print("micro_top4_joint_vel_rmse:", top_vel)
    print(f"wrote_csv={out_csv}")
    print(f"wrote_npz={out_npz}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
