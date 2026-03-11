#!/usr/bin/env python3
"""
MuJoCo 场景查看器 + 策略控制（模拟 sim2sim_test.py 的功能）
"""

import os
import sys
from pathlib import Path
import argparse
import time
import numpy as np

import mujoco
from mujoco import viewer
import torch
import yaml


def load_policy_config(env_yaml_path: str, io_descriptor_path: str):
    """加载环境参数和 I/O 配置"""
    
    # 加载环境 YAML
    with open(env_yaml_path) as f:
        env_config = yaml.safe_load(f)
    
    # 加载 I/O descriptor（如果需要）
    io_config = {}
    if os.path.exists(io_descriptor_path):
        with open(io_descriptor_path) as f:
            io_config = yaml.safe_load(f)
    
    return env_config, io_config


def load_policy(checkpoint_path: str, device: str = "cpu"):
    """加载 PyTorch 策略网络"""
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    
    # 假设网络架构（来自 sim2sim_test.py）
    num_obs = 48
    num_actions = 12
    
    net = torch.nn.Sequential(
        torch.nn.Linear(num_obs, 512),
        torch.nn.ELU(),
        torch.nn.Linear(512, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, num_actions),
    )
    
    # 提取 actor 权重
    actor_state = {}
    for key, value in state_dict.items():
        if key.startswith("actor."):
            actor_state[key.replace("actor.", "")] = value
    
    net.load_state_dict(actor_state)
    net.eval()
    
    return net, num_obs, num_actions


def build_observation(data, model, gravity, command, default_q, last_action):
    """构建策略观测（简化版）"""
    
    # body[0] = world, body[1] = base（根据 MuJoCo 约定）
    base_pos = data.xpos[1]
    base_quat = data.xquat[1]   # [w, x, y, z]
    base_lin_vel = data.cvel[1, 3:6]   # linear velocity
    base_ang_vel = data.cvel[1, 0:3]   # angular velocity
    
    # 关节信息
    joint_pos = data.qpos[7:]  # 跳过 base 的 7 个 DOF
    joint_vel = data.qvel[6:]  # 跳过 base 的 6 个 DOF
    
    # 投影重力：将世界坐标系重力向量旋转到机体坐标系
    grav_w = np.array([0.0, 0.0, -1.0])  # 单位重力方向
    q = base_quat / (np.linalg.norm(base_quat) + 1e-8)  # normalize wxyz
    q_vec = q[1:]   # xyz part
    q_w = q[0]      # w part
    t = 2.0 * np.cross(q_vec, grav_w)
    proj_grav = grav_w + q_w * t + np.cross(q_vec, t)

    obs = np.concatenate([
        base_lin_vel,
        base_ang_vel,
        proj_grav,            # projected gravity in body frame
        command,              # velocity command
        joint_pos - default_q,
        joint_vel,
        last_action.astype(np.float64),   # previous action
    ])
    
    return obs.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo 场景查看器 + 可选的策略控制"
    )
    parser.add_argument("--scene", type=str, default="WAVEGO_mujoco/scene.xml",
                        help="MuJoCo scene XML 路径")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="策略权重文件 (.pt)")
    parser.add_argument("--env-yaml", type=str, default=None,
                        help="环境配置文件 (env.yaml)")
    parser.add_argument("--io-descriptor", type=str, default=None,
                        help="I/O descriptor 文件")
    parser.add_argument("--cmd-x", type=float, default=0.0,
                        help="命令：线速度 x (m/s)")
    parser.add_argument("--cmd-y", type=float, default=0.0,
                        help="命令：线速度 y (m/s)")
    parser.add_argument("--cmd-wz", type=float, default=0.0,
                        help="命令：角速度 z (rad/s)")
    parser.add_argument("--steps", type=int, default=1000,
                        help="仿真步数")
    parser.add_argument("--real-time", action="store_true",
                        help="按实时速度运行")
    parser.add_argument("--print-every", type=int, default=100,
                        help="每 N 步打印一次状态")
    parser.add_argument("--no-visualize", action="store_true",
                        help="禁用可视化窗口（无 display 时自动启用）")
    
    args = parser.parse_args()
    
    # 设置工作目录
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    print(f"工作目录: {os.getcwd()}")
    print(f"加载场景: {args.scene}")
    
    if not os.path.exists(args.scene):
        raise FileNotFoundError(f"Scene file not found: {args.scene}")
    
    # 加载 MuJoCo 模型
    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    
    print(f"✓ 模型加载成功: {model.nq} DOFs, {model.nbody} bodies")
    
    # 加载策略（如果提供）
    policy = None
    if args.checkpoint and args.checkpoint.strip():
        print(f"加载策略: {args.checkpoint}")
        policy, num_obs, num_actions = load_policy(args.checkpoint)
        print(f"✓ 策略加载成功: {num_obs} 观测, {num_actions} 动作")
    
    # 加载环境配置（如果提供）— 仅用于参考，脚本本身不依赖它
    env_config = None
    if args.env_yaml and args.env_yaml.strip():
        try:
            # 注意: env.yaml 可能包含 Python 特定对象，无法通过标准 YAML 加载
            # 这里仅做日志，不影响仿真逻辑
            with open(args.env_yaml) as f:
                # 尝试加载，失败则忽略（脚本本身只需 MuJoCo XML 和策略权重）
                try:
                    env_config = yaml.load(f, Loader=yaml.FullLoader)
                except yaml.YAMLError:
                    pass
            print(f"✓ 环境配置文件已读取（可选）")
        except Exception as e:
            print(f"⚠ 环境配置加载失败（可选，脚本继续运行）: {e}")
    
    # 命令向量
    command = np.array([args.cmd_x, args.cmd_y, args.cmd_wz], dtype=np.float32)
    print(f"命令: vx={args.cmd_x:.2f}, vy={args.cmd_y:.2f}, wz={args.cmd_wz:.2f}")
    
    # 默认关节位置
    default_q = np.array([0.1, -0.65, 0.6, -0.1, 0.65, -0.6, 
                          -0.1, -0.65, 0.6, 0.1, 0.65, -0.6], dtype=np.float32)
    
    print(f"仿真 {args.steps} 步...")
    
    # 初始化机器人到站立姿态
    # 根据 scene.xml 的 keyframe "home"
    home_qpos = np.array([0, 0, 0.25, 1, 0, 0, 0,  # 基础位置和四元数
                          0.1, -0.65, 0.6,           # FL 腿
                          -0.1, 0.65, -0.6,          # FR 腿
                          -0.1, -0.65, 0.6,          # RL 腿
                          0.1, 0.65, -0.6], dtype=np.float32)  # RR 腿
    
    data.qpos[:] = home_qpos
    # 同步 ctrl 到站立姿态（PD 控制器目标）
    data.ctrl[:] = default_q

    mujoco.mj_forward(model, data)  # 刷新 xpos/xquat/cvel
    print(f"初始化完成: base_z={data.xpos[1, 2]:.3f}, 关节数={model.njnt}")
    
    step_count = 0
    sim_time = 0.0
    last_action = np.zeros(12, dtype=np.float32)
    
    # 检测是否有可用的 display
    has_display = (
        not args.no_visualize
        and os.environ.get("DISPLAY") is not None
        or os.environ.get("WAYLAND_DISPLAY") is not None
    )

    if has_display:
        print("检测到 display，启动可视化查看器...")
    else:
        print("未检测到 display，以纯仿真（无 GUI）模式运行...")

    def _run_steps(get_running):
        nonlocal step_count, sim_time, last_action
        while get_running() and step_count < args.steps:
            obs = build_observation(data, model, np.array([0, 0, -9.81]),
                                   command, default_q, last_action)
            if policy is not None:
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                    action = policy(obs_tensor).squeeze(0).cpu().numpy()
                    action = np.clip(action, -1, 1)
            else:
                action = np.zeros(12, dtype=np.float32)

            data.ctrl[:] = action
            mujoco.mj_step(model, data)
            sim_time += model.opt.timestep

            if step_count % args.print_every == 0:
                base_pos = data.xpos[1]
                base_vel = data.cvel[1, 3:6]
                print(f"[{step_count:5d}] t={sim_time:.3f}s  "
                      f"pos=[{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]  "
                      f"vel=[{base_vel[0]:.3f}, {base_vel[1]:.3f}, {base_vel[2]:.3f}]")

            step_count += 1
            last_action = action

            if args.real_time:
                time.sleep(model.opt.timestep)

    if has_display:
        try:
            with viewer.launch_passive(model, data) as v:
                _run_steps(v.is_running)
        except Exception as e:
            print(f"⚠ 查看器启动失败（{e}），切换到无 GUI 模式...")
            _run_steps(lambda: True)
    else:
        _run_steps(lambda: True)
    
    print(f"\n✓ 仿真完成: {step_count} 步, 仿真时间 {sim_time:.2f}s")


if __name__ == "__main__":
    main()
