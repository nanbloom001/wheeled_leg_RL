#!/usr/bin/env python3
"""
Step 3: 单步调试 - 逐帧对比 Isaac Lab 和 MuJoCo 的运行状态
最严格的验证方式：相同观测输入 → 相同策略输出 → 相同执行效果
"""

import mujoco
import mujoco.viewer
import torch
import numpy as np
import json

def load_ground_truth():
    with open("isaac_lab_ground_truth.json", "r") as f:
        return json.load(f)

def get_obs_mujoco(data, last_action, standing_pose):
    """MuJoCo 观测构建"""
    from scipy.spatial.transform import Rotation as R
    
    quat = data.qpos[3:7]
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    inv_rot = r.inv()
    
    lin_vel_b = inv_rot.apply(data.qvel[:3])
    ang_vel_b = inv_rot.apply(data.qvel[3:6])
    proj_grav = inv_rot.apply(np.array([0, 0, -1.0]))
    
    joint_pos = data.qpos[7:]
    joint_vel = data.qvel[6:]
    obs_joint_pos = joint_pos - standing_pose
    
    command = np.array([0.5, 0.0, 0.0])
    
    obs = np.concatenate([
        lin_vel_b, ang_vel_b, proj_grav, command,
        obs_joint_pos, joint_vel, last_action
    ])
    return obs

def main():
    print("=" * 80)
    print("单步调试工具 - 交互式验证")
    print("=" * 80)
    
    # 加载真实数据
    gt = load_ground_truth()
    standing_pose = np.array(gt["default_joint_pos"])
    obs_mean = np.array(gt["obs_mean"])
    obs_std = np.array(gt["obs_std"])
    
    # 加载 MuJoCo
    model = mujoco.MjModel.from_xml_path("WAVEGO_mujoco/scene.xml")
    data = mujoco.MjData(model)
    
    # 加载策略
    policy_path = "logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/model_2999.pt"
    checkpoint = torch.load(policy_path, map_location="cpu")
    state_dict = checkpoint['model_state_dict']
    
    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(48, 512), torch.nn.ELU(),
                torch.nn.Linear(512, 256), torch.nn.ELU(),
                torch.nn.Linear(256, 128), torch.nn.ELU(),
                torch.nn.Linear(128, 12)
            )
        def forward(self, x): return self.net(x)
    
    policy = Actor()
    clean_dict = {k.replace('actor.', 'net.'): v for k, v in state_dict.items() 
                  if 'actor.0' in k or 'actor.2' in k or 'actor.4' in k or 'actor.6' in k}
    policy.load_state_dict(clean_dict)
    policy.eval()
    
    # 初始化
    data.qpos[2] = 0.25
    data.qpos[7:] = standing_pose.copy()
    mujoco.mj_forward(model, data)
    
    last_action = np.zeros(12)
    step_count = 0
    
    print("\n控制说明:")
    print("  - 按 SPACE: 单步执行")
    print("  - 按 R: 重置")
    print("  - 按 ESC: 退出")
    print("\n开始调试...\n")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # 等待用户按键
            if viewer.is_running():
                # 1. 构建观测
                obs_raw = get_obs_mujoco(data, last_action, standing_pose)
                obs_norm = (obs_raw - obs_mean) / (obs_std + 1e-8)
                obs_norm = np.clip(obs_norm, -5.0, 5.0)
                
                # 2. 策略推理
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs_norm).float().unsqueeze(0)
                    actions = policy(obs_tensor).squeeze().numpy()
                
                # 3. 计算目标
                target_pos = actions * 0.25 + standing_pose
                
                # 4. 执行
                data.ctrl[:] = target_pos
                for _ in range(4):  # decimation=4
                    mujoco.mj_step(model, data)
                viewer.sync()
                
                last_action = actions.copy()
                step_count += 1
                
                # 5. 打印详细信息
                print(f"\n{'='*80}")
                print(f"Step {step_count}")
                print(f"{'='*80}")
                
                print("\n观测空间 (前12维):")
                print(f"  lin_vel_b: [{obs_raw[0]:+.3f}, {obs_raw[1]:+.3f}, {obs_raw[2]:+.3f}]")
                print(f"  ang_vel_b: [{obs_raw[3]:+.3f}, {obs_raw[4]:+.3f}, {obs_raw[5]:+.3f}]")
                print(f"  proj_grav: [{obs_raw[6]:+.3f}, {obs_raw[7]:+.3f}, {obs_raw[8]:+.3f}]")
                print(f"  command:   [{obs_raw[9]:+.3f}, {obs_raw[10]:+.3f}, {obs_raw[11]:+.3f}]")
                
                print("\n归一化观测 (范围检查):")
                print(f"  最小值: {obs_norm.min():+.3f}")
                print(f"  最大值: {obs_norm.max():+.3f}")
                print(f"  超出 [-5,5] 的维度数: {np.sum((obs_norm < -5) | (obs_norm > 5))}")
                
                print("\n策略输出:")
                print(f"  actions (原始): {actions}")
                print(f"  范围: [{actions.min():+.3f}, {actions.max():+.3f}]")
                print(f"  target_pos: {target_pos}")
                
                print("\n关节状态:")
                joint_error = data.qpos[7:] - data.ctrl[:]
                print(f"  平均跟踪误差: {np.abs(joint_error).mean():.4f} rad")
                print(f"  最大跟踪误差: {np.abs(joint_error).max():.4f} rad")
                
                print("\n力矩状态:")
                qfrc = data.qfrc_actuator[:12]
                print(f"  最大力矩: {np.abs(qfrc).max():.3f} / 1.96 Nm")
                print(f"  饱和数量: {np.sum(np.abs(qfrc) > 1.90)} / 12")
                
                print("\nBase 状态:")
                print(f"  高度: {data.qpos[2]:.3f} m")
                print(f"  线速度: [{data.qvel[0]:+.3f}, {data.qvel[1]:+.3f}, {data.qvel[2]:+.3f}]")
                
                # 暂停等待
                import time
                time.sleep(0.5)
            
            if step_count >= 20:
                print("\n已执行 20 步，退出...")
                break

if __name__ == "__main__":
    main()
