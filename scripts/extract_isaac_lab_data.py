#!/usr/bin/env python3
"""
Step 1: 从 Isaac Lab 提取真实的关节顺序、站立姿态和策略输出范围
这是 Sim2Sim 迁移的第一步 - 必须从源环境提取真实数据
"""

import torch
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="无头模式")
    args = parser.parse_args()
    
    # 延迟导入（需要在 isaaclab 环境中）
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher({"headless": args.headless})
    simulation_app = app_launcher.app
    
    import isaaclab.sim as sim_utils
    from isaaclab_tasks.manager_based.locomotion.velocity.config.wavego.flat_env_cfg import WavegoFlatEnvCfg
    
    print("=" * 80)
    print("Isaac Lab 真实数据提取")
    print("=" * 80)
    
    # 创建环境
    print("\n[1] 创建环境...")
    env_cfg = WavegoFlatEnvCfg()
    env_cfg.scene.num_envs = 1  # 只需一个环境
    env_cfg.sim.device = "cpu"  # 使用 CPU 便于调试
    
    import gymnasium as gym
    env = gym.make("Isaac-Velocity-Flat-WAVEGO-v0", cfg=env_cfg)
    
    # 提取关节信息
    robot = env.unwrapped.scene["robot"]
    
    print("\n[2] 关节顺序 (来自 robot.data.joint_names):")
    print("=" * 80)
    joint_names = robot.data.joint_names
    for i, name in enumerate(joint_names):
        print(f"  Joint {i:2d}: {name}")
    
    print("\n[3] 站立姿态 (来自 robot.data.default_joint_pos):")
    print("=" * 80)
    default_pos = robot.data.default_joint_pos[0].cpu().numpy()
    for i, name in enumerate(joint_names):
        print(f"  {name:20s}: {default_pos[i]:+.6f} rad  ({default_pos[i]*180/np.pi:+.2f}°)")
    
    print("\n[4] 动作空间索引 (来自 env.action_manager):")
    print("=" * 80)
    action_term = env.unwrapped.action_manager._terms["joint_pos"]
    print(f"  joint_names 正则: {action_term.cfg.joint_names}")
    print(f"  action_scale: {action_term.cfg.scale}")
    print(f"  use_default_offset: {action_term.cfg.use_default_offset}")
    print(f"  实际控制的关节索引: {action_term._joint_ids}")
    print(f"  实际控制的关节名称:")
    for idx in action_term._joint_ids:
        print(f"    action[{list(action_term._joint_ids).index(idx)}] -> {joint_names[idx]}")
    
    # 运行几步，观察策略输出
    print("\n[5] 加载策略并测试...")
    print("=" * 80)
    
    # 加载权重
    policy_path = "logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/model_2999.pt"
    checkpoint = torch.load(policy_path, map_location="cpu")
    state_dict = checkpoint['model_state_dict']
    
    # 构建策略网络
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
    
    obs_mean = state_dict['actor_obs_normalizer._mean'].cpu()
    obs_std = state_dict['actor_obs_normalizer._std'].cpu()
    
    # 重置环境
    obs, _ = env.reset()
    
    print("\n[6] 运行 100 步，统计策略输出范围:")
    print("=" * 80)
    
    action_log = []
    target_log = []
    
    for step in range(100):
        # 策略推理
        obs_tensor = obs["policy"]
        obs_norm = (obs_tensor - obs_mean) / (obs_std + 1e-8)
        obs_norm = torch.clamp(obs_norm, -5, 5)
        
        with torch.no_grad():
            actions = policy(obs_norm).cpu().numpy()
        
        action_log.append(actions[0])
        
        # 计算目标位置
        target_pos = actions[0] * action_term.cfg.scale + default_pos
        target_log.append(target_pos)
        
        # 执行动作
        obs, _, _, _, _ = env.step(torch.from_numpy(actions).float())
    
    action_log = np.array(action_log)
    target_log = np.array(target_log)
    
    print("\n策略原始输出 (actions) 统计:")
    print(f"  形状: {action_log.shape}")
    print(f"  范围: [{action_log.min():.3f}, {action_log.max():.3f}]")
    print(f"  均值: {action_log.mean(axis=0)}")
    print(f"  标准差: {action_log.std(axis=0)}")
    
    print("\n目标位置 (actions * 0.25 + default) 统计:")
    print(f"  范围: [{target_log.min():.3f}, {target_log.max():.3f}]")
    for i, name in enumerate(joint_names):
        print(f"  {name:20s}: [{target_log[:, i].min():+.3f}, {target_log[:, i].max():+.3f}] rad")
    
    # 保存到文件
    output = {
        "joint_names": joint_names,
        "default_joint_pos": default_pos.tolist(),
        "action_joint_ids": action_term._joint_ids.tolist(),
        "action_scale": action_term.cfg.scale,
        "action_range": [float(action_log.min()), float(action_log.max())],
        "target_range": [float(target_log.min()), float(target_log.max())],
        "obs_mean": obs_mean[0].cpu().numpy().tolist(),
        "obs_std": obs_std[0].cpu().numpy().tolist(),
    }
    
    import json
    with open("isaac_lab_ground_truth.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✅ 数据已保存到: isaac_lab_ground_truth.json")
    print("=" * 80)
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
