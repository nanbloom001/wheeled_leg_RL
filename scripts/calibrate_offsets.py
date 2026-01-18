
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Auto Calibrate Leg Offsets.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation
from isaaclab_assets.robots import QWE_DOG_CFG
from scipy.spatial.transform import Rotation

def main():
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    
    # 悬空，初始角度全0
    dog_cfg = QWE_DOG_CFG.copy()
    dog_cfg.prim_path = "/World/QWE_Dog"
    dog_cfg.init_state.pos = (0.0, 0.0, 1.0)
    # 保持摩擦力，避免乱晃
    
    dog = Articulation(dog_cfg)
    sim.reset()

    print("[INFO] Simulation running. Analyzing link orientations...")

    # 等待稳定
    for _ in range(50):
        sim.step()

    # 获取所有 Link 的名称和索引
    # Link names usually match body names in URDF
    body_names = dog.body_names
    print(f"[INFO] Body names: {body_names}")

    # 定义我们要校准的目标 Link 及其对应的【驱动关节】
    # 逻辑：调节 Joint X，改变 Link X 的姿态
    targets = {
        # 大腿 Link -> 由 Big Joint 驱动
        "big_qian_zuo": "big_qian_zuo_joint",
        "big_qian_you": "big_qian_you_joint",
        "big_hou_zuo":  "big_hou_zuo_joint",
        "big_hou_you":  "big_hou_you_joint",
        
        # 小腿 Link -> 由 Small Joint 驱动
        "small_qian_zuo": "small_qian_zuo_joint",
        "small_qian_you": "small_qian_you_joint",
        "small_hou_zuo":  "small_hou_zuo_joint",
        "small_hou_you":  "small_hou_you_joint",
        
        # 轮子 Link -> 由 Last Joint 驱动 (通常不需要校准垂直，但看看也无妨)
        # "last_qian_zuo": "last_qian_zuo_last", 
    }

    # 获取当前关节角度
    joint_pos = dog.data.joint_pos[0] # Tensor
    joint_names = dog.joint_names
    
    print("\n" + "="*60)
    print(f"{ 'Link Name':<20} | { 'Current Angle':<10} | { 'Estimated Offset Needed':<20}")
    print("-" * 60)

    for body_name, joint_name in targets.items():
        try:
            body_idx = dog.find_bodies(body_name)[0][0]
            joint_idx = dog.find_joints(joint_name)[0][0]
            
            # 获取 Link 在世界系下的姿态 (quat: w, x, y, z)
            quat_w = dog.data.body_quat_w[0, body_idx].cpu().numpy()
            # SciPy 需要 (x, y, z, w)
            quat_scipy = [quat_w[1], quat_w[2], quat_w[3], quat_w[0]]
            
            # 转欧拉角 (假设 Z-Y-X 顺序)
            # 我们主要关心的是绕 X 轴的旋转 (Roll)，因为关节是绕 X 转的
            r, p, y = Rotation.from_quat(quat_scipy).as_euler('xyz', degrees=False)
            
            # 分析：
            # 如果是垂直向下，Roll 应该是 0 (或者 PI, 取决于初始朝向)
            # 这里的 r 就是偏差
            
            current_angle = float(joint_pos[joint_idx])
            
            # 偏差校正：
            # 如果当前 Roll 是 0.5，说明它歪了 0.5。我们需要把关节转 -0.5 才能回正吗？
            # 这取决于关节轴的方向。
            # 简单估算：Offset = -Roll (假设目标是 Roll=0)
            
            print(f"{body_name:<20} | {current_angle: .4f}     | {-r: .4f} (Target Roll=0)")
            
        except Exception as e:
            print(f"[ERROR] Finding {body_name}: {e}")

    print("="*60)
    print("[TIP] If 'Estimated Offset' is close to +/- 3.14, it means the link is inverted.")
    print("[TIP] Add 'Current Angle' + 'Offset' to get the zero-pose angle.")

if __name__ == "__main__":
    main()
    simulation_app.close()
