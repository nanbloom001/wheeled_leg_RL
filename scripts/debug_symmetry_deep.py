
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deep Symmetry Debugger.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation
from isaaclab_assets.robots import QWE_DOG_CFG
import omni.usd
from pxr import Usd, UsdGeom, Gf

def get_prim_global_transform(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None, None
    xform = UsdGeom.Xformable(prim)
    time = Usd.TimeCode.Default()
    world_transform = xform.ComputeLocalToWorldTransform(time)
    translation = world_transform.ExtractTranslation()
    rotation = world_transform.ExtractRotationQuat()
    return translation, rotation

def main():
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    
    dog_cfg = QWE_DOG_CFG.copy()
    dog_cfg.prim_path = "/World/QWE_Dog"
    dog_cfg.init_state.pos = (0.0, 0.0, 1.0)
    dog_cfg.init_state.rot = (0.7071, 0.0, 0.7071, 0.0) # 保持修正后的姿态
    
    dog = Articulation(dog_cfg)
    sim.reset()
    
    print("[INFO] Simulating to settle...")
    for _ in range(20): sim.step()

    stage = omni.usd.get_context().get_stage()
    
    # 定义对比组：左 vs 右
    pairs = [
        ("small_qian_zuo", "small_qian_you"),
        ("small_hou_zuo", "small_hou_you")
    ]

    print("\n" + "="*100)
    print(f"{ 'Metric':<20} | { 'Left Value':<35} | { 'Right Value':<35} | {'Diff'}")
    print("-" * 100)

    for left_name, right_name in pairs:
        print(f"--- Comparing {left_name} vs {right_name} ---")
        
        # 1. Joint Position (Root Frame)
        # 获取 Body 索引
        l_idx = dog.find_bodies(left_name)[0][0]
        r_idx = dog.find_bodies(right_name)[0][0]
        
        l_pos = dog.data.body_pos_w[0, l_idx].cpu().numpy()
        r_pos = dog.data.body_pos_w[0, r_idx].cpu().numpy()
        
        # 将世界坐标转换回 Base Link 坐标系，排除基座位置干扰
        # 但既然基座在 (0,0,1)，我们直接看相对值
        # 左右对称意味着：X相等，Z相等，Y相反
        
        def vec_str(v): return f"[{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}]"
        
        print(f"{ 'Joint Pos (World)':<20} | {vec_str(l_pos):<35} | {vec_str(r_pos):<35} | Y_sum={l_pos[1]+r_pos[1]:.3f}")
        
        # 2. Rotation Matrix (Z-axis direction)
        l_quat = dog.data.body_quat_w[0, l_idx].cpu().numpy() # w, x, y, z
        r_quat = dog.data.body_quat_w[0, r_idx].cpu().numpy()
        
        # 简单的四元数对比很难，我们直接对比 Z 轴向量（长轴）
        # ... (此前已做过，确认是对称的)
        
        # 3. Visual Mesh Bounds (关键！)
        # 我们需要找到 Link 下面的 Visual Prim
        # 假设路径规则：/World/QWE_Dog/{link_name}/visuals
        # 注意：Isaac Lab 生成的路径可能包含 visuals/mesh_...
        
        def get_visual_center(link_name):
            base_path = f"/World/QWE_Dog/{link_name}"
            # 遍历寻找 visual mesh
            prim = stage.GetPrimAtPath(base_path)
            # 这是一个极其简化的查找，实际层级可能更深
            # 我们假设视觉中心就是 Link 原点，除非我们能遍历 vertices
            # 但我们可以通过 USD API 获取 Bounding Box
            
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            bound = bbox_cache.ComputeWorldBound(prim)
            range_ = bound.GetRange()
            center = (range_.GetMin() + range_.GetMax()) / 2.0
            return np.array([center[0], center[1], center[2]])

        l_mesh_center = get_visual_center(left_name)
        r_mesh_center = get_visual_center(right_name)
        
        print(f"{ 'Mesh Center':<20} | {vec_str(l_mesh_center):<35} | {vec_str(r_mesh_center):<35} | Y_sum={l_mesh_center[1]+r_mesh_center[1]:.3f}")
        
        # 4. 前后偏差分析
        x_diff = abs(l_mesh_center[0] - r_mesh_center[0])
        print(f"{ 'X-Alignment (F/B)':<20} | Diff: {x_diff:.4f} (Should be 0)")
        
    print("="*100)

if __name__ == "__main__":
    main()
    simulation_app.close()
