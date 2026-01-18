
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Calibrate Wheel Position.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation
from isaaclab_assets.robots import QWE_DOG_CFG
import omni.usd
from pxr import UsdGeom

def get_mesh_center(stage, link_path):
    # 尝试查找 link 下的 visual mesh
    prim = stage.GetPrimAtPath(link_path)
    if not prim.IsValid():
        return None
    
    # 计算 Bounding Box
    bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
    bound = bbox_cache.ComputeWorldBound(prim)
    range_ = bound.GetRange()
    center = (range_.GetMin() + range_.GetMax()) / 2.0
    return np.array([center[0], center[1], center[2]])

def main():
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    
    dog_cfg = QWE_DOG_CFG.copy()
    dog_cfg.prim_path = "/World/QWE_Dog"
    dog_cfg.init_state.pos = (0.0, 0.0, 1.0)
    
    dog = Articulation(dog_cfg)
    sim.reset()
    for _ in range(20): sim.step()

    stage = omni.usd.get_context().get_stage()
    
    # 我们只看左前腿 (qian_zuo)
    # 结构: Base -> Big -> Small -> Last
    
    small_path = "/World/QWE_Dog/small_qian_zuo"
    last_path = "/World/QWE_Dog/last_qian_zuo"
    
    # 获取 small 和 last 的 Mesh 中心（世界坐标）
    pos_small = get_mesh_center(stage, small_path)
    pos_last = get_mesh_center(stage, last_path)
    
    print(f"Small Mesh Center (World): {pos_small}")
    print(f"Last Mesh Center (World):  {pos_last}")
    
    # 获取 small link 的 Joint Frame 位置（世界坐标）
    small_body_idx = dog.find_bodies("small_qian_zuo")[0][0]
    pos_small_joint = dog.data.body_pos_w[0, small_body_idx].cpu().numpy()
    
    print(f"Small Joint Frame (World): {pos_small_joint}")
    
    # 理论上，Last Joint 的 Origin 应该是 (Last Mesh Center - Small Joint Frame) 在 Small Frame 下的投影
    # 但我们简单点，直接看 Last Mesh 相对于 Small Joint Frame 偏了多少
    
    diff = pos_last - pos_small_joint
    print(f"Difference (World Frame): {diff}")
    
    # 由于现在姿态是正的 (World Frame ~= Body Frame)，这个 Diff 大概就是我们要填的值
    print(f"Suggested Origin XYZ: [{diff[0]:.4f}, {diff[1]:.4f}, {diff[2]:.4f}]")

if __name__ == "__main__":
    main()
    simulation_app.close()
