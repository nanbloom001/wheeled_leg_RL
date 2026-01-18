import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug Link Orientation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation
from isaaclab_assets.robots import QWE_DOG_CFG
from scipy.spatial.transform import Rotation
import numpy as np
import torch

def main():
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    
    dog_cfg = QWE_DOG_CFG.copy()
    dog_cfg.prim_path = "/World/QWE_Dog"
    dog_cfg.init_state.pos = (0.0, 0.0, 1.0)
    
    dog = Articulation(dog_cfg)
    sim.reset()

    print("[INFO] Simulating...")
    for _ in range(20): sim.step()

    targets = [
        "small_qian_zuo", "small_qian_you", 
        "small_hou_zuo", "small_hou_you"
    ]

    header = f"{ 'Link Name':<20} | {'Local X (World)':<25} | {'Local Y':<25} | {'Local Z':<25}"
    print("\n" + "="*len(header))
    print(header)
    print("-" * len(header))

    for body_name in targets:
        try:
            body_idx = dog.find_bodies(body_name)[0][0]
            quat_w = dog.data.body_quat_w[0, body_idx].cpu().numpy()
            # quat is (w, x, y, z) -> scipy needs (x, y, z, w)
            r = Rotation.from_quat([quat_w[1], quat_w[2], quat_w[3], quat_w[0]])
            mat = r.as_matrix() 
            
            x_axis = mat[:, 0]
            y_axis = mat[:, 1]
            z_axis = mat[:, 2]
            
            def fmt(v): return f"[{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}]"
            print(f"{body_name:<20} | {fmt(x_axis):<25} | {fmt(y_axis):<25} | {fmt(z_axis):<25}")
            
        except Exception as e:
            print(f"Err {body_name}: {e}")
    print("="*len(header))

if __name__ == "__main__":
    main()
    simulation_app.close()