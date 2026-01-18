
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Power-Off Standing Test.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim import SimulationContext, SimulationCfg, GroundPlaneCfg, DistantLightCfg
from isaaclab.assets import Articulation
from isaaclab_assets.robots import QWE_DOG_CFG

def main():
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    
    cfg_ground = GroundPlaneCfg()
    cfg_ground.func("/World/ground", cfg_ground)
    cfg_light = DistantLightCfg(intensity=3000.0)
    cfg_light.func("/World/light", cfg_light)

    print("[INFO] Spawning QWE Dog for STANDING test (Stiffness=0)...")
    dog_cfg = QWE_DOG_CFG.copy()
    dog_cfg.prim_path = "/World/QWE_Dog"
    # 落地高度，让脚刚好接触地面，避免摔得太重
    dog_cfg.init_state.pos = (0.0, 0.0, 0.35) 
    
    # 确保基座是自由的（受重力）
    if hasattr(dog_cfg.spawn, "articulation_props"):
        dog_cfg.spawn.articulation_props.fix_root_link = False

    # 断电：刚度阻尼全为0
    for _, act in dog_cfg.actuators.items():
        act.stiffness = 0.0
        act.damping = 0.0

    dog = Articulation(dog_cfg)
    sim.reset()

    print("[INFO] Simulation running. If the dog stays standing, Friction > Body Weight Torque.")

    while simulation_app.is_running():
        sim.step()
        
        if sim.current_time % 1.0 < 0.02:
             # 监控基座高度
             root_height = dog.data.root_pos_w[0, 2]
             print(f"[Monitor] Body Height: {root_height:.3f} m")

if __name__ == "__main__":
    main()
    simulation_app.close()
