
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Power-Off Friction Test.")
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

    # 复制配置并“断电”
    print("[INFO] Spawning QWE Dog in POWER OFF mode (Stiffness=0)...")
    dog_cfg = QWE_DOG_CFG.copy()
    dog_cfg.prim_path = "/World/QWE_Dog"
    dog_cfg.init_state.pos = (0.0, 0.0, 1.0) 
    
    # 强制固定基座
    if hasattr(dog_cfg.spawn, "articulation_props"):
        dog_cfg.spawn.articulation_props.fix_root_link = True
    
    # 彻底断电：刚度阻尼全为0，只保留 URDF 里的物理摩擦
    for _, act in dog_cfg.actuators.items():
        act.stiffness = 0.0
        act.damping = 0.0 # 注意：这是 PD 阻尼，设为 0。URDF 里的 dynamics damping 依然存在。

    dog = Articulation(dog_cfg)
    sim.reset()

    print("[INFO] Simulation running. Watch if legs fall under gravity.")
    print("[INFO] Gravity Torque approx 0.25 Nm. Friction set to 0.5 Nm.")
    print("[INFO] If legs stay still, Friction is working. If they fall, Friction is invalid.")

    while simulation_app.is_running():
        # 不发送任何指令，完全被动
        sim.step()
        
        if sim.current_time % 1.0 < 0.02:
             # 监控关节速度
             vel = dog.data.joint_vel[0, 0]
             pos = dog.data.joint_pos[0, 0]
             effort = dog.data.applied_torque[0, 0]
             print(f"[Monitor] Joint[0] Pos: {pos:.3f} | Vel: {vel:.3f} | Effort: {effort:.3f}")

if __name__ == "__main__":
    main()
    simulation_app.close()
