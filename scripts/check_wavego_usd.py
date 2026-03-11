import os
from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.sim import SimulationContext, UsdFileCfg
from isaaclab.actuators import ImplicitActuatorCfg

def main():
    sim = SimulationContext()
    
    usd_path = os.path.abspath("source/isaaclab_assets/data/Robots/User/WAVEGO.usd")
    print(f"Loading USD from: {usd_path}")
    
    cfg = ArticulationCfg(
        spawn=UsdFileCfg(usd_path=usd_path),
        prim_path="/World/Robot",
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=100.0,
                damping=1.0,
            )
        }
    )
    
    robot = Articulation(cfg)
    # Note: spawning happens automatically if prim_path is set? 
    # Actually we usually call spawn or use the scene manager.
    # In Articulation class, if prim_path has regex, it's used during spawn.
    
    # We'll use the manual way for this simple test
    robot.spawn("/World/Robot")
    
    sim.reset()
    
    print("-" * 40)
    print(f"Joint Names: {robot.joint_names}")
    print(f"Number of Joints: {robot.num_joints}")
    print("-" * 40)
    
    simulation_app.close()

if __name__ == "__main__":
    main()