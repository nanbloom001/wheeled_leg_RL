
import os
from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.articulations import Articulation

def main():
    usd_path = os.path.abspath("source/isaaclab_assets/data/Robots/User/WAVEGO.usd")
    print(f"Loading USD from: {usd_path}")
    
    prim_utils.create_prim("/World/Robot", usd_path=usd_path)
    
    # Wait for a frame to let physics settle/load
    import omni.kit.app
    omni.kit.app.get_app().update()
    
    robot = Articulation("/World/Robot")
    robot.initialize()
    
    print("-" * 40)
    print(f"Joint Names: {robot.dof_names}")
    print(f"Number of Joints: {robot.num_dof}")
    print("-" * 40)
    
    simulation_app.close()

if __name__ == "__main__":
    main()
