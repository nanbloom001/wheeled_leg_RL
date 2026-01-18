from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation
from isaaclab_assets.robots import QWE_DOG_CFG

def main():
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    
    dog_cfg = QWE_DOG_CFG.copy()
    dog_cfg.prim_path = "/World/QWE_Dog"
    dog = Articulation(dog_cfg)
    
    print("[INFO] Resetting simulation...")
    sim.reset()
    print("[INFO] Success! Joint limits are unlocked.")

if __name__ == "__main__":
    main()
    simulation_app.close()
