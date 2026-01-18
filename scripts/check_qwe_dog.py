# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
脚本功能：加载 QWE_DOG 模型进行外观检查。
特点：
1. 使用最新的 Isaac Lab Spawner API 添加地面和灯光。
2. 演示如何在场景中加载机器人。
"""

import argparse
from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="Check QWE Dog Model.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Isaac Sim 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_assets.robots import QWE_DOG_CFG

def main():
    # 1. 配置仿真上下文
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # 2. 设置场景内容
    # 使用 Spawner API 添加地面
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/ground", cfg_ground)
    
    # 使用 Spawner API 添加远光灯
    cfg_light = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/light", cfg_light)

    # 3. 配置并生成机器狗
    print("[INFO] Spawning QWE Dog...")
    dog_cfg = QWE_DOG_CFG.copy()
    dog_cfg.prim_path = "/World/QWE_Dog"
    
    # 将初始位置设高一点
    dog_cfg.init_state.pos = (0.0, 0.0, 0.6)
    
    # 创建 Articulation 对象
    dog = Articulation(dog_cfg)

    # 4. 播放仿真前必须重置
    sim.reset()
    print("[INFO] Simulation running. Use mouse to rotate view in Isaac Sim.")

    while simulation_app.is_running():
        # 步进仿真
        sim.step()

if __name__ == "__main__":
    main()