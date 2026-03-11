# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
脚本功能：WAVEGO 关节调试器。
特点：
1. 加载 WAVEGO 机器人。
2. 使用滑块实时调整关节位置。
3. 自动打印可复制到配置文件的格式。
"""

import argparse
from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="Tune WAVEGO Robot Joints.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Isaac Sim 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import omni.ui as ui
import re
from isaaclab.sim import SimulationContext, SimulationCfg, GroundPlaneCfg, DomeLightCfg
from isaaclab.assets import Articulation
from isaaclab_assets.robots.wavego import WAVEGO_CFG

class JointTunerWindow(ui.Window):
    def __init__(self, robot: Articulation, joint_names: list[str], default_pos: dict):
        super().__init__("WAVEGO Joint Tuner", width=400, height=600)
        self.robot = robot
        self.joint_names = joint_names
        self.sliders = {}
        
        with self.frame:
            with ui.ScrollingFrame(
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):
                with ui.VStack(spacing=10):
                    ui.Label("Adjust Joint Targets (Radians)", height=30, alignment=ui.Alignment.CENTER)
                    
                    # 按照 FL, FR, RL, RR 逻辑分组显示
                    groups = ["FL", "FR", "RL", "RR"]
                    
                    for group in groups:
                        ui.Label(f"--- {group} Leg ---", height=20, style={"color": 0xFF00FF00})
                        for name in joint_names:
                            if name.startswith(group):
                                self._add_slider(name, default_pos)
                    
                    # 显示剩余关节
                    ui.Label("--- Other Joints ---", height=20, style={"color": 0xFF00AAAA})
                    for name in joint_names:
                        if not any(name.upper().startswith(group) for group in groups):
                            if name not in self.sliders:
                                self._add_slider(name, default_pos)

                    ui.Spacer(height=20)
                    ui.Button("Print Config Format", height=40, clicked_fn=self._print_config)

    def _add_slider(self, name, default_pos):
        # 获取默认值
        init_val = 0.0
        if default_pos:
            for pattern, value in default_pos.items():
                if re.match(pattern, name):
                    init_val = value
                    break
        
        with ui.HStack(height=30):
            ui.Label(name, width=150, alignment=ui.Alignment.LEFT_CENTER)
            slider = ui.FloatSlider(min=-3.14, max=3.14, step=0.01)
            slider.model.set_value(float(init_val))
            self.sliders[name] = slider
            
            label = ui.StringField(width=50)
            label.model.set_value(f"{init_val:.2f}")
            def update_label(m, l=label):
                l.model.set_value(f"{m.get_value_as_float():.2f}")
            slider.model.add_value_changed_fn(update_label)

    def get_joint_targets(self):
        targets = []
        for name in self.joint_names:
            if name in self.sliders:
                targets.append(self.sliders[name].model.get_value_as_float())
            else:
                targets.append(0.0)
        return torch.tensor([targets], device=self.robot.device)

    def _print_config(self):
        print("\n" + "="*30)
        print("Current Pose (Copy to wavego.py):")
        print("joint_pos={")
        for name in self.joint_names:
            val = self.sliders[name].model.get_value_as_float()
            print(f"    '{name}': {val:.3f},")
        print("}")
        print("="*30 + "\n")

def main():
    # 1. 配置仿真
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.6, 0.6, 0.6], target=[0.0, 0.0, 0.0])
    
    cfg_ground = GroundPlaneCfg()
    cfg_ground.func("/World/ground", cfg_ground)
    cfg_light = DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg_light.func("/World/light", cfg_light)

    # 2. 加载机器人
    print("[INFO] Spawning WAVEGO robot...")
    robot_cfg = WAVEGO_CFG.copy()
    robot_cfg.prim_path = "/World/Robot"
    robot_cfg.init_state.pos = (0.0, 0.0, 0.2) # WAVEGO 比较小，降低高度
    
    # 强制固定根部方便调试
    if hasattr(robot_cfg.spawn, "articulation_props"):
        robot_cfg.spawn.articulation_props.fix_root_link = True
    
    robot = Articulation(robot_cfg)

    sim.reset()
    
    # 初始化 UI
    window = JointTunerWindow(robot, robot.joint_names, WAVEGO_CFG.init_state.joint_pos)
    
    # 当前目标 buffer (用于平滑移动)
    current_targets = robot.data.joint_pos.clone()

    while simulation_app.is_running():
        # 获取滑块值
        desired_targets = window.get_joint_targets()
        
        # Slew rate limiter (平滑移动)
        max_step = 0.05
        diff = desired_targets - current_targets
        diff = torch.clamp(diff, -max_step, max_step)
        current_targets += diff
        
        # 应用并步进
        robot.set_joint_position_target(current_targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt=sim.get_physics_dt())

if __name__ == "__main__":
    main()
    simulation_app.close()
