# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
脚本功能：QWE_DOG 关节调试器。
特点：
1. 加载机器狗。
2. 使用配置文件中的 init_state 作为滑块的初始位置。
3. 方便验证配置文件的修正效果。
"""

import argparse
from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="Tune QWE Dog Joints.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Isaac Sim 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import omni.ui as ui
import re # 导入正则模块
from isaaclab.sim import SimulationContext, SimulationCfg, GroundPlaneCfg, DistantLightCfg
from isaaclab.assets import Articulation
from isaaclab_assets.robots import QWE_DOG_CFG
import isaaclab.sim as sim_utils

class JointTunerWindow(ui.Window):
    def __init__(self, robot: Articulation, joint_names: list[str], default_pos: dict):
        super().__init__("Joint Tuner", width=400, height=600)
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
                    
                    for name in joint_names:
                        # 获取默认值 - 支持正则匹配
                        init_val = 0.0
                        for pattern, value in default_pos.items():
                            if re.match(pattern, name):
                                init_val = value
                                break
                        
                        with ui.HStack(height=30):
                            ui.Label(name, width=150, alignment=ui.Alignment.LEFT_CENTER)
                            slider = ui.FloatSlider(min=-6.28, max=6.28) # 扩大范围到 2PI
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

def main():
    # 1. 配置仿真
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    cfg_ground = GroundPlaneCfg()
    cfg_ground.func("/World/ground", cfg_ground)
    cfg_light = DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/light", cfg_light)

    # 2. 生成机器人
    print("[INFO] Spawning QWE Dog for tuning...")
    dog_cfg = QWE_DOG_CFG.copy()
    dog_cfg.prim_path = "/World/QWE_Dog"
    
    # 抬高并固定
    dog_cfg.init_state.pos = (0.0, 0.0, 1.0)
    if hasattr(dog_cfg.spawn, "articulation_props"):
        dog_cfg.spawn.articulation_props.fix_root_link = True

    # 调大刚度
    for actuator_name, actuator_cfg in dog_cfg.actuators.items():
        actuator_cfg.stiffness = 200.0
        actuator_cfg.damping = 5.0

    # --- 关键修改：不再强制归零，保留配置文件的初始值 ---
    # for j_name in dog_cfg.init_state.joint_pos:
    #     dog_cfg.init_state.joint_pos[j_name] = 0.0

    dog = Articulation(dog_cfg)

    sim.reset()
    
    # 将初始位置应用一次（因为我们可能处于 reset 状态）
    # dog.set_joint_position_target( ... ) # 其实 reset 时 Articulation 会自动应用 init_state
    
    joint_names = dog.joint_names
    # 传入默认配置用于 UI 初始化
    window = JointTunerWindow(dog, joint_names, QWE_DOG_CFG.init_state.joint_pos)
    
    # 初始化平滑控制变量
    current_targets = dog.data.joint_pos.clone()
    
    print("[INFO] Tuner Window Opened.")
    print("[INFO] Sliders initialized from config file. Smoothing enabled.")

    while simulation_app.is_running():
        if window:
            # 获取滑块的目标值
            desired_targets = window.get_joint_targets()
            
            # --- 平滑滤波逻辑 (Slew Rate Limiter) ---
            # 限制每帧最大变化量，模拟"慢速移动"
            # dt = sim.get_physics_dt() # 约 0.01s
            # max_speed = 2.0 rad/s -> max_step = 0.02 rad/frame
            max_step = 0.05 
            
            # 计算差值
            diff = desired_targets - current_targets
            # 限制差值幅度
            diff = torch.clamp(diff, -max_step, max_step)
            # 更新当前指令
            current_targets = current_targets + diff
            
            dog.set_joint_position_target(current_targets)
            dog.write_data_to_sim()
        
        sim.step()
        dog.update(dt=sim.get_physics_dt())

        # --- 辅助调试：每 60 帧打印一次状态 ---
        if sim.current_time % 1.0 < 0.02: # 约每秒打印一次
            print(f"\n[DEBUG] Simulation Time: {sim.current_time:.2f}s")
            print(f"{'Joint Name':<25} | {'Pos':<8} | {'Vel':<8} | {'Effort':<8}")
            print("-" * 55)
            
            # 一次性获取所有关节数据
            all_pos = dog.data.joint_pos[0]
            all_vel = dog.data.joint_vel[0]
            all_effort = dog.data.applied_torque[0]
            
            for i, name in enumerate(dog.joint_names):
                # 注意：dog.joint_names 的顺序通常与 buffer 索引一致，但在复杂 articulation 中最好通过 find_joints 确认
                # 这里假设顺序一致（对于标准 Articulation 是成立的）
                pos = all_pos[i]
                vel = all_vel[i]
                effort = all_effort[i]
                
                warn_mark = " (!)" if abs(effort) > 1.2 else ""
                print(f"{name:<25} | {pos: .3f}   | {vel: .3f}   | {effort: .3f}{warn_mark}")
            print("-" * 55)

if __name__ == "__main__":
    main()