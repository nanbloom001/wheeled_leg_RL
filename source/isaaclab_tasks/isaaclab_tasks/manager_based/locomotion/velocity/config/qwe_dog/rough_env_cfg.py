# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.qwe_dog import QWE_DOG_CFG  # 导入你的机器人配置


@configclass
class QweDogRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # --- PhysX GPU 内存优化 (防止崩溃) ---
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**23
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_max_soft_body_contacts = 2**23
        self.sim.physx.gpu_max_particle_contacts = 2**23
        self.sim.physx.gpu_heap_capacity = 2**26 # 64MB -> 增大
        self.sim.physx.gpu_temp_buffer_capacity = 2**26
        self.sim.physx.gpu_resource_part_data_capacity = 2**20
        # 增加碰撞栈大小
        self.sim.physx.gpu_collision_stack_size = 2**28 # ~256MB

        # 1. 替换机器人
        self.scene.robot = QWE_DOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # 2. 高度扫描仪 (用于感知地形)
        # 你的机器人基座叫 "base_link"
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        
        # 3. 地形调整 (因为机器人较小)
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # 4. 动作缩放 (Action Scale)
        # 这里的 scale 意味着：实际目标位置 = current_pos + action * scale * dt (如果是相对控制)
        # 或者：实际目标位置 = default_pos + action * scale (如果是绝对位置控制)
        # LocomotionVelocityRoughEnvCfg 默认通常是绝对位置控制 (JointPositionAction)
        # MG996R 速度较慢 (6.3 rad/s)，设为 0.25 意味着 policy 输出 1.0 时，关节偏离 0.25 rad (约14度)
        self.actions.joint_pos.scale = 0.25

        # 5. 事件 (随机化)
        self.events.push_robot = None # 暂时禁用推力，先学走
        
        # 质量随机化：基座 0.9kg，随机加减 0.2kg
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.2, 0.2)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        
        # 外力干扰
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        
        # 初始关节随机化：在默认姿态附近随机 +/- 0.1 rad
        # 你的默认姿态是 shank=0.8/-0.8, thigh=0
        self.events.reset_robot_joints.params["position_range"] = (0.5, 0.5) # 范围稍微大点
        
        # 初始位置随机化
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.base_com = None

        # 6. 奖励函数 (Rewards)
        # 脚部腾空时间奖励：鼓励抬腿
        # 你的末端连杆叫 wheel_...
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = "wheel_.*"
        self.rewards.feet_air_time.weight = 0.01
        
        # 不需要检测非脚部接触（暂时），或者改为检测 thighs
        self.rewards.undesired_contacts = None
        
        # 力矩惩罚：你的电机本来就需要很大力矩来对抗摩擦，所以惩罚要小，否则它不动了
        self.rewards.dof_torques_l2.weight = -0.00005 # 比 A1 小很多
        
        # 跟踪速度奖励 (核心)
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        
        # 加速度惩罚：减少抖动
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # 7. 终止条件 (Terminations)
        # 身体接触地面就死 (摔倒)
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"


@configclass
class QweDogRoughEnvCfg_PLAY(QweDogRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
