# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.qwe_dog import QWE_DOG_CFG


@configclass
class QweDogFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
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
        self.sim.physx.gpu_heap_capacity = 2**26 
        self.sim.physx.gpu_temp_buffer_capacity = 2**26
        self.sim.physx.gpu_resource_part_data_capacity = 2**20
        self.sim.physx.gpu_collision_stack_size = 2**28

        # 1. 替换机器人
        self.scene.robot = QWE_DOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner = None # 平地不需要高度扫描

        # --- 观测空间修正 ---
        self.observations.policy.height_scan = None

        # 2. 强制使用平坦地形
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None 
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.collision_group = -1
        # 增加地面摩擦力
        self.scene.terrain.physics_material.static_friction = 1.0
        self.scene.terrain.physics_material.dynamic_friction = 1.0
        
        # 移除 curriculum
        self.curriculum.terrain_levels = None

        # --- 3. 动作空间定制 (只控制 8 个腿部关节) ---
        # 重写 joint_pos action
        # 仅匹配大腿和小腿，忽略轮子 (wheel_.*)
        # 轮子将保持在默认位置 (0.0) 并由 stiffness=400 锁定
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", 
            joint_names=["thigh_.*", "shank_.*"], 
            scale=0.25, 
            use_default_offset=True
        )

        # --- 4. 指令空间定制 (支持差速转向) ---
        # 允许 X 轴线速度和 Yaw 轴角速度，禁用 Y 轴平移
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.3) # 降低速度上限
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5) # 降低转向速度上限
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi) 

        # 5. 事件 (随机化)
        self.events.push_robot = None 
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.2, 0.2)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        self.events.reset_robot_joints.params["position_range"] = (0.5, 0.5)
        # 初始位置随机化也限制在 X-Z 平面附近
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.1, 0.1)}, # 允许微小偏航
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (0.0, 0.0), # 无横向速度
                "z": (-0.5, 0.5),
                "roll": (0.0, 0.0),
                "pitch": (-0.5, 0.5),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # 6. 奖励函数定制
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = "wheel_.*"
        self.rewards.feet_air_time.weight = 0.5 # 进一步提高抬腿奖励，鼓励踏步
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.00001 # 降低力矩惩罚
        
        # 禁用 Y 轴和 Yaw 轴速度跟踪奖励 (因为指令是 0，跟踪 0 没意义，或者保留作为惩罚)
        # 实际上，保留 track_lin_vel_xy_exp 即可，因为 target_y=0，如果有 y 速度自然会获得低分
        # 但我们可以增加对侧向速度的显式惩罚
        self.rewards.ang_vel_xy_l2.weight = -0.05
        
        # 新增/增强惩罚：严厉禁止侧向移动和侧倾
        # lin_vel_y_l2 不在默认列表中，需要自定义或重用
        # 但我们有 flat_orientation_l2 (惩罚 roll 和 pitch)
        # 对于平面行走，pitch 是允许的 (点头)，但 roll (侧倾) 绝对禁止
        # 默认 flat_orientation_l2 惩罚的是 (roll, pitch) 的投影重力误差
        # 我们可以调高它来惩罚侧倾，但也可能误伤 pitch
        self.rewards.flat_orientation_l2.weight = -2.0 

        # 7. 终止条件
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"


@configclass
class QweDogFlatEnvCfg_PLAY(QweDogFlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
