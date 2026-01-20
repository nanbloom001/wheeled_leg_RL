# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab_assets.robots.qwe_dog import QWE_DOG_CFG


@configclass
class QweDogFlatEnvCfg_DEBUG(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # --- PhysX GPU 内存优化 ---
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

        # --- DEBUG MODE: 默认频率 ---
        self.decimation = 10 
        self.sim.render_interval = self.decimation

        # 1. 替换机器人
        self.scene.num_envs = 4096 
        self.scene.robot = QWE_DOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner = None

        # --- 观测空间修正 ---
        self.observations.policy.height_scan = None
        # NO NOISE

        # 2. 地形
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None 
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.collision_group = -1
        # 默认摩擦力
        self.scene.terrain.physics_material.static_friction = 1.0
        self.scene.terrain.physics_material.dynamic_friction = 1.0
        
        self.curriculum.terrain_levels = None

        # --- 3. 动作空间 (8-DOF) ---
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", 
            joint_names=["thigh_.*", "shank_.*"], 
            scale=0.25, 
            use_default_offset=True
        )
        # NO NOISE

        # --- 4. 指令空间 ---
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.3) 
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5) 
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi) 

        # 5. 事件
        self.events.push_robot = None 
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.2, 0.2)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        self.events.reset_robot_joints.params["position_range"] = (0.5, 0.5)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.1, 0.1)}, 
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (0.0, 0.0), 
                "z": (-0.5, 0.5),
                "roll": (0.0, 0.0),
                "pitch": (-0.5, 0.5),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None
        # NO FRICTION RANDOMIZATION

        # 6. 奖励函数
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = "wheel_.*"
        self.rewards.feet_air_time.weight = 0.5 
        # NO UNDESIRED CONTACTS (KNEE)
        self.rewards.undesired_contacts = None
        
        self.rewards.dof_torques_l2.weight = -0.00001 
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -2.0 

        # 7. 终止条件
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link" # Only base
