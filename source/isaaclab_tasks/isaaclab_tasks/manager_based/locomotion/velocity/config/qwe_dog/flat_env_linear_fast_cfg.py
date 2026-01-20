# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject

# 引入基类配置
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

# 引入 MDP
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs import mdp as core_mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.qwe_dog import QWE_DOG_CFG

# ==================================================================================================
#                                     自定义奖励函数定义
# ==================================================================================================

def ang_vel_z_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚 Z 轴角速度 (Yaw Rate) 的 L2 范数 (防止转向)。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])

def track_lin_vel_x_exp(env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """仅追踪 X 轴 (前进/后退) 速度的指数奖励。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    vel_x = asset.data.root_lin_vel_b[:, 0]
    error = torch.square(cmd_x - vel_x)
    return torch.exp(-error / std**2)

def lin_vel_y_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚 Y 轴 (横移) 速度的 L2 范数。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 1])

# ==================================================================================================
#                                     环境配置类定义
# ==================================================================================================

@configclass
class QweDogFlatLinearFastEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ==========================================================
        #                 Sim-to-Real 直线行走奖励权重表
        # ==========================================================
        W = {
            # -------------------- 任务目标 (Task) --------------------
            "track_lin_vel_x":  3.0,      
            "lin_vel_y_l2":     -1.0,
            "feet_air_time":    0.15,      
            
            # -------------------- 姿态稳定性 (Stability) --------------------
            "flat_orientation":   -0.5,   
            "ang_vel_z_l2":       -1.0,   
            "ang_vel_xy_l2":      -0.2,   
            "lin_vel_z_l2":       -0.5,   
            
            # -------------------- 安全与保护 (Safety) --------------------
            "undesired_contacts": -2.0,   
            "foot_impact_forces": -0.05,  
            
            # -------------------- 动作正则化 (Regularization) --------------------
            "joint_deviation_l1": -0.2,   
            "action_rate_l2":     -0.02,  
            "joint_vel_l2":       -0.005, 
            "dof_torques_l2":     -1e-5,  
            "dof_acc_l2":         -2.5e-7 
        }
        # ==========================================================

        # --- PhysX GPU 内存优化 (High Mem) ---
        self.sim.physx.gpu_max_rigid_contact_count = 2**24
        self.sim.physx.gpu_max_rigid_patch_count = 2**24
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**24
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**24
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**24
        self.sim.physx.gpu_max_soft_body_contacts = 2**24
        self.sim.physx.gpu_max_particle_contacts = 2**24
        self.sim.physx.gpu_heap_capacity = 2**27
        self.sim.physx.gpu_temp_buffer_capacity = 2**27
        self.sim.physx.gpu_resource_part_data_capacity = 2**21
        self.sim.physx.gpu_collision_stack_size = 2**29

        # --- Sim-to-Real: 20Hz 控制频率 ---
        self.decimation = 10 
        self.sim.render_interval = self.decimation

        # 1. 场景配置
        self.scene.num_envs = 8192 
        self.scene.robot = QWE_DOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner = None 

        # 2. 观测空间
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.noise = Unoise(n_min=-0.05, n_max=0.05)
        self.observations.policy.joint_vel.noise = Unoise(n_min=-0.5, n_max=0.5)
        self.observations.policy.base_lin_vel.noise = Unoise(n_min=-0.1, n_max=0.1)
        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.2, n_max=0.2)

        # 3. 地形
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None 
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.collision_group = -1
        self.scene.terrain.physics_material.static_friction = 1.0
        self.scene.terrain.physics_material.dynamic_friction = 1.0
        
        # [修复] 显式禁用地形课程，因为平坦地形没有生成器
        self.curriculum.terrain_levels = None

        # 4. 动作空间
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", 
            joint_names=["thigh_.*", "shank_.*"], 
            scale=0.25, 
            use_default_offset=True
        )
        self.actions.joint_pos.noise = Unoise(n_min=-0.02, n_max=0.02) 

        # 5. 指令空间
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0) 
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # 6. 事件
        self.events.push_robot = None 
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.2, 0.2)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        self.events.reset_robot_joints.params["position_range"] = (0.5, 0.5)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.1, 0.1)}, 
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (0.0, 0.0), "z": (-0.5, 0.5),
                "roll": (0.0, 0.0), "pitch": (-0.5, 0.5), "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None
        self.events.physics_material.params["static_friction_range"] = (0.5, 1.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 1.25)

        # ==========================================================
        #                 应用奖励函数配置
        # ==========================================================
        
        # 1. 任务追踪
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        
        self.rewards.track_lin_vel_x_exp = RewTerm(
            func=track_lin_vel_x_exp,
            weight=W["track_lin_vel_x"],
            params={"command_name": "base_velocity", "std": 0.25}
        )
        self.rewards.lin_vel_y_l2 = RewTerm(
            func=lin_vel_y_l2,
            weight=W["lin_vel_y_l2"]
        )
        
        # 2. 步态鼓励
        self.rewards.feet_air_time.weight = W["feet_air_time"]
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = "wheel_.*"
        
        # 3. 姿态惩罚
        self.rewards.flat_orientation_l2.weight = W["flat_orientation"]
        self.rewards.ang_vel_xy_l2.weight = W["ang_vel_xy_l2"]
        self.rewards.lin_vel_z_l2.weight = W["lin_vel_z_l2"]
        
        # 4. 直线特供
        self.rewards.ang_vel_z_l2 = RewTerm(
            func=ang_vel_z_l2, 
            weight=W["ang_vel_z_l2"]
        )

        # 5. 安全与接触
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=W["undesired_contacts"], 
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", "shank_.*"]), 
                "threshold": 1.0 
            },
        )
        self.rewards.foot_impact_forces = RewTerm(
            func=core_mdp.contact_forces,
            weight=W["foot_impact_forces"],
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*"),
                "threshold": 20.0
            }
        )

        # 6. 动作规范
        self.rewards.joint_deviation_l1 = RewTerm(
            func=core_mdp.joint_deviation_l1,
            weight=W["joint_deviation_l1"],
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["thigh_.*", "shank_.*"])}
        )
        self.rewards.action_rate_l2.weight = W["action_rate_l2"]
        self.rewards.dof_torques_l2.weight = W["dof_torques_l2"]
        self.rewards.dof_acc_l2.weight = W["dof_acc_l2"]
        
        # 7. 终止条件
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"


@configclass
class QweDogFlatLinearFastEnvCfg_PLAY(QweDogFlatLinearFastEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
