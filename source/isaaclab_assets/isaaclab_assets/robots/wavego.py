# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the WAVEGO robot."""

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration - Actuators.
##

WAVEGO_LEGS_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    stiffness=400.0,
    damping=10.0,
    effort_limit_sim=1.96, 
    velocity_limit_sim=11.1,
    armature=0.01,
    friction=0.4, # 匹配减速箱自锁特性
)

##
# Configuration - Articulation.
##

WAVEGO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/User/WAVEGO.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        # 注意：足端与地面摩擦力 0.5 建议在训练环境的 TerrainCfg 或 MaterialCfg 中显式设定
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=16, 
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.25), 
        rot=(1.0, 0.0, 0.0, 0.0), 
        
        joint_pos={
            "FL_hip_joint": 0.100,
            "FR_hip_joint": -0.100,
            "RL_hip_joint": -0.100,
            "RR_hip_joint": 0.100,
            "FL_thigh_joint": -0.650,
            "FR_thigh_joint": 0.650,
            "RL_thigh_joint": -0.650,
            "RR_thigh_joint": 0.650,
            "FL_calf_joint": 0.600,
            "FR_calf_joint": -0.600,
            "RL_calf_joint": 0.600,
            "RR_calf_joint": -0.600,
        },
    ),
    actuators={
        "legs": WAVEGO_LEGS_CFG, 
    },
    soft_joint_pos_limit_factor=0.9,
)
