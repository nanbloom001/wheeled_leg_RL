# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the QWE Dog robot."""

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration - Actuators.
##

# 1. 腿部执行器 (位置控制)
QWE_DOG_LEGS_CFG = ImplicitActuatorCfg(
    joint_names_expr=["thigh_.*", "shank_.*"],
    stiffness=400.0,
    damping=10.0,
    effort_limit_sim=1.27,
    velocity_limit_sim=6.3,
    armature=0.01,
    friction=0.4,
)

# 2. 轮子执行器 (强力阻尼/刹车模式)
# Sim-to-Real Phase 1: 锁死轮子，模拟圆足
QWE_DOG_WHEELS_CFG = ImplicitActuatorCfg(
    joint_names_expr=["wheel_.*"],
    stiffness=0.0,       # 不进行位置伺服
    damping=1000.0,      # 极大的阻尼，相当于刹车
    effort_limit_sim=1000.0, # 允许足够大的刹车力矩
    velocity_limit_sim=100.0,
    armature=0.01,
    friction=1.0,
)

##
# Configuration - Articulation.
##

QWE_DOG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/User/qwe_dog.usd",
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
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=16, 
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        rot=(1.0, 0.0, 0.0, 0.0), 
        
        joint_pos={
            "thigh_.*": 0.0,
            "shank_front_.*": -0.8,
            "shank_rear_.*": 0.8,
            "wheel_.*": 0.0,
        },
    ),
    # 注册两个 Actuator
    actuators={
        "legs": QWE_DOG_LEGS_CFG, 
        "wheels": QWE_DOG_WHEELS_CFG
    },
    soft_joint_pos_limit_factor=0.9,
)
"""Configuration of QWE Dog robot."""