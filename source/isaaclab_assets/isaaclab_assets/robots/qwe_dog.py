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

QWE_DOG_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=["thigh_.*", "shank_.*", "wheel_.*"],
    stiffness=400.0,
    damping=10.0,
    effort_limit_sim=1.27,
    velocity_limit_sim=6.3,
    armature=0.01,
    friction=0.4,
)
"""Configuration for QWE Dog actuators (Implicit PD)."""

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
            # --- 关键修改：开启自碰撞 ---
            enabled_self_collisions=True, 
            solver_position_iteration_count=16, 
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        # 修正后的 URDF 应该不需要额外的全局旋转
        rot=(1.0, 0.0, 0.0, 0.0), 
        
        joint_pos={
            "thigh_.*": 0.0,
            "shank_front_.*": -0.8,
            "shank_rear_.*": 0.8,
            "wheel_.*": 0.0,
        },
    ),
    actuators={"legs": QWE_DOG_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.9,
)
"""Configuration of QWE Dog robot."""