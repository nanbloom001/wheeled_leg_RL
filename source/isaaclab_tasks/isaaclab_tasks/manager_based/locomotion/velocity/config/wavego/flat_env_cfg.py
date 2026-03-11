# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from collections.abc import Sequence
import torch
import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg, GaussianNoiseCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs import mdp as core_mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.wavego import WAVEGO_CFG


# --- 自定义奖励: 惩罚横向 (x轴) 基座速度 ---
def _lin_vel_x_l2(
    env,
    command_name: str,
    strafe_command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Penalize unintended x-axis base velocity when strafe command is near zero.

    For WAVEGO, +y is the learned forward axis, so x corresponds to lateral drift.
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    no_strafe_cmd = torch.abs(command[:, 0]) < strafe_command_threshold
    return torch.square(asset.data.root_lin_vel_b[:, 0]) * no_strafe_cmd


class StratifiedVelocityCommand(UniformVelocityCommand):
    """Velocity command sampler with fixed curriculum buckets for targeted skill training."""

    cfg: "StratifiedVelocityCommandCfg"

    def __init__(self, cfg: "StratifiedVelocityCommandCfg", env):
        super().__init__(cfg, env)

        probs = cfg.specialized_env_probs
        if any(prob < 0.0 for prob in probs):
            raise ValueError(f"specialized_env_probs must be non-negative, got: {probs}")
        if sum(probs) > 1.0:
            raise ValueError(f"specialized_env_probs must sum to <= 1.0, got: {sum(probs):.3f}")

    def _sample_uniform_with_min_abs(
        self,
        env_id_tensor: torch.Tensor,
        axis: int,
        value_range: tuple[float, float],
        min_abs: float,
    ):
        if env_id_tensor.numel() == 0:
            return

        values = torch.empty(env_id_tensor.numel(), device=self.device).uniform_(*value_range)
        if min_abs > 0.0 and value_range[0] < 0.0 < value_range[1]:
            small_mask = values.abs() < min_abs
            for _ in range(4):
                if not torch.any(small_mask):
                    break
                values[small_mask] = torch.empty(int(small_mask.sum()), device=self.device).uniform_(*value_range)
                small_mask = values.abs() < min_abs
            if torch.any(small_mask):
                random_sign = torch.where(
                    torch.rand(int(small_mask.sum()), device=self.device) < 0.5,
                    -torch.ones(int(small_mask.sum()), device=self.device),
                    torch.ones(int(small_mask.sum()), device=self.device),
                )
                values[small_mask] = random_sign * min_abs

        self.vel_command_b[env_id_tensor, axis] = values

    def _resample_command(self, env_ids: Sequence[int]):
        env_id_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_id_tensor.numel() == 0:
            return

        self.vel_command_b[env_id_tensor, :] = 0.0
        self.is_heading_env[env_id_tensor] = False
        self.is_standing_env[env_id_tensor] = False

        standing_prob, yaw_prob, forward_prob, strafe_prob = self.cfg.specialized_env_probs
        random_selector = torch.rand(env_id_tensor.numel(), device=self.device)

        standing_mask = random_selector < standing_prob
        yaw_mask = (random_selector >= standing_prob) & (random_selector < standing_prob + yaw_prob)
        forward_mask = (random_selector >= standing_prob + yaw_prob) & (
            random_selector < standing_prob + yaw_prob + forward_prob
        )
        strafe_mask = (random_selector >= standing_prob + yaw_prob + forward_prob) & (
            random_selector < standing_prob + yaw_prob + forward_prob + strafe_prob
        )
        random_mask = ~(standing_mask | yaw_mask | forward_mask | strafe_mask)

        standing_env_ids = env_id_tensor[standing_mask]
        yaw_env_ids = env_id_tensor[yaw_mask]
        forward_env_ids = env_id_tensor[forward_mask]
        strafe_env_ids = env_id_tensor[strafe_mask]
        random_env_ids = env_id_tensor[random_mask]

        self.is_standing_env[standing_env_ids] = True

        self._sample_uniform_with_min_abs(
            yaw_env_ids, 2, self.cfg.ranges.ang_vel_z, self.cfg.specialized_min_abs_ang_vel_z
        )
        self._sample_uniform_with_min_abs(
            forward_env_ids, 1, self.cfg.ranges.lin_vel_y, self.cfg.specialized_min_abs_lin_vel_y
        )
        self._sample_uniform_with_min_abs(
            strafe_env_ids, 0, self.cfg.ranges.lin_vel_x, self.cfg.specialized_min_abs_lin_vel_x
        )

        if random_env_ids.numel() > 0:
            r = torch.empty(random_env_ids.numel(), device=self.device)
            self.vel_command_b[random_env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
            self.vel_command_b[random_env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
            self.vel_command_b[random_env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)


@configclass
class StratifiedVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for stratified velocity command sampling."""

    class_type: type = StratifiedVelocityCommand
    specialized_env_probs: tuple[float, float, float, float] = (0.15, 0.15, 0.15, 0.15)
    specialized_min_abs_lin_vel_x: float = 0.1
    specialized_min_abs_lin_vel_y: float = 0.1
    specialized_min_abs_ang_vel_z: float = 0.2


def _fore_hind_air_time_balance(
    env,
    command_name: str,
    front_sensor_cfg: SceneEntityCfg,
    rear_sensor_cfg: SceneEntityCfg,
):
    """Penalize front-rear air-time imbalance to reduce fore-hind gait asymmetry."""
    contact_sensor: ContactSensor = env.scene.sensors[front_sensor_cfg.name]
    front_air = contact_sensor.data.current_air_time[:, front_sensor_cfg.body_ids]
    rear_air = contact_sensor.data.current_air_time[:, rear_sensor_cfg.body_ids]
    imbalance = torch.abs(front_air.mean(dim=1) - rear_air.mean(dim=1))
    command_mag = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    return imbalance * (command_mag > 0.05)


def _randomize_rigid_body_com_gaussian(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    std_dev: tuple[float, float, float],
    clip_range: tuple[float, float, float],
):
    """Randomize base-link CoM with a clipped zero-mean Gaussian distribution."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    std_tensor = torch.tensor(std_dev, device="cpu")
    clip_tensor = torch.tensor(clip_range, device="cpu")
    samples = torch.randn((len(env_ids), 3), device="cpu") * std_tensor
    samples = torch.clamp(samples, min=-clip_tensor, max=clip_tensor).unsqueeze(1)

    coms = asset.root_physx_view.get_coms().clone()
    coms[env_ids[:, None], body_ids, :3] += samples
    asset.root_physx_view.set_coms(coms, env_ids)


def randomize_rigid_body_material_ordered(
    env,
    env_ids: torch.Tensor | None,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    restitution_range: tuple[float, float],
    num_buckets: int,
    asset_cfg: SceneEntityCfg,
    make_consistent: bool = True,
    static_bias_range: tuple[float, float] = (0.03, 0.15),
):
    """Randomize rigid-body materials while biasing static friction above dynamic friction."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if isinstance(asset, Articulation) and asset_cfg.body_ids != slice(None):
        num_shapes_per_body = []
        for link_path in asset.root_physx_view.link_paths[0]:
            link_physx_view = asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore[attr-defined]
            num_shapes_per_body.append(link_physx_view.max_shapes)
    else:
        num_shapes_per_body = None

    dynamic_samples = math_utils.sample_uniform(
        dynamic_friction_range[0], dynamic_friction_range[1], (num_buckets,), device="cpu"
    )
    if make_consistent:
        static_bias = math_utils.sample_uniform(static_bias_range[0], static_bias_range[1], (num_buckets,), device="cpu")
        static_samples = torch.clamp(
            dynamic_samples + static_bias,
            min=static_friction_range[0],
            max=static_friction_range[1],
        )
        dynamic_samples = torch.minimum(dynamic_samples, static_samples - 1.0e-4)
    else:
        static_samples = math_utils.sample_uniform(
            static_friction_range[0], static_friction_range[1], (num_buckets,), device="cpu"
        )
    restitution_samples = math_utils.sample_uniform(
        restitution_range[0], restitution_range[1], (num_buckets,), device="cpu"
    )
    material_buckets = torch.stack((static_samples, dynamic_samples, restitution_samples), dim=1)

    total_num_shapes = asset.root_physx_view.max_shapes
    bucket_ids = torch.randint(0, num_buckets, (len(env_ids), total_num_shapes), device="cpu")
    material_samples = material_buckets[bucket_ids]
    materials = asset.root_physx_view.get_material_properties()

    if num_shapes_per_body is not None:
        for body_id in asset_cfg.body_ids:
            start_idx = sum(num_shapes_per_body[:body_id])
            end_idx = start_idx + num_shapes_per_body[body_id]
            materials[env_ids, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
    else:
        materials[env_ids] = material_samples[:]

    asset.root_physx_view.set_material_properties(materials, env_ids)


@configclass
class WavegoFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # 执行父类初始化
        super().__post_init__()

        # ==========================================================
        #                 WAVEGO 奖励权重表
        # ==========================================================
        W = {
            # -------------------- 任务目标 (Task) --------------------
            "track_lin_vel_xy": 1.75,
            "track_ang_vel_z":  0.75,
            "feet_air_time":    0.3,
            "base_height":      2.0,

            # -------------------- 姿态稳定性 (Stability) --------------------
            "flat_orientation": -3.0,
            "ang_vel_xy_l2":    -0.12,
            "lin_vel_z_l2":     -2.0,
            "joint_deviation_l1": -0.1,

            # -------------------- 零命令稳定性 (Zero-cmd Stability) --------------------
            "stand_still":      -1.0,
            "lin_vel_x_l2":     -1.5,

            # -------------------- 安全与保护 (Safety) --------------------
            "undesired_contacts": -1.0,
            "feet_slide":       -0.08,
            "fore_hind_balance": -0.2,

            # -------------------- 动作正则化 (Regularization, 强化平滑) --------------------
            "action_rate_l2":   -0.075,
            "dof_torques_l2":   -1e-5,
            "dof_acc_l2":       -5e-7
        }
        # ==========================================================

        # --- PhysX GPU 优化 ---
        self.sim.physx.gpu_collision_stack_size = 2**28 
        
        # --- 控制频率提升至 50Hz ---
        self.decimation = 4 
        self.sim.render_interval = self.decimation

        # 1. 场景配置 (DelayedPD 执行器, 模拟通信延迟)
        WAVEGO_DELAYED_LEGS = DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            stiffness=400.0,
            damping=10.0,
            effort_limit=1.96,
            velocity_limit=11.1,
            armature=0.01,
            friction=0.4,
            min_delay=2,   # 10ms @ 200Hz phys
            max_delay=8,   # 40ms @ 200Hz phys
        )
        self.scene.num_envs = 8192
        self.scene.robot = WAVEGO_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            actuators={"legs": WAVEGO_DELAYED_LEGS},
        )
        self.scene.height_scanner = None 

        # --- 观测空间噪音 (含 IMU 偏置漂移) ---
        self.observations.policy.height_scan = None
        self.observations.policy.projected_gravity.noise = Unoise(n_min=-0.05, n_max=0.05)
        # IMU 角速度: 基础均匀噪声 + 上电零偏漂移 (每次 reset 采样)
        self.observations.policy.base_ang_vel.noise = NoiseModelWithAdditiveBiasCfg(
            noise_cfg=Unoise(n_min=-0.2, n_max=0.2),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05),
        )
        self.observations.policy.joint_pos.noise = Unoise(n_min=-0.03, n_max=0.03)
        # 关节速度: 加大噪声, 模拟 50Hz 低频差分估计
        self.observations.policy.joint_vel.noise = Unoise(n_min=-2.0, n_max=2.0)
        
        # 2. 地形 (平地)
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None 
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.collision_group = -1
        self.scene.terrain.physics_material.static_friction = 1.0
        self.scene.terrain.physics_material.dynamic_friction = 1.0
        self.curriculum.terrain_levels = None

        # 3. 动作空间
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", 
            joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"], 
            scale=0.25, 
            use_default_offset=True
        )
        self.actions.joint_pos.noise = Unoise(n_min=-0.02, n_max=0.02)

        # 4. 指令空间
        self.commands.base_velocity = StratifiedVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.15,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=True,
            specialized_env_probs=(0.15, 0.15, 0.15, 0.15),
            specialized_min_abs_lin_vel_x=0.1,
            specialized_min_abs_lin_vel_y=0.1,
            specialized_min_abs_ang_vel_z=0.2,
            ranges=StratifiedVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.8, 0.8),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-1.0, 1.0),
                heading=None,
            ),
        )

        # 5. 随机化事件 (全面恢复并增强)
        
        # 外部扰动: 每 10-15s 推一次
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        )
        
        # 质量随机化: ±100g
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.1, 0.1)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        
        # 质心(CoM)偏移随机化: 零均值高斯, 截断到 ±3cm
        self.events.base_com = EventTerm(
            func=_randomize_rigid_body_com_gaussian,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "std_dev": (0.01, 0.01, 0.01),
                "clip_range": (0.03, 0.03, 0.03),
            },
        )

        # 摩擦力随机化: 静摩擦略大于动摩擦
        self.events.physics_material = EventTerm(
            func=randomize_rigid_body_material_ordered,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.3, 1.25),
                "dynamic_friction_range": (0.15, 1.0),
                "static_bias_range": (0.05, 0.2),
                "restitution_range": (0.0, 0.1),
                "num_buckets": 64,
                "make_consistent": True,
            },
        )

        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        self.events.reset_robot_joints.params["position_range"] = (0.5, 0.5)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.1, 0.1)}, 
            "velocity_range": {
                "x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.2, 0.2),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }

        # 6. 奖励函数
        self.rewards.track_lin_vel_xy_exp.weight = W["track_lin_vel_xy"]
        self.rewards.track_ang_vel_z_exp.weight = W["track_ang_vel_z"]
        self.rewards.feet_air_time.weight = W["feet_air_time"]
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_calf"
        self.rewards.feet_air_time.params["threshold"] = 0.35
        
        self.rewards.flat_orientation_l2.weight = W["flat_orientation"]
        self.rewards.ang_vel_xy_l2.weight = W["ang_vel_xy_l2"]
        self.rewards.lin_vel_z_l2.weight = W["lin_vel_z_l2"]
        
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=W["base_height"],
            params={"target_height": 0.20, "asset_cfg": SceneEntityCfg("robot", body_names="base_link")}
        )
        
        self.rewards.joint_deviation_l1 = RewTerm(
            func=core_mdp.joint_deviation_l1,
            weight=W["joint_deviation_l1"],
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])}
        )
        
        # --- 零命令稳定性 ---
        self.rewards.stand_still = RewTerm(
            func=mdp.stand_still_joint_deviation_l1,
            weight=W["stand_still"],
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]),
            },
        )
        
        # --- 横向漂移抑制: 对 WAVEGO 来说 x 是横向, y 是前进 ---
        self.rewards.lin_vel_x_l2 = RewTerm(
            func=_lin_vel_x_l2,
            weight=W["lin_vel_x_l2"],
            params={
                "command_name": "base_velocity",
                "strafe_command_threshold": 0.1,
            },
        )

        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=W["feet_slide"],
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf", "RL_calf", "RR_calf"]),
                "asset_cfg": SceneEntityCfg("robot", body_names=["FL_calf", "FR_calf", "RL_calf", "RR_calf"]),
            },
        )

        self.rewards.fore_hind_balance = RewTerm(
            func=_fore_hind_air_time_balance,
            weight=W["fore_hind_balance"],
            params={
                "command_name": "base_velocity",
                "front_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"]),
                "rear_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"]),
            },
        )
        
        self.rewards.undesired_contacts.weight = W["undesired_contacts"]
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["base_link", ".*_thigh"]
        
        self.rewards.action_rate_l2.weight = W["action_rate_l2"]
        self.rewards.dof_torques_l2.weight = W["dof_torques_l2"]
        self.rewards.dof_acc_l2.weight = W["dof_acc_l2"]

        # 7. 终止条件
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["base_link"]


@configclass
class WavegoFlatEnvCfg_PLAY(WavegoFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None