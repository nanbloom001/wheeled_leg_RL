from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


EXPECTED_JOINT_ORDER = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

EXPECTED_ACTUATOR_ORDER = [
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
]

OBS_ORDER = (
    ["base_lin_vel"] * 3
    + ["base_ang_vel"] * 3
    + ["projected_gravity"] * 3
    + ["velocity_commands"] * 3
    + ["joint_pos_rel"] * 12
    + ["joint_vel_rel"] * 12
    + ["last_action"] * 12
)

NUM_OBS = 48
NUM_ACTIONS = 12


@dataclass
class IsaacParams:
    sim_dt: float
    decimation: int
    kp: float
    kd: float
    effort_limit: float
    armature: float
    friction: float
    default_joint_pos_mj: np.ndarray


@dataclass
class PolicyIOConfig:
    policy_joint_names: list[str]
    default_joint_pos_policy: np.ndarray
    action_scale: float


@dataclass
class ObsNormalizer:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-8
    clip: float = 5.0


@dataclass
class RolloutDiagnostics:
    vx_err_sum: float = 0.0
    vy_err_sum: float = 0.0
    wz_err_sum: float = 0.0
    vx_err_abs_sum: float = 0.0
    vy_err_abs_sum: float = 0.0
    wz_err_abs_sum: float = 0.0
    joint_vel_rms_sum: float = 0.0
    joint_pos_rel_rms_sum: float = 0.0
    action_delta_l2_sum: float = 0.0
    base_z_min: float = 1e9
    base_z_max: float = -1e9
    pitch_abs_max_deg: float = 0.0
    roll_abs_max_deg: float = 0.0
    count: int = 0

    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        action_prev: np.ndarray,
        command: np.ndarray,
        base_z: float,
        roll_deg: float,
        pitch_deg: float,
    ) -> None:
        vx, vy = float(obs[0]), float(obs[1])
        wz = float(obs[5])
        self.vx_err_sum += vx - float(command[0])
        self.vy_err_sum += vy - float(command[1])
        self.wz_err_sum += wz - float(command[2])
        self.vx_err_abs_sum += abs(vx - float(command[0]))
        self.vy_err_abs_sum += abs(vy - float(command[1]))
        self.wz_err_abs_sum += abs(wz - float(command[2]))

        joint_pos_rel = obs[12:24]
        joint_vel_rel = obs[24:36]
        self.joint_pos_rel_rms_sum += float(np.sqrt(np.mean(np.square(joint_pos_rel))))
        self.joint_vel_rms_sum += float(np.sqrt(np.mean(np.square(joint_vel_rel))))
        self.action_delta_l2_sum += float(np.linalg.norm(action - action_prev))

        self.base_z_min = min(self.base_z_min, base_z)
        self.base_z_max = max(self.base_z_max, base_z)
        self.roll_abs_max_deg = max(self.roll_abs_max_deg, abs(roll_deg))
        self.pitch_abs_max_deg = max(self.pitch_abs_max_deg, abs(pitch_deg))
        self.count += 1

    def summary_str(self) -> str:
        if self.count == 0:
            return "diagnostics: empty"
        return (
            "diagnostics: "
            f"vx_err_mean={self.vx_err_sum/self.count:.4f}, "
            f"vy_err_mean={self.vy_err_sum/self.count:.4f}, "
            f"wz_err_mean={self.wz_err_sum/self.count:.4f}, "
            f"|vx-cmd|_mean={self.vx_err_abs_sum/self.count:.4f}, "
            f"|vy-cmd|_mean={self.vy_err_abs_sum/self.count:.4f}, "
            f"|wz-cmd|_mean={self.wz_err_abs_sum/self.count:.4f}, "
            f"joint_vel_rms_mean={self.joint_vel_rms_sum/self.count:.4f}, "
            f"joint_pos_rel_rms_mean={self.joint_pos_rel_rms_sum/self.count:.4f}, "
            f"action_delta_l2_mean={self.action_delta_l2_sum/self.count:.4f}, "
            f"base_z_range=[{self.base_z_min:.4f},{self.base_z_max:.4f}], "
            f"roll_abs_max_deg={self.roll_abs_max_deg:.2f}, "
            f"pitch_abs_max_deg={self.pitch_abs_max_deg:.2f}"
        )


class ActorMLP:
    def __init__(self, checkpoint_path: str | Path):
        import torch

        self._torch = torch
        ckpt = torch.load(Path(checkpoint_path), map_location="cpu")
        state_dict = ckpt["model_state_dict"]

        self.net = torch.nn.Sequential(
            torch.nn.Linear(NUM_OBS, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, NUM_ACTIONS),
        )

        actor_state = {}
        for key, value in state_dict.items():
            if key.startswith("actor."):
                actor_state[key.replace("actor.", "")] = value
        self.net.load_state_dict(actor_state)
        self.net.eval()

    def __call__(self, obs_norm: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            x = self._torch.from_numpy(obs_norm.astype(np.float32)).unsqueeze(0)
            out = self.net(x).squeeze(0).cpu().numpy().astype(np.float64)
        return out


def _normalize_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat_wxyz)
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be > 0.")
    return quat_wxyz / norm


def quat_apply_inverse_wxyz(quat_wxyz: np.ndarray, vec_w: np.ndarray) -> np.ndarray:
    q = _normalize_quat_wxyz(np.asarray(quat_wxyz, dtype=np.float64))
    v = np.asarray(vec_w, dtype=np.float64)

    q_vec = q[1:]
    q_w = q[0]

    t = 2.0 * np.cross(q_vec, v)
    rotated = v + q_w * t + np.cross(q_vec, t)
    return rotated


def base_lin_vel(root_lin_vel_b: np.ndarray) -> np.ndarray:
    return np.asarray(root_lin_vel_b, dtype=np.float64)


def base_ang_vel(root_ang_vel_b: np.ndarray) -> np.ndarray:
    return np.asarray(root_ang_vel_b, dtype=np.float64)


def projected_gravity(root_link_quat_wxyz: np.ndarray, gravity_vec_w: np.ndarray) -> np.ndarray:
    return quat_apply_inverse_wxyz(root_link_quat_wxyz, gravity_vec_w)


def generated_commands(command: np.ndarray) -> np.ndarray:
    return np.asarray(command, dtype=np.float64)


def joint_pos_rel(joint_pos: np.ndarray, default_joint_pos: np.ndarray) -> np.ndarray:
    return np.asarray(joint_pos, dtype=np.float64) - np.asarray(default_joint_pos, dtype=np.float64)


def joint_vel_rel(joint_vel: np.ndarray, default_joint_vel: np.ndarray) -> np.ndarray:
    return np.asarray(joint_vel, dtype=np.float64) - np.asarray(default_joint_vel, dtype=np.float64)


def last_action(action: np.ndarray) -> np.ndarray:
    return np.asarray(action, dtype=np.float64)


def build_policy_observation(
    root_lin_vel_b: np.ndarray,
    root_ang_vel_b: np.ndarray,
    root_link_quat_wxyz: np.ndarray,
    gravity_vec_w: np.ndarray,
    command: np.ndarray,
    joint_pos: np.ndarray,
    default_joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    default_joint_vel: np.ndarray,
    previous_action: np.ndarray,
) -> np.ndarray:
    obs = np.concatenate(
        [
            base_lin_vel(root_lin_vel_b),
            base_ang_vel(root_ang_vel_b),
            projected_gravity(root_link_quat_wxyz, gravity_vec_w),
            generated_commands(command),
            joint_pos_rel(joint_pos, default_joint_pos),
            joint_vel_rel(joint_vel, default_joint_vel),
            last_action(previous_action),
        ]
    )
    if obs.shape != (NUM_OBS,):
        raise ValueError(f"Observation shape mismatch: expected {(NUM_OBS,)}, got {obs.shape}")
    return obs


def apply_obs_normalization(obs: np.ndarray, normalizer: ObsNormalizer) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float64)
    if obs.shape != (NUM_OBS,):
        raise ValueError(f"Observation shape mismatch for normalization: expected {(NUM_OBS,)}, got {obs.shape}")
    if normalizer.mean.shape != (NUM_OBS,) or normalizer.std.shape != (NUM_OBS,):
        raise ValueError("Normalizer mean/std must be 48-d vectors.")

    normalized = (obs - normalizer.mean) / (normalizer.std + normalizer.eps)
    if normalizer.clip is not None:
        normalized = np.clip(normalized, -normalizer.clip, normalizer.clip)
    return normalized


def load_isaac_params_from_env_yaml(env_yaml_path: str | Path, default_joint_pos_mj: np.ndarray) -> IsaacParams:
    import yaml

    with open(env_yaml_path) as f:
        env_cfg = yaml.load(f, Loader=yaml.Loader)

    legs_cfg = env_cfg["scene"]["robot"]["actuators"]["legs"]
    return IsaacParams(
        sim_dt=float(env_cfg["sim"]["dt"]),
        decimation=int(env_cfg["decimation"]),
        kp=float(legs_cfg["stiffness"]),
        kd=float(legs_cfg["damping"]),
        effort_limit=float(legs_cfg["effort_limit_sim"]),
        armature=float(legs_cfg["armature"]),
        friction=float(legs_cfg["friction"]),
        default_joint_pos_mj=default_joint_pos_mj,
    )


def load_policy_io_config(io_descriptor_path: str | Path) -> PolicyIOConfig:
    import yaml

    with open(io_descriptor_path) as f:
        io_cfg = yaml.safe_load(f)

    action_term = io_cfg["actions"][0]
    policy_joint_names = list(action_term["joint_names"])
    action_scale = float(action_term["scale"])

    default_joint_pos_policy = np.array(io_cfg["articulations"]["robot"]["default_joint_pos"], dtype=np.float64)
    return PolicyIOConfig(
        policy_joint_names=policy_joint_names,
        default_joint_pos_policy=default_joint_pos_policy,
        action_scale=action_scale,
    )


def _policy_to_mj_index(policy_joint_names: list[str], mujoco_joint_names: list[str]) -> np.ndarray:
    if set(policy_joint_names) != set(mujoco_joint_names):
        raise ValueError(
            "Policy joint set and MuJoCo joint set mismatch. "
            f"policy={policy_joint_names}, mujoco={mujoco_joint_names}"
        )
    return np.array([mujoco_joint_names.index(name) for name in policy_joint_names], dtype=np.int64)


def reorder_mj_to_policy(vec_mj: np.ndarray, policy_to_mj_idx: np.ndarray) -> np.ndarray:
    return np.asarray(vec_mj, dtype=np.float64)[policy_to_mj_idx]


def reorder_policy_to_mj(vec_policy: np.ndarray, policy_to_mj_idx: np.ndarray) -> np.ndarray:
    out = np.zeros_like(np.asarray(vec_policy, dtype=np.float64))
    out[policy_to_mj_idx] = np.asarray(vec_policy, dtype=np.float64)
    return out


def load_actor_obs_normalizer_from_checkpoint(checkpoint_path: str | Path) -> ObsNormalizer:
    import torch

    ckpt = torch.load(Path(checkpoint_path), map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    mean = state_dict["actor_obs_normalizer._mean"].cpu().numpy().astype(np.float64).reshape(-1)
    std = state_dict["actor_obs_normalizer._std"].cpu().numpy().astype(np.float64).reshape(-1)
    return ObsNormalizer(mean=mean, std=std)


def sync_mujoco_timing(model: mujoco.MjModel, isaac_params: IsaacParams) -> int:
    model.opt.timestep = isaac_params.sim_dt
    policy_update_interval = isaac_params.decimation
    return policy_update_interval


def _joint_dof_ids(model: mujoco.MjModel) -> list[int]:
    return [int(model.jnt_dofadr[jid]) for jid in range(1, model.njnt)]


def verify_and_sync_actuator_params(model: mujoco.MjModel, isaac_params: IsaacParams, atol: float = 1e-6) -> None:
    kp = model.actuator_gainprm[:, 0]
    kv = -model.actuator_biasprm[:, 2]
    force_low = model.actuator_forcerange[:, 0]
    force_high = model.actuator_forcerange[:, 1]

    if not np.allclose(kp, isaac_params.kp, atol=atol):
        model.actuator_gainprm[:, 0] = isaac_params.kp
        model.actuator_biasprm[:, 1] = -isaac_params.kp
    if not np.allclose(kv, isaac_params.kd, atol=atol):
        model.actuator_biasprm[:, 2] = -isaac_params.kd
    if not np.allclose(force_low, -isaac_params.effort_limit, atol=atol) or not np.allclose(
        force_high, isaac_params.effort_limit, atol=atol
    ):
        model.actuator_forcerange[:, 0] = -isaac_params.effort_limit
        model.actuator_forcerange[:, 1] = isaac_params.effort_limit

    dof_ids = _joint_dof_ids(model)
    if not np.allclose(model.dof_armature[dof_ids], isaac_params.armature, atol=atol):
        model.dof_armature[dof_ids] = isaac_params.armature
    if not np.allclose(model.dof_frictionloss[dof_ids], isaac_params.friction, atol=atol):
        model.dof_frictionloss[dof_ids] = isaac_params.friction

    kp = model.actuator_gainprm[:, 0]
    kv = -model.actuator_biasprm[:, 2]
    force_low = model.actuator_forcerange[:, 0]
    force_high = model.actuator_forcerange[:, 1]

    if not np.allclose(kp, isaac_params.kp, atol=atol):
        raise ValueError("Actuator kp mismatch after sync.")
    if not np.allclose(kv, isaac_params.kd, atol=atol):
        raise ValueError("Actuator kd mismatch after sync.")
    if not np.allclose(force_low, -isaac_params.effort_limit, atol=atol) or not np.allclose(
        force_high, isaac_params.effort_limit, atol=atol
    ):
        raise ValueError("Actuator forcerange mismatch after sync.")


def print_joint_mapping(model: mujoco.MjModel) -> None:
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)]
    print("joint_mapping(index -> name):")
    for i, name in enumerate(joint_names):
        print(f"  {i:02d} -> {name}")


def validate_mujoco_joint_actuator_order(model: mujoco.MjModel) -> None:
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)]
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]

    if joint_names != EXPECTED_JOINT_ORDER:
        raise ValueError(f"MuJoCo joint order mismatch. expected={EXPECTED_JOINT_ORDER}, actual={joint_names}")
    if actuator_names != EXPECTED_ACTUATOR_ORDER:
        raise ValueError(
            f"MuJoCo actuator order mismatch. expected={EXPECTED_ACTUATOR_ORDER}, actual={actuator_names}"
        )


def extract_policy_observation_from_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    default_joint_pos_policy: np.ndarray,
    previous_action_policy: np.ndarray,
    command: np.ndarray,
    policy_to_mj_idx: np.ndarray,
    gravity_vec_w: np.ndarray | None = None,
) -> np.ndarray:
    if gravity_vec_w is None:
        gravity_vec_w = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    vel6 = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_id, vel6, 1)

    root_ang_vel_b = vel6[:3]
    root_lin_vel_b = vel6[3:]
    root_link_quat_wxyz = np.asarray(data.qpos[3:7], dtype=np.float64)
    joint_pos_mj = np.asarray(data.qpos[7:], dtype=np.float64)
    joint_vel_mj = np.asarray(data.qvel[6:], dtype=np.float64)
    joint_pos = reorder_mj_to_policy(joint_pos_mj, policy_to_mj_idx)
    joint_vel = reorder_mj_to_policy(joint_vel_mj, policy_to_mj_idx)
    default_joint_vel = np.zeros(NUM_ACTIONS, dtype=np.float64)

    return build_policy_observation(
        root_lin_vel_b=root_lin_vel_b,
        root_ang_vel_b=root_ang_vel_b,
        root_link_quat_wxyz=root_link_quat_wxyz,
        gravity_vec_w=gravity_vec_w,
        command=command,
        joint_pos=joint_pos,
        default_joint_pos=default_joint_pos_policy,
        joint_vel=joint_vel,
        default_joint_vel=default_joint_vel,
        previous_action=previous_action_policy,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sim2Sim observation alignment test helper")
    parser.add_argument("--scene", type=str, default="WAVEGO_mujoco/scene.xml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/model_2999.pt",
    )
    parser.add_argument(
        "--env-yaml",
        type=str,
        default="logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/params/env.yaml",
    )
    parser.add_argument(
        "--io-descriptor",
        type=str,
        default="tmp/io_descriptors_wavego/isaac_velocity_flat_wavego_v0_IO_descriptors.yaml",
    )
    parser.add_argument("--cmd-x", type=float, default=0.5)
    parser.add_argument("--cmd-y", type=float, default=0.0)
    parser.add_argument("--cmd-wz", type=float, default=0.0)
    parser.add_argument("--use-heading-hold", action="store_true")
    parser.add_argument("--heading-target-yaw", type=float, default=0.0)
    parser.add_argument("--heading-kp", type=float, default=0.5)
    parser.add_argument("--heading-wz-limit", type=float, default=1.0)
    parser.add_argument("--action-scale", type=float, default=None)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument(
        "--action-lpf-alpha",
        type=float,
        default=1.0,
        help="Action low-pass alpha in (0,1]. 1.0 means disabled; smaller gives smoother actions.",
    )
    parser.add_argument(
        "--diagnostic-window",
        type=int,
        default=100,
        help="Window steps for periodic diagnostic summary print.",
    )
    parser.add_argument(
        "--stand-assist-gain",
        type=float,
        default=0.0,
        help="Blend actions toward zero when command is near zero. 0 disables, typical 0.05-0.25.",
    )
    parser.add_argument(
        "--stand-assist-threshold",
        type=float,
        default=0.05,
        help="Command norm threshold for activating stand assist.",
    )
    parser.add_argument("--print-full-obs", action="store_true")
    parser.add_argument("--print-full-action", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not (0.0 < args.action_lpf_alpha <= 1.0):
        raise ValueError("--action-lpf-alpha must be in (0, 1].")
    if not (0.0 <= args.stand_assist_gain < 1.0):
        raise ValueError("--stand-assist-gain must be in [0, 1).")
    model = mujoco.MjModel.from_xml_path(args.scene)
    mj_joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)]

    io_cfg = load_policy_io_config(args.io_descriptor)
    policy_to_mj_idx = _policy_to_mj_index(io_cfg.policy_joint_names, mj_joint_names)
    default_joint_pos_mj = reorder_policy_to_mj(io_cfg.default_joint_pos_policy, policy_to_mj_idx)

    isaac_params = load_isaac_params_from_env_yaml(args.env_yaml, default_joint_pos_mj=default_joint_pos_mj)

    policy_update_interval = sync_mujoco_timing(model, isaac_params)
    verify_and_sync_actuator_params(model, isaac_params)

    data = mujoco.MjData(model)
    validate_mujoco_joint_actuator_order(model)
    print_joint_mapping(model)

    default_joint_pos_mj = isaac_params.default_joint_pos_mj.copy()
    default_joint_pos_policy = io_cfg.default_joint_pos_policy.copy()
    previous_action_policy = np.zeros(NUM_ACTIONS, dtype=np.float64)
    base_command = np.array([args.cmd_x, args.cmd_y, args.cmd_wz], dtype=np.float64)
    command = base_command.copy()
    action_scale = io_cfg.action_scale if args.action_scale is None else args.action_scale

    control_dt = isaac_params.sim_dt * policy_update_interval

    data.qpos[2] = 0.25
    data.qpos[7:] = default_joint_pos_mj
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    normalizer = load_actor_obs_normalizer_from_checkpoint(args.checkpoint)
    actor = ActorMLP(args.checkpoint)

    def one_policy_step(command_local: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_local = extract_policy_observation_from_mujoco(
            model=model,
            data=data,
            default_joint_pos_policy=default_joint_pos_policy,
            previous_action_policy=previous_action_policy,
            command=command_local,
            policy_to_mj_idx=policy_to_mj_idx,
        )
        obs_norm_local = apply_obs_normalization(obs_local, normalizer)
        action_local = actor(obs_norm_local)
        if not np.all(np.isfinite(obs_local)):
            raise ValueError("Observation contains non-finite values.")
        if not np.all(np.isfinite(action_local)):
            raise ValueError("Action contains non-finite values.")
        return obs_local, obs_norm_local, action_local

    obs, obs_norm, action = one_policy_step(command)

    print("SIM2SIM_OBS_READY")
    print("SYNC_SUMMARY")
    print(f"mujoco_timestep={model.opt.timestep:.6f} (isaac_dt={isaac_params.sim_dt:.6f})")
    print(
        f"policy_update_interval={policy_update_interval} -> control_dt={control_dt:.6f}s "
        f"(control_hz={1.0 / control_dt:.2f})"
    )
    print(
        "actuator_params: "
        f"kp={model.actuator_gainprm[0,0]:.3f}, "
        f"kv={-model.actuator_biasprm[0,2]:.3f}, "
        f"forcerange=[{model.actuator_forcerange[0,0]:.3f}, {model.actuator_forcerange[0,1]:.3f}]"
    )
    print(f"command=[{command[0]:.3f}, {command[1]:.3f}, {command[2]:.3f}] (非静止需 cmd-x/cmd-y/cmd-wz 全为 0 才是静止)")
    print(f"action_scale={action_scale:.6f}")
    print(f"action_lpf_alpha={args.action_lpf_alpha:.3f} (1.0=disabled)")
    print(
        f"heading_hold={args.use_heading_hold}, heading_target_yaw={args.heading_target_yaw:.3f}, "
        f"heading_kp={args.heading_kp:.3f}, heading_wz_limit={args.heading_wz_limit:.3f}"
    )
    print(
        f"stand_assist_gain={args.stand_assist_gain:.3f}, "
        f"stand_assist_threshold={args.stand_assist_threshold:.3f}"
    )
    print(f"normalizer_loaded: mean_shape={normalizer.mean.shape}, std_shape={normalizer.std.shape}")
    print(
        "normalizer_head: "
        f"mean[:4]={np.array2string(normalizer.mean[:4], precision=6)}, "
        f"std[:4]={np.array2string(normalizer.std[:4], precision=6)}"
    )
    print(f"policy_joint_order(from IO)={','.join(io_cfg.policy_joint_names)}")
    print(f"mujoco_joint_order={','.join(mj_joint_names)}")
    print(f"obs_dim={obs.shape[0]}, obs_order_terms={len(OBS_ORDER)}")
    print(f"obs_norm_min={obs_norm.min():.6f}, obs_norm_max={obs_norm.max():.6f}")
    print(f"action_min={action.min():.6f}, action_max={action.max():.6f}, action_l2={np.linalg.norm(action):.6f}")

    print("STATIC_TEST_OBS_HEAD")
    print(np.array2string(obs[:16], precision=6, suppress_small=False))
    print("STATIC_TEST_ACTION_HEAD")
    print(np.array2string(action[:12], precision=6, suppress_small=False))

    if args.print_full_obs:
        print("STATIC_TEST_OBS_FULL")
        print(np.array2string(obs, precision=6, suppress_small=False, max_line_width=200))
    if args.print_full_action:
        print("STATIC_TEST_ACTION_FULL")
        print(np.array2string(action, precision=6, suppress_small=False, max_line_width=200))

    if args.steps <= 1:
        return

    print(f"ROLLOUT_START steps={args.steps}, visualize={args.visualize}, real_time={args.real_time}")

    try:
        import mujoco.viewer as mj_viewer
    except Exception:
        mj_viewer = None

    viewer_ctx = mj_viewer.launch_passive(model, data) if args.visualize and mj_viewer is not None else None

    next_wall_time = time.perf_counter()
    step_count = 0
    diagnostics = RolloutDiagnostics()
    diagnostics_window = RolloutDiagnostics()
    action_filtered_policy = previous_action_policy.copy()

    while step_count < args.steps:
        quat = np.asarray(data.qpos[3:7], dtype=np.float64)
        w, x, y, z = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = float(np.arctan2(siny_cosp, cosy_cosp))

        if args.use_heading_hold:
            yaw_err = args.heading_target_yaw - yaw
            yaw_err = (yaw_err + np.pi) % (2.0 * np.pi) - np.pi
            cmd_wz = np.clip(args.heading_kp * yaw_err, -args.heading_wz_limit, args.heading_wz_limit)
            command = np.array([base_command[0], base_command[1], cmd_wz], dtype=np.float64)
        else:
            command = base_command.copy()

        obs, obs_norm, action = one_policy_step(command)

        action_filtered_policy = (
            args.action_lpf_alpha * action + (1.0 - args.action_lpf_alpha) * action_filtered_policy
        )

        if np.linalg.norm(command) < args.stand_assist_threshold and args.stand_assist_gain > 0.0:
            action_filtered_policy = (1.0 - args.stand_assist_gain) * action_filtered_policy

        action_mj = reorder_policy_to_mj(action_filtered_policy, policy_to_mj_idx)
        target_pos = action_mj * action_scale + default_joint_pos_mj
        data.ctrl[:] = target_pos
        prev_action_before_update = previous_action_policy.copy()
        previous_action_policy[:] = action_filtered_policy

        for _ in range(policy_update_interval):
            mujoco.mj_step(model, data)
            if viewer_ctx is not None:
                viewer_ctx.sync()

        step_count += 1

        quat = np.asarray(data.qpos[3:7], dtype=np.float64)
        # Extract roll/pitch via quaternion -> Euler decomposition.
        w, x, y, z = quat
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = float(np.arctan2(sinr_cosp, cosr_cosp))
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = float(np.sign(sinp) * (np.pi / 2.0))
        else:
            pitch = float(np.arcsin(sinp))

        diagnostics.update(
            obs=obs,
            action=action_filtered_policy,
            action_prev=prev_action_before_update,
            command=command,
            base_z=float(data.qpos[2]),
            roll_deg=np.degrees(roll),
            pitch_deg=np.degrees(pitch),
        )
        diagnostics_window.update(
            obs=obs,
            action=action_filtered_policy,
            action_prev=prev_action_before_update,
            command=command,
            base_z=float(data.qpos[2]),
            roll_deg=np.degrees(roll),
            pitch_deg=np.degrees(pitch),
        )

        if step_count % args.print_every == 0 or step_count == args.steps:
            base_height = float(data.qpos[2])
            vel_x = float(obs[0])
            vel_y = float(obs[1])
            act_l2 = float(np.linalg.norm(action_filtered_policy))
            print(
                f"rollout_step={step_count} time={data.time:.3f} base_z={base_height:.3f} "
                f"vel_x={vel_x:.3f} vel_y={vel_y:.3f} action_l2={act_l2:.3f}"
            )

        if step_count % args.diagnostic_window == 0 or step_count == args.steps:
            print(f"window@{step_count}: {diagnostics_window.summary_str()}")
            diagnostics_window = RolloutDiagnostics()

        if args.real_time:
            next_wall_time += control_dt
            now = time.perf_counter()
            sleep_time = next_wall_time - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_wall_time = now

    if viewer_ctx is not None:
        viewer_ctx.close()

    print(diagnostics.summary_str())
    print("ROLLOUT_DONE")


if __name__ == "__main__":
    main()