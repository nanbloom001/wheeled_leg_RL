import mujoco
import mujoco.viewer
import torch
import numpy as np
import time
import argparse
import csv
from contextlib import nullcontext
from scipy.spatial.transform import Rotation as R

# ==========================================
#                配置参数
# ==========================================
MODEL_PATH = "WAVEGO_mujoco/scene.xml"
POLICY_PATH = "logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/model_2999.pt"

ACTION_SCALE = 0.25
ISAAC_DT = 0.005  # Isaac Lab 中设置的 sim.dt
DECIMATION = 4    # 控制频率降频系数 (50Hz control)
CONTROL_DT = ISAAC_DT * DECIMATION  # 0.02s = 50Hz

NUM_OBS = 48
NUM_ACTIONS = 12
COMMAND = np.array([0.5, 0.0, 0.0])  # 速度命令: [lin_x, lin_y, ang_z]
PRINT_EVERY = 50
DETAIL_EVERY = 500
ACTION_JITTER_WINDOW = 200

OBS_LABELS = (
    [f"base_lin_vel[{i}]" for i in range(3)]
    + [f"base_ang_vel[{i}]" for i in range(3)]
    + [f"proj_grav[{i}]" for i in range(3)]
    + ["cmd_lin_x", "cmd_lin_y", "cmd_ang_z"]
    + [f"joint_pos_rel[{i}]" for i in range(12)]
    + [f"joint_vel[{i}]" for i in range(12)]
    + [f"last_action[{i}]" for i in range(12)]
)

# ✅ 修正：站立姿态按深度优先顺序排列 (FL, FR, RL, RR) x (hip, thigh, calf)
# 从 wavego.py 的 init_state.joint_pos 转换而来
STANDING_POSE = np.array([
    0.100, -0.650,  0.600,   # FL: hip, thigh, calf
   -0.100,  0.650, -0.600,   # FR: hip, thigh, calf
   -0.100, -0.650,  0.600,   # RL: hip, thigh, calf
    0.100,  0.650, -0.600    # RR: hip, thigh, calf
])

# ✅ 修正：Isaac Lab 和 MuJoCo 都使用深度优先遍历的关节顺序，无需映射
# Isaac Lab joint order (verified from USD traversal):
# [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]
# MuJoCo joint order (from qpos[7:]):
# [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]
# => Mapping is IDENTITY!

EXPECTED_JOINT_ORDER = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

EXPECTED_ACTUATOR_ORDER = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]


def validate_model_order(model):
    mujoco_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(1, model.njnt)
    ]
    mujoco_actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]

    if mujoco_joint_names != EXPECTED_JOINT_ORDER:
        print("\n[ERROR] 关节顺序不匹配！")
        print(f"Expected: {EXPECTED_JOINT_ORDER}")
        print(f"Actual:   {mujoco_joint_names}")
        raise RuntimeError("MuJoCo joint order mismatch")

    if mujoco_actuator_names != EXPECTED_ACTUATOR_ORDER:
        print("\n[ERROR] 执行器顺序不匹配！")
        print(f"Expected: {EXPECTED_ACTUATOR_ORDER}")
        print(f"Actual:   {mujoco_actuator_names}")
        raise RuntimeError("MuJoCo actuator order mismatch")

    print("[OK] 关节顺序与执行器顺序校验通过（与 Isaac 训练定义一致）")

def get_obs(model, data, last_action, command):
    """构建观测向量 (48维): [lin_vel(3), ang_vel(3), gravity(3), cmd(3), joint_pos(12), joint_vel(12), last_action(12)]"""
    
    # ✅ 修正：WAVEGO 坐标系 = Isaac Lab body 坐标系 (identity rotation)
    # 无需坐标变换，直接使用 MuJoCo body 坐标系下的观测
    quat = data.qpos[3:7]  # MuJoCo [w, x, y, z]
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy [x,y,z,w]
    inv_rot = r.inv()

    # 使用 MuJoCo 官方 API 获取 base 的 6D 速度，避免 qvel 语义歧义
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    vel6 = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_id, vel6, 1)
    ang_vel_b = vel6[:3]
    lin_vel_b = vel6[3:]
    proj_grav = inv_rot.apply(np.array([0, 0, -1.0]))

    # ✅ 修正：关节状态无需映射（identity mapping）
    joint_pos = data.qpos[7:]   # shape: (12,)
    joint_vel = data.qvel[6:]   # shape: (12,)
    obs_joint_pos = joint_pos - STANDING_POSE  # 相对于默认姿态的偏移
    
    obs = np.concatenate([
        lin_vel_b, ang_vel_b, proj_grav, command,
        obs_joint_pos, joint_vel, last_action
    ])
    return obs, lin_vel_b[0]  # 返回前进速度用于监控


def parse_args():
    parser = argparse.ArgumentParser(description="WAVEGO Sim2Sim (MuJoCo) with diagnostics")
    parser.add_argument("--cmd-x", type=float, default=float(COMMAND[0]))
    parser.add_argument("--cmd-y", type=float, default=float(COMMAND[1]))
    parser.add_argument("--cmd-wz", type=float, default=float(COMMAND[2]))
    parser.add_argument(
        "--action-clip",
        type=float,
        default=None,
        help="可选：对策略原始输出 actions 先做 [-clip, clip] 截断，用于对齐/诊断",
    )
    parser.add_argument(
        "--torque-scale",
        type=float,
        default=1.0,
        help="可选：临时按比例放大 MuJoCo forcerange（仅诊断，不代表真实硬件）",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="最大步数，0 表示不限制")
    parser.add_argument("--headless", action="store_true", help="无GUI模式（适合批量测试）")
    parser.add_argument("--no-real-time", action="store_true", help="关闭墙钟同步，尽快跑完（用于离线诊断）")
    parser.add_argument("--joint-damping", type=float, default=None, help="运行时覆盖关节damping（12个关节统一）")
    parser.add_argument("--joint-frictionloss", type=float, default=None, help="运行时覆盖关节frictionloss（12个关节统一）")
    parser.add_argument("--joint-armature", type=float, default=None, help="运行时覆盖关节armature（12个关节统一）")
    parser.add_argument("--actuator-kp", type=float, default=None, help="运行时覆盖 position actuator kp（12个执行器统一）")
    parser.add_argument("--actuator-kv", type=float, default=None, help="运行时覆盖 position actuator kv（12个执行器统一）")
    parser.add_argument("--floor-friction", type=float, default=None, help="运行时覆盖地面主摩擦系数")
    parser.add_argument("--reset-on-fall", action="store_true", help="跌倒后自动重置，避免长时间发散干扰定位")
    parser.add_argument("--fall-height", type=float, default=0.11, help="判定跌倒的最小 base 高度")
    parser.add_argument("--fall-pitch-deg", type=float, default=70.0, help="判定跌倒的最大 pitch 绝对值（度）")
    parser.add_argument("--log-csv", type=str, default="", help="可选：将每步关键指标写入 CSV")
    return parser.parse_args()

def main():
    args = parse_args()
    command = np.array([args.cmd_x, args.cmd_y, args.cmd_wz], dtype=np.float64)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    model.opt.timestep = ISAAC_DT

    joint_slice = slice(6, 18)
    if args.joint_damping is not None:
        model.dof_damping[joint_slice] = args.joint_damping
        print(f"[INFO] 覆盖 joint damping = {args.joint_damping:.4f}")
    if args.joint_frictionloss is not None:
        model.dof_frictionloss[joint_slice] = args.joint_frictionloss
        print(f"[INFO] 覆盖 joint frictionloss = {args.joint_frictionloss:.4f}")
    if args.joint_armature is not None:
        model.dof_armature[joint_slice] = args.joint_armature
        print(f"[INFO] 覆盖 joint armature = {args.joint_armature:.4f}")
    if args.actuator_kp is not None:
        model.actuator_gainprm[:, 0] = args.actuator_kp
        model.actuator_biasprm[:, 1] = -args.actuator_kp
        print(f"[INFO] 覆盖 actuator kp = {args.actuator_kp:.4f}")
    if args.actuator_kv is not None:
        model.actuator_biasprm[:, 2] = -args.actuator_kv
        print(f"[INFO] 覆盖 actuator kv = {args.actuator_kv:.4f}")
    if args.floor_friction is not None:
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id >= 0:
            model.geom_friction[floor_id, 0] = args.floor_friction
            print(f"[INFO] 覆盖 floor friction = {args.floor_friction:.4f}")
        else:
            print("[WARN] 未找到 floor geom，忽略 floor-friction 覆盖")

    if args.torque_scale != 1.0:
        model.actuator_forcerange[:, 0] *= args.torque_scale
        model.actuator_forcerange[:, 1] *= args.torque_scale
        print(f"[WARN] 启用诊断模式 torque-scale={args.torque_scale:.2f} (仅用于定位，不代表硬件现实)")

    data = mujoco.MjData(model)
    validate_model_order(model)
    if args.action_clip is not None:
        print(f"[INFO] 启用动作裁剪: [-{args.action_clip:.3f}, +{args.action_clip:.3f}]")
    print(f"[INFO] 命令输入: [{command[0]:+.3f}, {command[1]:+.3f}, {command[2]:+.3f}]")
    print(f"[INFO] 运行模式: {'headless' if args.headless else 'viewer'} | {'no-real-time' if args.no_real_time else 'real-time'}")
    if np.allclose(command, 0.0):
        print("[WARN] 当前为零命令测试；训练中 standing 比例较低，零命令可能是弱覆盖分布")
    if args.headless and args.max_steps <= 0:
        print("[WARN] headless 模式下未设置 max-steps，自动设为 1000")
        args.max_steps = 1000

    checkpoint = torch.load(POLICY_PATH, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(NUM_OBS, 512), torch.nn.ELU(),
                torch.nn.Linear(512, 256), torch.nn.ELU(),
                torch.nn.Linear(256, 128), torch.nn.ELU(),
                torch.nn.Linear(128, NUM_ACTIONS)
            )
        def forward(self, x): return self.net(x)

    policy = Actor().to(device)
    clean_dict = {k.replace('actor.', 'net.'): v for k, v in state_dict.items() if 'actor.0' in k or 'actor.2' in k or 'actor.4' in k or 'actor.6' in k}
    policy.load_state_dict(clean_dict)
    policy.eval()

    obs_mean = state_dict['actor_obs_normalizer._mean'].cpu().numpy().flatten()
    obs_std = state_dict['actor_obs_normalizer._std'].cpu().numpy().flatten()

    last_action = np.zeros(NUM_ACTIONS)
    action_delta_hist = []
    wall_period_hist = []
    sim_period_hist = []
    count = 0
    reset_count = 0

    csv_file = None
    csv_writer = None
    if args.log_csv:
        csv_file = open(args.log_csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "step", "sim_time", "vel_x", "base_height", "pitch_deg", "joint_err_mean", "joint_err_max",
            "torque_max", "sat_ratio", "dact_norm", "obs_max", "clip_dims", "sim_hz", "wall_hz", "resets"
        ])

    viewer_ctx = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(model, data)

    with viewer_ctx as viewer:
        # 初始化: 站立姿态 + 0.25m 离地高度
        data.qpos[2] = 0.25
        data.qpos[7:] = STANDING_POSE.copy()  # ✅ 直接赋值（无需映射）
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        
        next_control_wall = time.perf_counter()
        last_control_wall = None
        last_control_sim = data.time

        while True:
            if (not args.headless) and (not viewer.is_running()):
                break

            now = time.perf_counter()
            if (not args.no_real_time) and (now < next_control_wall):
                time.sleep(next_control_wall - now)

            control_wall_now = time.perf_counter()
            wall_dt = CONTROL_DT if last_control_wall is None else (control_wall_now - last_control_wall)
            last_control_wall = control_wall_now
            wall_period_hist.append(wall_dt)
            if len(wall_period_hist) > ACTION_JITTER_WINDOW:
                wall_period_hist.pop(0)

            # 1. 构建观测
            obs_raw, vel_x = get_obs(model, data, last_action, command)
            obs_norm = (obs_raw - obs_mean) / (obs_std + 1e-8)
            obs_norm = np.clip(obs_norm, -5.0, 5.0)  # ✅ 添加 clipping (EmpiricalNormalization 标准)
            clip_mask = np.isclose(np.abs(obs_norm), 5.0)
            clipped_dims = int(np.sum(clip_mask))
            
            # 2. 策略推理
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs_norm).float().to(device).unsqueeze(0)
                actions = policy(obs_tensor).squeeze().cpu().numpy()

            if args.action_clip is not None:
                actions = np.clip(actions, -args.action_clip, args.action_clip)

            action_delta = actions - last_action
            action_delta_hist.append(float(np.linalg.norm(action_delta)))
            if len(action_delta_hist) > ACTION_JITTER_WINDOW:
                action_delta_hist.pop(0)
            
            # 3. 动作解码 + 执行
            target_pos = actions * ACTION_SCALE + STANDING_POSE  # ✅ 无需映射
            data.ctrl[:] = target_pos
            last_action = actions.copy()

            for _ in range(DECIMATION):
                mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()

            sim_dt = data.time - last_control_sim
            last_control_sim = data.time
            sim_period_hist.append(sim_dt)
            if len(sim_period_hist) > ACTION_JITTER_WINDOW:
                sim_period_hist.pop(0)

            if not args.no_real_time:
                next_control_wall += CONTROL_DT
                if control_wall_now > next_control_wall + CONTROL_DT:
                    next_control_wall = control_wall_now + CONTROL_DT

            count += 1
            if count % PRINT_EVERY == 0:
                # 计算关节跟踪误差
                joint_error = np.abs(data.qpos[7:] - data.ctrl[:])
                max_error = np.max(joint_error)
                mean_error = np.mean(joint_error)
                
                # 计算实际施加的力矩
                qfrc_actuator = data.qfrc_actuator[6:18]  # 跳过 free-joint 的6个自由度
                max_torque = np.max(np.abs(qfrc_actuator))
                torque_limit = float(np.abs(model.actuator_forcerange[0, 1]))
                sat_ratio = float(np.mean(np.abs(qfrc_actuator) >= (0.98 * torque_limit)))

                avg_action_delta = float(np.mean(action_delta_hist)) if action_delta_hist else 0.0
                avg_wall_dt = float(np.mean(wall_period_hist)) if wall_period_hist else CONTROL_DT
                avg_sim_dt = float(np.mean(sim_period_hist)) if sim_period_hist else CONTROL_DT
                wall_ctrl_hz = 1.0 / max(avg_wall_dt, 1e-6)
                sim_ctrl_hz = 1.0 / max(avg_sim_dt, 1e-6)
                
                # 提取关键观测
                base_height = data.qpos[2]
                pitch = np.arctan2(2*(data.qpos[3]*data.qpos[5] + data.qpos[4]*data.qpos[6]), 
                                    1 - 2*(data.qpos[5]**2 + data.qpos[6]**2))
                
                print(f"[Step {count:5d}] VelX: {vel_x:+.3f} m/s | Height: {base_height:.3f}m | "
                      f"Pitch: {pitch*180/np.pi:+.1f}° | "
                      f"JointErr: {mean_error:.3f}/{max_error:.3f} rad | "
                      f"Torque: {max_torque:.2f}/{torque_limit:.2f} Nm | Sat: {sat_ratio*100:.0f}% | "
                      f"|dAct|: {avg_action_delta:.3f} | SimHz: {sim_ctrl_hz:.1f} | WallHz: {wall_ctrl_hz:.1f} | "
                      f"ObsMax: {np.max(np.abs(obs_norm)):.2f} | ClipDims: {clipped_dims}")

                if csv_writer is not None:
                    csv_writer.writerow([
                        count,
                        data.time,
                        vel_x,
                        base_height,
                        pitch * 180 / np.pi,
                        mean_error,
                        max_error,
                        max_torque,
                        sat_ratio,
                        avg_action_delta,
                        np.max(np.abs(obs_norm)),
                        clipped_dims,
                        sim_ctrl_hz,
                        wall_ctrl_hz,
                        reset_count,
                    ])
                
                # 每500步输出详细信息
                if count % DETAIL_EVERY == 0:
                    print(f"\n{'='*80}")
                    print(f"详细诊断 (Step {count}):")
                    print(f"{'='*80}")
                    print(f"策略输出 (actions): {actions}")
                    print(f"动作变化 (actions-last): {action_delta}")
                    print(f"目标位置 (ctrl):    {data.ctrl[:]}")
                    print(f"实际位置 (qpos):    {data.qpos[7:]}")
                    print(f"位置误差 (error):   {data.qpos[7:] - data.ctrl[:]}")
                    print(f"obs前12维:         {obs_raw[:12]}")
                    if clipped_dims > 0:
                        clipped_idx = np.where(clip_mask)[0].tolist()
                        clipped_names = [OBS_LABELS[i] for i in clipped_idx]
                        print(f"clip维度索引:      {clipped_idx}")
                        print(f"clip维度名称:      {clipped_names}")
                    print(f"实际力矩 (qfrc):    {qfrc_actuator}")
                    print(f"关节速度 (qvel):    {data.qvel[6:]}")
                    print(f"Base 速度: lin={data.qvel[:3]}, ang={data.qvel[3:6]}")
                    print(f"仿真控制周期: {avg_sim_dt:.4f}s (目标 {CONTROL_DT:.4f}s)")
                    print(f"墙钟控制周期: {avg_wall_dt:.4f}s (目标 {CONTROL_DT:.4f}s)")
                    print(f"仿真控制频率: {sim_ctrl_hz:.2f} Hz (目标 {1.0/CONTROL_DT:.2f} Hz)")
                    print(f"墙钟控制频率: {wall_ctrl_hz:.2f} Hz (目标 {1.0/CONTROL_DT:.2f} Hz)")
                    print(f"{'='*80}\n")

            if args.reset_on_fall:
                base_height = data.qpos[2]
                pitch_deg = abs(np.arctan2(
                    2 * (data.qpos[3] * data.qpos[5] + data.qpos[4] * data.qpos[6]),
                    1 - 2 * (data.qpos[5] ** 2 + data.qpos[6] ** 2),
                ) * 180 / np.pi)
                if (base_height < args.fall_height) or (pitch_deg > args.fall_pitch_deg):
                    reset_count += 1
                    print(
                        f"[RESET {reset_count}] fall detected: height={base_height:.3f}, pitch={pitch_deg:.1f}deg"
                    )
                    data.qpos[:] = 0.0
                    data.qvel[:] = 0.0
                    data.qpos[2] = 0.25
                    data.qpos[3] = 1.0
                    data.qpos[7:] = STANDING_POSE.copy()
                    data.ctrl[:] = STANDING_POSE.copy()
                    mujoco.mj_forward(model, data)
                    last_action[:] = 0.0
                    next_control_wall = time.perf_counter() + CONTROL_DT
                    last_control_sim = data.time

            if args.max_steps > 0 and count >= args.max_steps:
                break

    if csv_file is not None:
        csv_file.close()

if __name__ == "__main__": main()
