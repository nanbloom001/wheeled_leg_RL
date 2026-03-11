# WAVEGO IsaacLab → MuJoCo Sim2Sim 信息采集清单（重新校验版）

> 目标：仅基于仓库与日志中已存在文件提取信息。  
> 原则：查不到就留白，不做推测。  
> 说明：本版本**不依赖旧的 `scripts/sim2sim_mujoco.py` 实现细节**，用于你“从头重建 sim2sim”的基线资料。

## 0) 信息来源（已核对）

- `source/isaaclab_assets/isaaclab_assets/robots/wavego.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/wavego/flat_env_cfg.py`
- `source/isaaclab/isaaclab/envs/mdp/observations.py`
- `source/isaaclab/isaaclab/assets/rigid_object/rigid_object_data.py`
- `logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/params/env.yaml`
- `logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/params/agent.yaml`
- `logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/model_2999.pt`
- `logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/exported/policy.pt`
- `logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/exported/policy.onnx`
- `tmp/io_descriptors_wavego/isaac_velocity_flat_wavego_v0_IO_descriptors.yaml`
- `tmp/mujoco_compiled_body_mass_inertia.csv`
- `tmp/mujoco_compiled_summary.txt`
- `tmp/urdf_vs_mujoco_mass_inertia_compare.csv`
- `tmp/urdf_vs_mujoco_mass_inertia_compare.txt`
- `WAVEGO_description/WAVEGO_description/urdf/WAVEGO.urdf`
- `WAVEGO_mujoco/wavego.xml`
- `WAVEGO_mujoco/scene.xml`

---

## 1) 仿真时序参数

| 项目 | 已提取值 | 来源 |
|---|---|---|
| Isaac `sim.dt` | `0.005` s | `env.yaml`, `wavego.py` |
| Isaac `decimation` | `4` | `env.yaml`, `flat_env_cfg.py` |
| 控制频率（推算） | `1 / (0.005 * 4) = 50 Hz` | 由上两项推算 |
| `episode_length_s` | `20.0` s | `env.yaml` |
| MuJoCo `timestep` | `0.005` s | `wavego.xml` |

备注：Isaac 与 MuJoCo 的基础步长已对齐（均为 5ms）。

---

## 2) 策略网络与权重

| 项目 | 已提取值 | 来源 |
|---|---|---|
| Checkpoint | `model_2999.pt`（同时存在多轮 `model_*.pt`） | `logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/` |
| 导出模型 | `exported/policy.pt`, `exported/policy.onnx` | 同上 |
| Policy 类型 | `ActorCritic`（PPO） | `agent.yaml` |
| Actor 隐层 | `[512, 256, 128]` | `agent.yaml` + checkpoint shape |
| Critic 隐层 | `[512, 256, 128]` | `agent.yaml` + checkpoint shape |
| 激活函数 | `elu` | `agent.yaml` |
| 是否 RNN/LSTM/GRU | 未见相关配置或权重键（按现有文件判断：无） | `agent.yaml`, checkpoint key |
| obs normalizer 开关 | `actor_obs_normalization=true`, `critic_obs_normalization=true` | `agent.yaml` |
| `empirical_normalization` | `false`（训练器配置项） | `agent.yaml` |
| `obs_groups` | `{}`（未显式提供独立 critic 观测组） | `agent.yaml` |
| 归一化统计量（checkpoint内） | 存在：`actor_obs_normalizer._mean/_var/_std/count`、`critic_obs_normalizer.*` | `model_2999.pt` |

### 2.1 checkpoint 中已提取的关键数值（actor 观测归一化）

- `obs_dim`: `48`
- `actor_obs_normalizer.count`: `589824000`
- `actor_obs_mean`: 已提取（48维，见附录 A）
- `actor_obs_std`: 已提取（48维，见附录 A）

### 2.2 动作噪声参数（checkpoint）

- 键：`std`，shape=`[12]`
- 当前值（迭代 2999）：见附录 A

---

## 3) 观测空间（Observation）

### 3.1 观测项与顺序（policy）

根据 `env.yaml` 中 `observations.policy`，拼接顺序为：

1. `base_lin_vel` (3)
2. `base_ang_vel` (3)
3. `projected_gravity` (3)
4. `velocity_commands` (3)
5. `joint_pos` (`joint_pos_rel`) (12)
6. `joint_vel` (`joint_vel_rel`) (12)
7. `actions` (`last_action`) (12)

总维度：`48`（与 checkpoint 第一层权重 `actor.0.weight = [512, 48]` 一致）。

### 3.2 噪声、clip、scale（训练侧）

| 项目 | 已提取值 | 来源 |
|---|---|---|
| `base_lin_vel` 噪声 | `Uniform[-0.1, 0.1]` | `env.yaml` |
| `base_ang_vel` 噪声 | `Uniform[-0.2, 0.2]` | `env.yaml` |
| `projected_gravity` 噪声 | `Uniform[-0.05, 0.05]` | `env.yaml` |
| `joint_pos_rel` 噪声 | `Uniform[-0.03, 0.03]` | `env.yaml` |
| `joint_vel_rel` 噪声 | `Uniform[-0.5, 0.5]` | `env.yaml` |
| 观测项 `clip` | 各项均为 `null`（训练配置里未设 term clip） | `env.yaml` |
| 观测项 `scale` | 各项均为 `null` | `env.yaml` |

### 3.3 从零重建推理端时应留白的实现项（当前未绑定）

以下是你新 sim2sim 实现里必须自行定义并文档化的项，当前仓库文件无法直接给出“唯一正确值”：

- 观测在推理端是否做 normalize（以及公式）
- 推理端 `obs clip` 是否启用、区间为何
- 基座线速度/角速度在 MuJoCo 端的取值 API 与参考系定义

### 3.4 观测计算细节文档（本次新增）

已新增独立文档：`WAVEGO_ISAACLAB_OBS_COMPUTATION.md`，包含：

- 48维 policy 观测的逐项公式与代码入口
- `projected_gravity` 的底层计算路径（`quat_apply_inverse(root_link_quat_w, GRAVITY_VEC_W)`）
- 噪声/clip/scale 的训练配置实值
- 运行时 IO descriptors 交叉验证结果

---

## 4) 动作空间

| 项目 | 已提取值 | 来源 |
|---|---|---|
| action 维度 | `12` | `agent.yaml` / checkpoint |
| 动作类型 | 关节位置动作（`JointPositionAction`） | `env.yaml` |
| `action_scale` | `0.25` | `env.yaml`, `flat_env_cfg.py` |
| `use_default_offset` | `true` | `env.yaml` |
| 动作噪声（训练） | `Uniform[-0.02, 0.02]` | `env.yaml` |
| 动作 clip（训练配置） | `null` | `env.yaml`, `agent.yaml(clip_actions=null)` |

默认站立姿态（12关节）在训练配置中可由以下字段获得：

- `wavego.py` 的 `init_state.joint_pos`
- `env.yaml` 的 `scene.robot.init_state.joint_pos`

---

## 5) 执行器与关节参数

### 5.1 IsaacLab 训练使用（已提取）

| 项目 | 值 | 来源 |
|---|---|---|
| 执行器类型 | `ImplicitActuator` | `env.yaml` / `wavego.py` |
| 关节匹配 | `.*_hip_joint`, `.*_thigh_joint`, `.*_calf_joint` | 同上 |
| `stiffness (kp)` | `400.0` | 同上 |
| `damping (kd)` | `10.0` | 同上 |
| `effort_limit_sim` | `1.96` | 同上 |
| `velocity_limit_sim` | `11.1` | 同上 |
| `armature` | `0.01` | 同上 |
| `friction` | `0.4` | 同上 |

### 5.2 MuJoCo 当前模型（已提取）

| 项目 | 值 | 来源 |
|---|---|---|
| `<position kp>` | `400` | `wavego.xml` |
| `<position kv>` | `10` | 同上 |
| `<position forcerange>` | `[-1.96, 1.96]` | 同上 |
| `<joint damping>` | `0.0`（default） | 同上 |
| `<joint armature>` | `0.01`（default） | 同上 |
| `<joint frictionloss>` | `0.4`（default） | 同上 |

---

## 6) 关节顺序映射（高优先级）

### 6.1 实测 MuJoCo 编译结果（env_isaaclab）

通过 `mujoco==3.4.0` 加载 `scene.xml` 后得到：

- `joint_names`（去掉 freejoint）：
  `FL_hip_joint, FL_thigh_joint, FL_calf_joint, FR_hip_joint, FR_thigh_joint, FR_calf_joint, RL_hip_joint, RL_thigh_joint, RL_calf_joint, RR_hip_joint, RR_thigh_joint, RR_calf_joint`
- `actuator_names`：
  `FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf`

结论：当前 MJCF 的关节与执行器顺序在结构上是清晰且一致的。

### 6.2 留白待确认

- Isaac 端已通过 IO descriptors 运行时导出关节顺序（`joint_names`），与上述 12 关节顺序一致。
- “是否始终一致（跨版本/跨资产迭代）”仍建议在每次资产改动后重新导出一次并归档。

---

## 7) 机器人模型一致性（URDF / USD / MJCF）

### 7.1 本次校验与修正（已完成）

对 `WAVEGO_mujoco/wavego.xml` 做了 URDF 对账后，已修正以下确定性差异：

1. 12个关节 `range` 统一对齐 URDF（含 `FL/FR_calf` 从 `±1.57` 修正为 `±2.094395`）
2. `RL_hip` 的 body/geom 局部位姿对齐 URDF
3. `FR_calf` 的 body/geom 局部位姿对齐 URDF
4. `RL_thigh` 的 geom 局部位姿对齐 URDF

### 7.2 实测可运行性（已完成）

在 `env_isaaclab`（`mujoco 3.4.0`）中加载 `scene.xml`：

- `load_ok=True`
- `nq=19, nv=18, nu=12, njnt=13, nbody=14`
- `gravity=[0,0,-9.81]`
- `timestep=0.005`
- `key_qpos_shape=(1,19)`，`key_ctrl_shape=(1,12)`

结论：**当前 MJCF 可编译加载**，结构层面通过。

### 7.3 仍未闭合的准确性风险（留白）

1. **惯量可追溯性不足**
   - URDF 每个 link 明确给了质量/惯量。
   - 当前 MJCF 仅 base 明确写了 `<inertial mass="0.75" ...>`；其余 link 未显式写 inertial。
   - 这会导致“与 URDF 逐 link 质量/惯量严格一致性”当前无法直接确认。

2. **USD 与 MJCF 的动态参数逐项对账**
   - 目前尚未有自动导出的 USD↔MJCF 对账文件。

3. **URDF 与 MuJoCo 编译后惯量存在数值偏差（已实测）**
  - 已导出 MuJoCo 编译后每个 body 的 `mass/inertia`（`tmp/mujoco_compiled_body_mass_inertia.csv`）。
  - 已完成 URDF↔MuJoCo 逐 link 对账（`tmp/urdf_vs_mujoco_mass_inertia_compare.csv`，13/13 匹配）。
  - 汇总结果（`tmp/urdf_vs_mujoco_mass_inertia_compare.txt`）：
    - `max_abs_delta_mass=0.0747577270587`
    - `max_abs_delta_inertia_diag=0.004545`
  - 说明：当前 MJCF 未对所有 link 显式写 inertial，编译后惯量与 URDF 不完全一致。

---

## 8) 物理参数（影响 sim gap）

| 项目 | Isaac 训练值 | MuJoCo 当前值 | 备注 |
|---|---|---|---|
| 重力 | `[0,0,-9.81]` | 编译后实测 `[0,0,-9.81]` | 已对齐 |
| 地面摩擦 | `static=1.0, dynamic=1.0` | floor geom `friction="1.0 0.05 0.01"` | 参数模型不同，需等效标定 |
| 恢复系数 | `0.0` | `solref/solimp` 已设 | 接触模型不同，需标定 |
| 关节阻尼 | 执行器 `damping=10.0` | 执行器 `kv=10` + joint `damping=0.0` | 需按控制目标再确认 |

---

## 9) Domain Randomization（训练期）

| 项目 | 范围/配置 | 来源 |
|---|---|---|
| 摩擦随机化 | `static,dynamic ∈ [0.4,1.25]`，`restitution ∈ [0,0.1]` | `env.yaml events.physics_material` |
| 基座质量随机化 | `add [-0.1, 0.1]` kg | `events.add_base_mass` |
| 基座 CoM 随机化 | x/y/z `[-0.015,0.015]` m | `events.base_com` |
| Push 扰动 | 每 `10~15s`，速度 x/y `[-0.5,0.5]` | `events.push_robot` |
| 观测噪声 | 见第 3 节 | `observations.policy.*.noise` |
| 动作噪声 | `[-0.02,0.02]` | `actions.joint_pos.noise` |

---

## 10) 归一化与数值处理

| 项目 | 已提取值 | 来源 |
|---|---|---|
| obs normalizer 是否存在 | 存在（actor/critic） | `model_2999.pt` |
| obs mean/std | 已提取（48维） | `model_2999.pt` |
| 新 sim2sim 推理端 normalize 公式 | **留白（你将从头实现）** | - |
| 新 sim2sim 推理端 obs clip | **留白（你将从头实现）** | - |
| 新 sim2sim 推理端 action clip | **留白（你将从头实现）** | - |

---

## 11) 留白项（当前文件中未查到/无法直接确认）

1. USD vs URDF vs MJCF 的逐 link 质量/惯量逐项对账（当前仅完成 URDF↔MuJoCo）
2. 推理端 normalize/clip 的最终实现规范（这是新 sim2sim 设计决策，不是现有训练配置可唯一反推项）
3. 训练时是否存在 action delay / obs delay：配置文件未见显式项，但尚缺“逐算子级”运行时链路证明

---

## 12) 下一步最小补齐建议

1. ✅ 已完成：导出 IO descriptors（`tmp/io_descriptors_wavego/isaac_velocity_flat_wavego_v0_IO_descriptors.yaml`）
2. ✅ 已完成：通过 `play.py` 成功加载 `model_2999.pt` 并确认 policy obs/action 维度（48/12）与网络结构
3. ✅ 已完成：导出 MuJoCo 编译后 `body_mass/body_inertia` 并与 URDF 做逐 link 对账（见 `tmp/urdf_vs_mujoco_mass_inertia_compare.csv`）
4. 待做：若目标是“严格复刻惯量”，需在 MJCF 为各 link 显式设置 inertial（质量+惯量）并重新对账
5. 待做：对地面接触参数做一轮等效标定（摩擦 + 接触求解参数）

---

## 13) 本次全面性复核结论

以“重建 sim2sim 所需信息”划分：

- **已明确并可直接使用**：时序、观测定义与顺序、动作定义、关节顺序、策略结构、normalizer 统计量、训练随机化范围、MuJoCo 编译可运行性。
- **已明确且发现差异**：URDF 与 MuJoCo 编译后质量/惯量存在可量化偏差（见第 7.3）。
- **仍是实现决策或外部对账问题**：推理端 normalize/clip 最终策略、USD↔MJCF 动态参数逐项一致性。

结论：**当前资料已覆盖“从头重建并跑通 sim2sim”的核心必需信息；剩余项主要影响“高保真一致性”而非“能否搭建与运行”。**

---

## 附录 A：checkpoint 提取结果（`model_2999.pt`）

### A.1 actor_obs_normalizer

- `count`: `589824000`
- `obs_mean` (48):

```text
[0.0007, 0.0014, -0.0013, 0.0023, -0.0011, -0.0001, -0.0006, -0.0042, -0.9979, 0.0007, -0.0002, -0.0026, -0.016, 0.017, -0.0159, 0.034, 0.0139, -0.0018, -0.0143, 0.0019, -0.004, -0.0173, 0.0162, -0.032, 0.0175, -0.0138, -0.0022, 0.0045, -0.0305, 0.0342, 0.0054, -0.0074, 0.0033, 0.0001, 0.0141, -0.0148, 0.0113, 0.0124, -0.0836, 0.1694, 0.0003, 0.0484, -0.0646, 0.0117, -0.0865, 0.0098, -0.0052, -0.0526]
```

- `obs_std` (48):

```text
[0.5289, 0.2861, 0.1737, 0.763, 1.1121, 0.4987, 0.052, 0.0462, 0.0325, 0.5712, 0.2858, 0.306, 0.1177, 0.1211, 0.1356, 0.1353, 0.1112, 0.12, 0.1176, 0.1193, 0.1441, 0.1335, 0.1199, 0.121, 2.9915, 3.0103, 3.3106, 3.3324, 2.805, 2.9812, 2.8629, 2.9136, 3.6772, 3.4195, 3.1409, 3.1827, 0.5953, 0.5856, 0.635, 0.6403, 0.6576, 0.684, 0.6177, 0.6237, 0.8305, 0.7825, 0.6942, 0.7136]
```

### A.2 动作噪声参数 `std`（12维）

```text
[0.1967, 0.1904, 0.1573, 0.1488, 0.3479, 0.328, 0.2848, 0.3007, 0.3869, 0.384, 0.287, 0.2705]
```
