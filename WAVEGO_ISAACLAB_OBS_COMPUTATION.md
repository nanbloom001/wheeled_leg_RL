# WAVEGO IsaacLab 观测项计算说明（Policy 48 维）

> 目标：精确说明 IsaacLab 训练/推理中每个观测项如何计算。  
> 原则：仅引用仓库代码与运行导出结果；不做推测。

## 0) 依据文件

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/wavego/flat_env_cfg.py`
- `source/isaaclab/isaaclab/envs/mdp/observations.py`
- `source/isaaclab/isaaclab/assets/rigid_object/rigid_object_data.py`
- `logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/params/env.yaml`
- `tmp/io_descriptors_wavego/isaac_velocity_flat_wavego_v0_IO_descriptors.yaml`

## 1) Policy 观测拼接顺序与维度

按 `observations.policy` 顺序拼接（`concatenate_terms=true`）：

1. `base_lin_vel` (3)
2. `base_ang_vel` (3)
3. `projected_gravity` (3)
4. `velocity_commands` (3)
5. `joint_pos` (`joint_pos_rel`) (12)
6. `joint_vel` (`joint_vel_rel`) (12)
7. `actions` (`last_action`) (12)

总维度：`48`。

## 2) 各观测项计算定义

### 2.1 `base_lin_vel` (3)

- 代码入口：`isaaclab.envs.mdp.observations.base_lin_vel`
- 返回：`asset.data.root_lin_vel_b`
- 含义：机体根部线速度在机体坐标系 B 下的分量（X, Y, Z）
- 单位：m/s

表达式（按接口语义）：

$$
\mathbf{v}_B = R_{WB}^{\top} \mathbf{v}_W
$$

其中 `root_lin_vel_b` 在 `RigidObjectData` 中等价于 `root_com_lin_vel_b`。

### 2.2 `base_ang_vel` (3)

- 代码入口：`isaaclab.envs.mdp.observations.base_ang_vel`
- 返回：`asset.data.root_ang_vel_b`
- 含义：机体根部角速度在机体坐标系 B 下的分量（X, Y, Z）
- 单位：rad/s

表达式：

$$
\boldsymbol{\omega}_B = R_{WB}^{\top} \boldsymbol{\omega}_W
$$

其中 `root_ang_vel_b` 在 `RigidObjectData` 中等价于 `root_com_ang_vel_b`。

### 2.3 `projected_gravity` (3)

- 代码入口：`isaaclab.envs.mdp.observations.projected_gravity`
- 返回：`asset.data.projected_gravity_b`
- 底层定义（`RigidObjectData.projected_gravity_b`）：
  `math_utils.quat_apply_inverse(self.root_link_quat_w, self.GRAVITY_VEC_W)`
- 含义：世界重力方向投影到机体根坐标系 B

表达式：

$$
\mathbf{g}_B = R_{WB}^{\top} \mathbf{g}_W, \quad \mathbf{g}_W = [0,0,-1] \text{ 或按实现约定的重力方向向量}
$$

说明：该项用于姿态相关信息编码，数值语义与根部姿态直接相关。

### 2.4 `velocity_commands` (3)

- 代码入口：`isaaclab.envs.mdp.observations.generated_commands`
- 调用：`env.command_manager.get_command("base_velocity")`
- 含义：当前 command manager 生成的速度指令
- 当前任务配置分量：`[lin_vel_x, lin_vel_y, ang_vel_z]`

### 2.5 `joint_pos` / `joint_pos_rel` (12)

- 代码入口：`isaaclab.envs.mdp.observations.joint_pos_rel`
- 计算：`asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]`
- 含义：相对默认关节位的关节角
- 单位：rad

表达式：

$$
\mathbf{q}_{rel} = \mathbf{q} - \mathbf{q}_{default}
$$

### 2.6 `joint_vel` / `joint_vel_rel` (12)

- 代码入口：`isaaclab.envs.mdp.observations.joint_vel_rel`
- 计算：`asset.data.joint_vel[:, ids] - asset.data.default_joint_vel[:, ids]`
- 当前任务中默认关节速度为零向量，因此等价于关节速度本身
- 单位：rad/s

表达式：

$$
\dot{\mathbf{q}}_{rel} = \dot{\mathbf{q}} - \dot{\mathbf{q}}_{default}
$$

### 2.7 `actions` / `last_action` (12)

- 代码入口：`isaaclab.envs.mdp.observations.last_action`
- 计算：若未指定 action term，返回 `env.action_manager.action`
- 含义：上一个环境 step 输入到动作管理器的动作向量

## 3) 训练配置中的噪声 / clip / scale（本任务）

基于 `env.yaml`：

- `base_lin_vel`: `Uniform[-0.1, 0.1]`
- `base_ang_vel`: `Uniform[-0.2, 0.2]`
- `projected_gravity`: `Uniform[-0.05, 0.05]`
- `velocity_commands`: 无噪声
- `joint_pos_rel`: `Uniform[-0.03, 0.03]`
- `joint_vel_rel`: `Uniform[-0.5, 0.5]`
- `last_action`: 无噪声

所有 policy term：`clip=null`，`scale=null`。

## 4) 运行导出交叉验证（已执行）

通过：

```bash
./isaaclab.sh -p scripts/environments/export_IODescriptors.py \
  --task Isaac-Velocity-Flat-WAVEGO-v0 \
  --output_dir /home/user/IsaacLab/tmp/io_descriptors_wavego
```

已验证：

- policy 组形状为 `(48,)`
- 7 个观测项顺序与训练配置一致
- action 维度为 `12`
- joint_names 顺序导出为：
  `FL_hip_joint, FR_hip_joint, RL_hip_joint, RR_hip_joint, FL_thigh_joint, FR_thigh_joint, RL_thigh_joint, RR_thigh_joint, FL_calf_joint, FR_calf_joint, RL_calf_joint, RR_calf_joint`
- `joint_pos_offsets` 与默认站立位一致

## 5) 推理端复现要点（给新 sim2sim 实现）

1. 先按第 1 节顺序构造原始 48 维观测。
2. 再决定是否应用 checkpoint normalizer（若应用，需固定 mean/std 与数值稳定项）。
3. `projected_gravity` 必须使用“世界重力向量逆旋到机体系”的同一实现语义。
4. `last_action` 必须使用“上一时刻输入动作”，不能错用当前未执行动作。

## 6) 仍需在新 sim2sim 中自行定版的项（留白）

以下属于“实现策略选择”，当前仓库不能给出唯一答案：

- 推理端是否启用 obs clip（以及区间）
- 推理端是否启用 action clip（以及区间）
- 推理端 normalize 的确切公式细节（是否含 epsilon、是否先 clip）
