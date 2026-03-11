# WAVEGO IsaacLab → MuJoCo Sim2Sim 迁移与调优完整记录（截至 2026-02-18）

> 目标：沉淀从 IsaacLab 训练策略迁移到 MuJoCo 可运行 sim2sim 的全过程，记录每一类改动、每一轮验证、每一组调优结果。  
> 原则：仅记录仓库内已执行/已产出的内容；未验证项明确标注为“待验证”。

---

## 1. 项目目标与约束

- **目标**：让 `Isaac-Velocity-Flat-WAVEGO-v0` 的已训练策略在 MuJoCo 中稳定复现“前进/静止”等行为，并将 sim gap 定位到可量化指标。
- **硬约束**：
  - 不依赖臆测，信息来源必须落到现有文件/日志。
  - 观测维度、顺序、标准化必须与训练策略一致。
  - 宏观（轨迹）与微观（关节级）必须同时评估。

---

## 2. 关键输入与基线资产

### 2.1 策略与配置来源

- 策略 checkpoint：`logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/model_2999.pt`
- 导出策略：`logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/exported/policy.pt` / `policy.onnx`
- 训练参数：`logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/params/env.yaml`、`agent.yaml`
- IO descriptor：`tmp/io_descriptors_wavego/isaac_velocity_flat_wavego_v0_IO_descriptors.yaml`

### 2.2 模型与场景来源

- MuJoCo 机器人模型：`WAVEGO_mujoco/wavego.xml`
- MuJoCo 场景：`WAVEGO_mujoco/scene.xml`
- 机器人 URDF：`WAVEGO_description/WAVEGO_description/urdf/WAVEGO.urdf`

---

## 3. 迁移过程时间线（阶段化）

### 阶段 A：信息采集与可信基线建立

1. 汇总训练产物、参数与模型文件，建立“仅事实、可追溯”清单。  
2. 输出资料文档：
   - `WAVEGO_SIM2SIM_INFO_CHECKLIST.md`
   - `WAVEGO_ISAACLAB_OBS_COMPUTATION.md`

### 阶段 B：模型一致性校验与修补

1. 对齐 URDF ↔ MJCF 关节范围与局部位姿。  
2. 修正 `wavego.xml` 中关节范围与部分 link 局部位姿。  
3. 验证 MuJoCo 模型可编译、可加载并可步进。

### 阶段 C：从头实现可诊断 sim2sim 执行链

1. 新建并迭代 `scripts/sim2sim_test.py`：
   - 48 维观测构造（逐项复现）
   - actor normalizer 应用与裁剪
   - policy joint order ↔ MuJoCo joint order 重排
   - 执行器参数 / timing 对齐校验
   - rollout 诊断与窗口统计
2. 新增单测：`scripts/tools/test/test_sim2sim_test.py`，覆盖观测构造与标准化关键逻辑。

### 阶段 D：跨仿真逐步对比系统化

1. 新建并迭代 `scripts/compare_isaac_mujoco_rollout.py`：
   - Isaac 与 MuJoCo 同命令、同步数 rollout
   - 每步记录 obs/action/base pose
   - 计算 obs/action RMSE、分组 RMSE、逐关节 RMSE
2. 增加宏观坐标对齐评估：按两侧各自初始朝向分解 forward/lateral 位移。

### 阶段 E：参数消融与 MuJoCo 调优

1. 新增运行时参数开关（不改策略）：
   - `--joint-damping`
   - `--joint-frictionloss-scale`
   - `--armature-scale`
   - `--floor-friction-scale`
   - `--solver-iterations`
   - `--noslip-iterations`
2. 对前进与静止两类工况进行 10s（500 步）消融评估。  
3. 按联合评分选最优配置，并回写到 MJCF 默认参数。

---

## 4. 关键代码改动记录（文件级）

## 4.1 `scripts/sim2sim_test.py`

- 实现/强化内容：
  - 48维 policy 观测构造链路
  - checkpoint normalizer 加载与应用
  - policy joint order 显式映射到 MuJoCo joint order
  - 执行器参数对齐：`kp/kd/forcerange`，以及 `armature/frictionloss`
  - 仿真步频对齐：`sim.dt=0.005`、`decimation=4`
  - rollout 诊断（速度误差、姿态、动作变化率、关节RMS）

## 4.2 `scripts/tools/test/test_sim2sim_test.py`

- 新增测试覆盖：
  - `projected_gravity` 计算
  - `joint_pos_rel` 与观测拼接维度/顺序
  - normalizer + clip 行为

## 4.3 `scripts/compare_isaac_mujoco_rollout.py`

- 新增/强化能力：
  - 宏观指标：世界系轨迹误差 + heading 对齐的 `forward/lateral` 误差
  - 微观指标：每关节 `joint_pos_rel_rmse` / `joint_vel_rel_rmse`
  - 优化评分：
    - `score_macro_forward_lateral = forward_disp_err + lateral_disp_err`
    - `score_micro_jointvel_action = joint_vel_rel_rmse + action_total_rmse`
    - `score_total = score_macro_forward_lateral + score_micro_jointvel_action`
  - 运行时参数消融开关（见阶段 E）

## 4.4 `WAVEGO_mujoco/wavego.xml`

- 早期修正：关节范围、局部位姿（与 URDF 对账）。
- 本轮调优回写（2026-02-18）：
  - `<option ... noslip_iterations="60" .../>`
  - `<joint damping="0.05" .../>`

---

## 5. 核心问题与根因定位

### 5.1 已定位并修复的主问题

- **关节顺序语义错位**：policy 顺序与 MuJoCo 深度优先顺序未对齐，导致动作语义错绑。  
  - 结果：修复后抖动显著下降，动作更平滑。
- **观测/标准化链路闭环**：确认 48 维观测、normalizer mean/std、clip 流程按策略要求执行。
- **执行器与时序一致性**：`dt/decimation` 与 `kp/kd/forcerange` 对齐。

### 5.2 仍存在的主要 gap

- 误差主导项长期集中在 `joint_vel_rel`（其次 `last_action`），说明差异更多来自动力学/接触层，而非观测公式本身。

---

## 6. 宏观 + 微观指标体系

## 6.1 宏观（行为层）

- 世界系轨迹偏差：`final_pos_err_norm`, `mean_pos_err_norm`
- heading 对齐偏差：
  - `forward_disp_err`
  - `lateral_disp_err`

## 6.2 微观（关节层）

- 分组 RMSE：`joint_pos_rel`, `joint_vel_rel`, `last_action`
- 逐关节 RMSE：12 个关节分别统计 `joint_pos_rel_rmse` / `joint_vel_rel_rmse`

---

## 7. 10 秒消融实验记录（500步）

数据汇总文件：`tmp/mujoco_ablation_summary_10s.csv`

### 7.1 前进命令（cmd = [0.5, 0, 0]）

| profile | score_total | macro_fl | micro_jv_act | forward_err | lateral_err | joint_vel_rmse | action_rmse |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 4.572529 | 0.585064 | 3.987465 | 0.501744 | 0.083320 | 3.316801 | 0.670664 |
| damp005 | 4.768347 | 1.007047 | 3.761300 | 0.554992 | 0.452056 | 3.113404 | 0.647896 |
| solver_hi | 4.604656 | 0.613518 | 3.991138 | 0.484081 | 0.129437 | 3.316645 | 0.674493 |
| **damp005_solver_hi** | **4.372476** | 0.655382 | **3.717093** | 0.598061 | **0.057322** | **3.070830** | **0.646263** |
| solver_hi_fric12 | 4.743738 | 0.762454 | 3.981284 | 0.527990 | 0.234463 | 3.301768 | 0.679516 |

结论：前进工况下综合最优为 `damp005_solver_hi`。

### 7.2 静止命令（cmd = [0, 0, 0]）

| profile | score_total | macro_fl | micro_jv_act | forward_err | lateral_err | joint_vel_rmse | action_rmse |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 3.481982 | 0.271423 | 3.210559 | 0.041725 | 0.229698 | 2.651650 | 0.558909 |
| **damp005** | **3.116466** | **0.105625** | **3.010840** | 0.030645 | **0.074981** | **2.470360** | **0.540480** |
| solver_hi | 3.795411 | 0.505048 | 3.290363 | 0.062698 | 0.442350 | 2.724586 | 0.565777 |
| damp005_solver_hi | 3.400997 | 0.214245 | 3.186752 | **0.019517** | 0.194728 | 2.618088 | 0.568663 |
| solver_hi_fric12 | 3.543067 | 0.255454 | 3.287613 | 0.087580 | 0.167875 | 2.696873 | 0.590740 |

结论：静止工况单独最优为 `damp005`；但前进+静止联合总分最优为 `damp005_solver_hi`。

---

## 8. 本轮回写优化与复验结果

### 8.1 已回写到默认 MJCF 的参数

- `joint damping = 0.05`
- `noslip_iterations = 60`

### 8.2 回写后复验（postxml vs base）

#### cmd050（前进）

- `score_total`: 4.572529 → 4.372476（改善 -0.200053）
- `micro_jv_act`: 3.987465 → 3.717093（改善 -0.270372）
- `macro_fl`: 0.585064 → 0.655382（变差 +0.070318；主要前进误差上升）

#### cmd000（静止）

- `score_total`: 3.481982 → 3.400997（改善 -0.080985）
- `macro_fl`: 0.271423 → 0.214245（改善 -0.057177）
- `micro_jv_act`: 3.210559 → 3.186752（改善 -0.023808）

结论：本轮回写更偏向“整体稳定性与微观误差下降”，对前进方向位移误差有一定副作用，需要后续定向补偿。

---

## 9. 微观关节层变化（postxml 相对 base）

### 9.1 前进工况改进最显著关节（joint_vel_rel_rmse）

- `RR_hip_joint`: 4.3633 → 3.9037
- `RL_hip_joint`: 3.6080 → 3.2121
- `FL_calf_joint`: 4.2674 → 3.9260
- `FR_calf_joint`: 3.6753 → 3.3582

### 9.2 静止工况改进最显著关节

- `FR_calf_joint`: 3.6508 → 3.3966
- `RL_calf_joint`: 3.1237 → 2.9534
- `RR_calf_joint`: 3.0264 → 2.8935

说明：误差高发关节主要集中在 hip/calf，符合“接触-摆动转换段敏感”的动力学特征。

---

## 10. 复现实验命令（可直接复跑）

```bash
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/user/IsaacLab

# 前进 10s（500步）
./isaaclab.sh -p scripts/compare_isaac_mujoco_rollout.py \
  --cmd-x 0.5 --cmd-y 0 --cmd-wz 0 --steps 500 \
  --output-csv tmp/repro_cmd050.csv --output-npz tmp/repro_cmd050.npz

# 静止 10s（500步）
./isaaclab.sh -p scripts/compare_isaac_mujoco_rollout.py \
  --cmd-x 0 --cmd-y 0 --cmd-wz 0 --steps 500 \
  --output-csv tmp/repro_cmd000.csv --output-npz tmp/repro_cmd000.npz
```

---

## 11. 当前结论

1. **迁移链路已闭环**：观测、标准化、动作映射、执行器参数、时序已可稳定复现。  
2. **主瓶颈已明确**：残余差异主要在动力学层（尤其 `joint_vel_rel`），非观测拼接错误。  
3. **已完成第一轮默认调优回写**：`damping=0.05` + `noslip_iterations=60`，整体联合评分下降。  
4. **仍需第二轮定向优化**：针对少数后腿关节与前进误差做分关节/分参数精调。

---

## 12. 下一步建议（待执行）

- 分关节调参（尤其后腿 hip/thigh/calf）并重排优先级：
  - per-joint damping
  - per-joint frictionloss
- 增加 20s（1000步）长时评估，确认误差是否累积。
- 将“宏观前进误差”与“微观关节速度误差”做多目标权重扫描，输出 Pareto 前沿。
