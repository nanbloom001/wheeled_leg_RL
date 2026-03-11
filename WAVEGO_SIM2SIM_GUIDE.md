# WAVEGO Sim2Sim 快速使用指南

## 🎯 修复完成

所有 7 个关键 Bug 已修复（2026-02-13）：

| Bug | 描述 | 状态 |
|-----|------|------|
| 1 | 关节排序错误（字母序 vs 深度优先） | ✅ 已修复 |
| 2 | 站立姿态顺序错乱 | ✅ 已修复 |
| 3 | 坐标系变换多余且错误 | ✅ 已修复 |
| 4 | 关节映射不必要 | ✅ 已修复 |
| 5 | 缺少观测 clipping | ✅ 已修复 |
| 6 | MuJoCo 执行器缺少 kv | ✅ 已修复 |
| 7 | Keyframe 站立位姿错误 | ✅ 已修复 |

---

## 🚀 运行 Sim2Sim

### 方法 1: 使用策略自动控制

```bash
# 激活环境
conda activate env_isaaclab

# 进入工作目录
cd /home/user/IsaacLab

# 运行 sim2sim（使用训练好的策略）
python scripts/sim2sim_mujoco.py
```

**预期输出：**
```
[Step    50] VelX: 0.123 m/s | ObsNormMax: 2.34
[Step   100] VelX: 0.287 m/s | ObsNormMax: 2.56
[Step   150] VelX: 0.456 m/s | ObsNormMax: 2.78
...
```

**预期行为：**
- ✅ 机器人从站立姿态开始（4条腿稳定支撑）
- ✅ 平稳前进，无抽搐或异常震动
- ✅ VelX 逐渐增加至 ~0.5 m/s（目标速度）
- ✅ 关节运动连贯、对称

---

### 方法 2: 使用 MuJoCo Viewer 手动查看

```bash
conda activate env_isaaclab
cd /home/user/IsaacLab
python -m mujoco.viewer --mjcf WAVEGO_mujoco/scene.xml
```

**操作：**
- 按 `Space` 暂停/继续仿真
- 按 `Ctrl+R` 重置到 `home` keyframe
- 鼠标拖拽旋转视角
- 滚轮缩放

---

## 📊 关键参数说明

### 训练环境参数（Isaac Lab）
```python
# 来自: flat_env_cfg.py
sim.dt = 0.005          # 5ms 物理步
decimation = 4          # 控制频率降频
control_freq = 50 Hz    # 实际策略频率

action_scale = 0.25     # 动作缩放
num_obs = 48            # 观测维度
num_actions = 12        # 动作维度（12个关节）
```

### MuJoCo 模型参数
```xml
<!-- 来自: wavego.xml -->
timestep = 0.005        <!-- 对齐 Isaac Lab -->
kp = 400                <!-- Stiffness -->
kv = 10                 <!-- Damping -->
forcerange = [-1.96, 1.96]  <!-- 1.96 Nm = 20 kg·cm -->
```

### 关节排序（深度优先）
```
Index | Joint Name       | Standing Pose
------|------------------|---------------
  0   | FL_hip_joint     | +0.100 rad
  1   | FL_thigh_joint   | -0.650 rad
  2   | FL_calf_joint    | +0.600 rad
  3   | FR_hip_joint     | -0.100 rad
  4   | FR_thigh_joint   | +0.650 rad
  5   | FR_calf_joint    | -0.600 rad
  6   | RL_hip_joint     | -0.100 rad
  7   | RL_thigh_joint   | -0.650 rad
  8   | RL_calf_joint    | +0.600 rad
  9   | RR_hip_joint     | +0.100 rad
 10   | RR_thigh_joint   | +0.650 rad
 11   | RR_calf_joint    | -0.600 rad
```

---

## 🔍 观测空间构成（48维）

```
Offset | Observation      | Dims | Description
-------|------------------|------|---------------------------
  0-2  | base_lin_vel     |  3   | Body系线速度 [x,y,z]
  3-5  | base_ang_vel     |  3   | Body系角速度 [x,y,z]
  6-8  | projected_gravity|  3   | Body系重力投影
  9-11 | velocity_commands|  3   | 速度指令 [vx,vy,wz]
 12-23 | joint_pos_rel    | 12   | 关节位置（相对默认值）
 24-35 | joint_vel_rel    | 12   | 关节速度
 36-47 | last_action      | 12   | 上一步动作
```

**归一化：**
```python
obs_norm = (obs_raw - mean) / (std + 1e-8)
obs_norm = np.clip(obs_norm, -5.0, 5.0)  # ← 重要！
```

---

## 🛠️ 故障排查

### 问题 1: 机器人剧烈抽搐
**可能原因：**
- ❌ 关节排序错误（已修复）
- ❌ 坐标系变换错误（已修复）

**验证：**
```bash
python scripts/verify_sim2sim_fix.py
```

### 问题 2: 机器人倒地
**可能原因：**
- 站立姿态初始化错误
- Keyframe 配置错误

**检查：**
```python
# 应该看到对称的站立角度
print(STANDING_POSE)
# [ 0.1 -0.65  0.6  -0.1  0.65 -0.6  -0.1 -0.65  0.6   0.1  0.65 -0.6]
```

### 问题 3: 策略加载失败
**检查文件路径：**
```bash
ls -lh logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/model_2999.pt
```

---

## 📚 相关文件

| 文件 | 用途 |
|------|------|
| `scripts/sim2sim_mujoco.py` | 主脚本（策略推理 + MuJoCo仿真） |
| `WAVEGO_mujoco/wavego.xml` | 机器人模型定义（执行器、求解器） |
| `WAVEGO_mujoco/scene.xml` | 场景定义（地面、光照、keyframe） |
| `scripts/verify_sim2sim_fix.py` | 验证脚本（检查所有修复） |
| `logs/rsl_rl/.../model_2999.pt` | 训练权重（Actor+Normalizer） |
| `source/.../wavego.py` | Isaac Lab 资产配置 |
| `source/.../flat_env_cfg.py` | 训练环境配置 |

---

## 📖 更多信息

详细修复日志见：[GEMINI_LOG.md](GEMINI_LOG.md#2026-02-13-1430-修复-wavego-sim2sim-的-7-个关键-bug)

完整迁移与调优记录见：[WAVEGO_SIM2SIM_MIGRATION_TUNING_LOG.md](WAVEGO_SIM2SIM_MIGRATION_TUNING_LOG.md)
