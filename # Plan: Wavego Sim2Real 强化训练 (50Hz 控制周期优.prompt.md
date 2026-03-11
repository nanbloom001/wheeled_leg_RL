# Plan: Wavego Sim2Real 强化训练 (50Hz 控制周期优化版)

本计划将针对 50Hz 控制链路下的延迟和噪声进行深度建模，通过在 `model_2999.pt` 基础上引入硬件级非理想特性，解决实机抖动、漂移及控制不平滑问题。

## Steps

1. **执行器与链路仿真 (`flat_env_cfg.py`)**：
    - **延迟模拟**：将 `WAVEGO_LEGS_CFG` 切换为 `DelayedPDActuatorCfg`。在 200Hz 物理频率下，设置延迟范围为 2-8 步（对应 10ms-40ms），覆盖 50Hz 控制周期下的 1 步左右延迟。
    - **执行器非线性**：在执行器配置中引入 `friction`（摩擦）和 `armature`（电枢惯量）的随机扰动，模拟舵机齿轮组的迟滞。

2. **传感器深度随机化 (`flat_env_cfg.py`)**：
    - **IMU 零偏与漂移**：为 `base_ang_vel` 观测项配置 `NoiseModelWithAdditiveBiasCfg`，模拟真实 IMU 常见的慢漂和上电偏移。
    - **关节反馈异常**：调高 `joint_vel` 的噪声上限至 `2.0`（原为 0.5），模拟 50Hz 低频采样下差分速度计算的剧烈波动。
    - **更新抖动**：通过观测噪音模拟传感器链路的非等间隔采样（Update Jitter）。

3. **奖励函数精调 (`flat_env_cfg.py`)**：
    - **零命令锁死**：引入 `stand_still_joint_deviation_l1`（权重 -2.0），在速度指令为零时强制回归默认姿态，彻底解决"零速度乱动"问题。
    - **漂移抑制与平滑**：
        - 新增 `lin_vel_y_l2` 惩罚项（权重 -1.5），抑制 X 轴行进时的横向漂移。
        - 将 `action_rate_l2` 权重从 -0.01 显著提升至 -0.05，强制生成更连续、平滑的控制信号。
    - **步态稳健性**：调整 `feet_air_time` 权重，引导产生频率更低、落地更稳的步态。

4. **续训流程设置 (`agents/rsl_rl_ppo_cfg.py`)**：
    - 修改 `max_iterations = 6000`（在现有 3000 轮基础上再训练 3000 轮）。
    - 降低 `init_noise_std = 0.2`（减小微调阶段的探索噪声，防止策略在加载后剧烈波动）。

5. **命令行启动**：
    - 运行续训指令，显式指向旧模型路径。

## Relevant files

- `/home/user/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/wavego/flat_env_cfg.py` — 主要配置修改：观测偏置、延迟执行器、新增稳定性奖励。
- `/home/user/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/wavego/agents/rsl_rl_ppo_cfg.py` — 调整迭代上限。
- `/home/user/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/wavego.py` — （可选）若需全局修改执行器类型。

## Verification

1. **静止测试**：在仿真中验证指令为 (0,0,0) 时，机器人是否能在 1 秒内完全静止且无微小抖动。
2. **直线度测试**：给定 $V_x=0.5 m/s$，观察 5 秒内 $V_y$ 和 $Yaw$ 的累积偏差。
3. **输出平滑度**：通过 `play.py` 可视化关节目标位置曲线，确认其是否呈平滑正弦类波动，无阶跃跳变。

## Decisions

- **控制频率固定**：维持 `decimation=4` (50Hz)，所有延迟建模以此为基准。
- **续训策略**：不改变观测维度，仅改变观测质量（加噪/偏置）和惩罚权重，确保模型能直接加载。

## Further Considerations

1. **指令提示**：续训时请确保 `--task Isaac-Velocity-Flat-WAVEGO-v0` 与模型对应。
2. **训练命令参考**：
   ```bash
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-WAVEGO-v0 --resume --load_run 2026-02-12_03-17-29 --checkpoint model_2999.pt
   ```
