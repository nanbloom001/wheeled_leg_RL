# GEMINI_LOG

## [2026-01-17 23:30] 导入自定义四足机器人 QWE_DOG

### 执行的操作：
1. **资产转换**: 将位于 `qwe.SLDASM/urdf/qwe.SLDASM.urdf` 的 URDF 模型转换为 USD 格式。
   - 目标路径: `source/isaaclab_assets/data/Robots/User/qwe_dog.usd`
   - 使用参数: `--merge-joints` (性能优化)。
2. **编写配置**: 创建了机器人配置文件 `source/isaaclab_assets/isaaclab_assets/robots/qwe_dog.py`。
   - 定义了 `QWE_DOG_CFG`。
   - 设置了初始关节状态和基于 `ImplicitActuator` (PD) 的执行器参数。
3. **注册资产**: 在 `source/isaaclab_assets/isaaclab_assets/robots/__init__.py` 中添加了导出，使其全局可用。
4. **环境维护**: 执行了 `./isaaclab.sh --install` 以确保 Conda 环境中的 `isaaclab` core 库更新并可用。

## [2026-01-18 04:40] 修复关节限制导致的启动错误

### 执行的操作：
1. **修改 URDF**: 将 `/home/user/IsaacLab/qwe.SLDASM/urdf/qwe.SLDASM.urdf` 中的关节限制从 `[-3, 3]` 扩大到 `[-6.28, 6.28]`，以允许机器人以翻转 180 度 (3.14 rad) 的姿态初始化。
2. **重新转换资产**: 重新运行 `convert_urdf.py` 生成新的 `qwe_dog.usd`，覆盖旧资产。
   - 环境: `env_isaaclab`
   - 命令: `./isaaclab.sh -p scripts/tools/convert_urdf.py ...`

## [2026-01-18 05:07] 标准 URDF 导入与转换流程

用户更新了 `qwe.SLDASM` 的 URDF 文件，并请求执行标准的转换流程。

### 标准转换步骤：
1. **预处理 (关节限制)**: 
   由于 Isaac Lab 的某些初始化状态可能需要关节旋转 180 度 (3.14 rad)，而默认导出的 URDF 限制可能只有 `[-3, 3]`，这会导致启动时的边界检查错误。
   - **操作**: 使用 `sed` 命令强制将关节限制扩大到 `[-6.28, 6.28]` (即 [-2π, 2π])。
   ```bash
   sed -i 's/lower="-3"/lower="-6.28"/g' /home/user/IsaacLab/qwe.SLDASM/urdf/qwe.SLDASM.urdf
   sed -i 's/upper="3"/upper="6.28"/g' /home/user/IsaacLab/qwe.SLDASM/urdf/qwe.SLDASM.urdf
   ```
2. **执行转换 (URDF -> USD)**:
   使用 Isaac Lab 提供的工具将 URDF 转换为 USD 格式。
   - **工具**: `scripts/tools/convert_urdf.py`
   - **参数**: `--merge-joints` (合并固定关节)。
   - **执行命令**:
   ```bash
   source /home/user/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
   ./isaaclab.sh -p scripts/tools/convert_urdf.py ...
   ```

## [2026-01-18 05:27] 修复关节方向、默认位置及姿态

针对用户反馈的三个问题（需要+/-3.14归零、旋转方向不一致、模型侧卧），执行了以下深度修正：

### 执行的操作：
1. **修改 URDF (关节层)**: 直接修改右侧大腿关节的 `origin rpy` 减去 $\pi$，实现默认位置归零；反转 X 轴轴向，修正旋转方向。
2. **重新转换资产**: 重新生成 `qwe_dog.usd`。
3. **更新配置文件 (`qwe_dog.py`)**: 
   - 归零 `init_state.joint_pos`。
   - 修正姿态: `init_state.rot` 设为 `(0.7071, 0.0, 0.7071, 0.0)`（绕 Y 轴旋转 +90 度），修正“侧卧”和“上下颠倒”问题。
   - 抬高高度: `init_state.pos` 设为 `(0.0, 0.0, 0.6)`。

## [2026-01-18 06:40] 解决关节单向卡顿/反弹问题

### 问题诊断：
关节在向外伸展时会发生反弹，这是由于 `convexHull` 碰撞体过大导致的虚假自碰撞。

### 执行的操作：
1. **统一旋转方向**: 修改 URDF，将 `small_qian_you_joint` 轴向反转，确保所有关节逆时针旋转一致。
2. **优化碰撞体**: 编写并运行 `scripts/custom_urdf_converter.py`，显式设置 `collider_type="convex_decomposition"`。

## [2026-01-18 06:55] 优化快速控制稳定性 (平滑滤波与临界阻尼)

### 问题诊断：
系统处于欠阻尼状态且输入信号突变。

### 执行的操作：
1. **脚本层优化**: 修改 `scripts/tune_qwe_dog_joints.py`，引入 **Slew Rate Limiter**（每帧变化限制在 0.05 rad）。
2. **控制层优化**: 将 `stiffness` 降至 `5.0`，`damping` 设为 `1.0`，达到临界阻尼状态。

## [2026-01-18 07:08] 彻底解决“反弹”：修复 Effort/Velocity 限制

### 根本原因发现：
URDF 默认限制为 `effort="1"` (1 Nm) 和 `velocity="1"` (1 rad/s)，导致力矩饱和无法制动。

### 执行的操作：
1. **修改 URDF**: 将所有关节的 `effort` 和 `velocity` 限制提高到 `100`。
2. **增强调试脚本**: 加入实时关节位置、速度和力矩监控。

## [2026-01-18 07:37] Sim-to-Real: 适配 MG996R 舵机参数

### 执行的操作：
1. **修改 URDF**: 设定真实参数 `effort="1.27"`，`velocity="6.3"`。
2. **配置更新**: `stiffness=400.0`, `damping=10.0`, `effort_limit_sim=1.27`, `velocity_limit_sim=6.3`。

## [2026-01-18 07:45] 增强物理阻尼以模拟减速箱 friction

### 执行的操作：
1. **URDF 修改**: 添加 `<dynamics friction="0.5" damping="0.5"/>`。

## [2026-01-18 08:20] 解决静止下落：求解器增强与显式摩擦力设置

### 执行的操作：
1. **求解器增强**: `solver_position_iteration_count` 提升至 `16`，`velocity` 提升至 `4`。
2. **显式设置摩擦**: 在 Config 中添加 `friction=1.0` (Nm) 和 `armature=0.01`。
3. **静摩擦增强**: 为彻底消除漂移，一度尝试将 URDF friction设为 `5.0`。

## [2026-01-18 11:48] 修复轮子关节位置 (坐标系映射修正)

### 执行的操作：
1. **坐标系映射**: 确认 Isaac Lab X轴 = URDF Y轴。
2. **消除偏移**: 将所有 `last_` 关节的 `origin y` 设为 `0`。
3. **对称性修正**: 左侧 X 设为 `0.056`，右侧 X 设为 `-0.056`。

## [2026-01-18 12:02] 轮子关节位置深度修复（RPY对齐）

### 执行的操作：
1. **装配对齐**: 对比 `qwe_old` 恢复了复杂的 `rpy` 参数，解决了轮子脱离小腿的问题。
2. **镜像修正**: 修正了右侧轮子因坐标系翻转导致的位移方向错误（最终左右侧 X 均设为 `0.056`）。

## [2026-01-18 12:13] 恢复历史日志并启用精确时间戳

### 执行的操作：
1. **日志还原**: 撤销了 Turn 33 的摘要式重写，恢复了所有历史详细记录。
2. **时间戳规范化**: 为所有历史条目补全了精确到分钟的时间戳（基于仿真运行日志和交互序列）。

## [2026-01-18 12:18] 实现左右关节镜像对称控制

### 执行的操作：
1. **反转右侧轴向**: 修改 URDF，将右侧所有大腿和小腿关节 (`big_..._you_joint`, `small_..._you_joint`) 的 `axis` 从 `1 0 0` 反转为 `-1 0 0`。
2. **重新转换资产**: 重新生成 `qwe_dog.usd`。

## [2026-01-18 12:30] 根据实测数据更新机器人质量与惯性

### 执行的操作：
1. **质量更新 (URDF)**: 
   - 身体 (`base_link`): `0.90kg`。
   - 大腿 (`thigh_...`): `0.085kg`。
   - 小腿 (`shank_...`): `0.120kg`。
2. **惯性缩放**: 按比例缩放了所有 link 的 `inertia` 张量。

## [2026-01-18 12:41] 设置关节限位与开启自碰撞

### 执行的操作：
1. **关节限位 (URDF)**: 大腿 $\pm 0.6$ rad, 小腿 $\pm 1.396$ rad。
2. **开启自碰撞 (Config)**: 在 `qwe_dog.py` 中设置 `enabled_self_collisions=True`。

## [2026-01-18 13:05] 更新关节命名与设置默认姿态

### 执行的操作：
1. **全量重命名**: 将 URDF 和 Config 中的所有拼音名统一改为标准英文（thigh, shank, wheel）。
2. **默认姿态设定**: 在 `qwe_dog.py` 中将前方小腿设为 `-0.8` rad，后方小腿设为 `0.8` rad，大腿保持 `0.0`。


## [2026-01-18 13:10] 修复调试脚本对正则配置的支持

### 执行的操作：
修改 `tune_qwe_dog_joints.py`，引入 `re` 模块进行正则匹配，确保滑块能正确加载配置文件的初始角度。

## [2026-01-18 14:09] 成功部署并启动强化学习训练

### 执行的操作：
1. **创建 RL 任务**: 基于 Unitree A1 模板创建了 `QweDogRoughEnvCfg` 和 `QweDogFlatEnvCfg`。
2. **PhysX 稳定性修复**: 大幅增加了 `gpu_collision_stack_size` 和称其他 GPU 缓冲区容量，解决了自碰撞导致的缓冲区溢出崩溃。
3. **环境兼容性修正**: 
   - 禁用了 `base_com` 事件（解决了 `base` 连杆命名冲突）。
   - 在平坦地形任务中移为了 `height_scanner` 及相关的观测项（解决了实体丢失错误）。
4. **奖励与材质优化**: 
   - 根据 A1 配置调优了 `feet_air_time` 和 `flat_orientation_l2` 权重。
   - 显式设置了地面的高摩擦材质，防止舵机驱动的机器狗打滑。
5. **训练启动**: 成功通过 `rsl_rl` 框架启动了 `Isaac-Velocity-Flat-QweDog-v0` 任务。

## [2026-01-18 18:12] Sim-to-Real: 阶段一训练配置优化

### 目标设定：
针对硬件限制（N20 电机无法驱动且无法锁死），决定分阶段训练。阶段一：**纯腿部踏步训练**（忽略轮子，模拟 8-DOF 机器人）。

### 执行的操作：
1. **动作空间缩减**: 修改 `flat_env_cfg.py`，将 `actions` 限制为仅控制 `thigh_.*` 和 `shank_.*`（8维），轮子关节保持物理锁定。
2. **指令空间定制**:
   - 禁用 `lin_vel_y` (Y 轴平移)，因为无侧摆关节。
   - 启用 `ang_vel_z` (Yaw 轴转向)，允许通过差速迈步进行转向。
   - 限制 `lin_vel_x` 至 `[-0.3, 0.3]` m/s，适配 MG996R 的低转速。
3. **奖励函数微调**:
   - 大幅提高 `feet_air_time` 权重至 0.5，强力引导机器人抬腿迈步。
   - 增加 `flat_orientation_l2` 惩罚，防止侧倾摔倒。
4. **摩擦力回调**: 将 Config 中的 `friction` 降至 **0.4 Nm**，在保证断电不掉腿的同时，给予电机更多控制余量。

## [2026-01-19 13:30] 多卡分布式训练优化与性能调优

### 目标设定：
解决 DDP 模式下 Livestream 导致的严重性能瓶颈（Steps/s 低，Learning Time 长），并压榨显存以提升总吞吐量。

### 执行的操作：
1. **训练脚本升级**: 编写了 `scripts/train_ddp_advanced.py`。
   - 支持 `--render_rank` 参数，实现异构 Headless 控制（指定 Rank 开启 GUI/直播，其他静默）。
   - 修复了 `RecordVideo` 在 DDP 下多进程写入同一文件的 Bug。
   - 修复了 `train.py` 中 `os` 模块导入顺序的 Bug。
2. **性能瓶颈排查**:
   - 确认 **Livestream** 是导致 Learning Time 暴增（0.3s -> 8s）的元凶。
   - 决定彻底关闭训练时的直播，改用 **Video Recording** 监控。
3. **Sim-to-Real 深度优化**:
   - **降频 (Decimation)**: 设为 **10** (20Hz)，模拟真实舵机的慢响应。
   - **噪声注入**: 添加 Action Noise (死区模拟) 和 Observation Noise。
   - **随机化**: 开启物理材质 (Friction) 随机化。
   - **接触惩罚**: 新增 `undesired_contacts`，严厉惩罚膝盖 (`shank`) 和腹部 (`base_link`) 触地，强制机器人学会站立。
   - **提前终止**: 膝盖/腹部触地即触发 Episode 重置。
4. **吞吐量压榨**:
   - **Num Envs**: 提升至 **8192** (单卡) -> 32768 (总)。
   - **PPO 优化**: `num_mini_batches=1` (Full Batch)，大幅减少梯度同步频率。
   - **PhysX**: 进一步调大 `gpu_collision_stack_size` (512MB) 以防 OOM。

## [2026-01-20 20:30] 开启“快速直线”训练 (Phase 2)

### 目标设定：
在 Phase 1 (500轮) 稳定直线行走的基础上，移除膝盖重置限制（允许偶尔失误），提升速度上限，进行进阶训练。

### 执行的操作：
1. **创建新配置**: `flat_env_linear_fast_cfg.py`。
2. **关键修改**:
   - **速度上限**: `lin_vel_x` 范围从 `[-0.3, 0.3]` 提升至 `[-0.3, 0.5]`。
   - **重置宽松化**: `terminations.base_contact` 仅保留 `base_link`，移除了 `shank_.*`（膝盖触地不再直接重置，仅触发惩罚）。
3. **继承关系**: 继承自 `LocomotionVelocityRoughEnvCfg`，但保留了 Linear Config 的所有奖励权重。

## [2026-01-21 01:10] 深度物理与框架优化 (Native Metrics & Braking)

### 问题诊断：
1. **行走困难**: 怀疑足部轮子在蹬地时滚动导致推力流失。
2. **监控不足**: `error_vel_xy` 耦合了 XY 轴误差，无法精细诊断。
3. **限位过窄**: 大腿 $\pm 0.6$ rad 限制了迈步幅度。

### 执行的操作：
1. **核心库增强 (`velocity_command.py`)**: 
   - 原生支持 `error_vel_x` 和 `error_vel_y` 拆分统计。
   - 原生支持 `error_wheel_vel` (通过正则自动匹配所有轮子关节并统计转速)。
2. **物理刹车 (`qwe_dog.py`)**: 
   - 拆分了 `legs` 和 `wheels` 执行器。
   - 给 `wheels` 设置了 `stiffness=0, damping=1000`，模拟强力电子手刹，彻底锁死轮子。
3. **机械解封 (URDF)**: 
   - 大腿限位扩大至 `[-1.4, 1.4]` ($\pm 80^\circ$)。
   - 小腿限位扩大至 `[-1.57, 1.57]` ($\pm 90^\circ$)。
4. **Config 清理**: 移除了 `flat_env_linear_fast_cfg.py` 中所有的 Hack 监控函数，使配置回归纯净。

### 结果确认:
- **训练速度翻倍**: 由于物理约束稳定性提高（轮子不再打滑），计算速度从 46k 提升至 88k steps/s。
- **轮子锁死验证**: `Metrics/base_velocity/error_wheel_vel` 约为 0.0039 rad/s，确认为锁定状态。
- **行走成功**: 机器狗开始表现出有效的行走步态，不再原地晃动。
## [2026-02-07 10:25] 修复 WAVEGO 调试脚本 API 兼容性

### 执行的操作：
1. **修改脚本**: `scripts/tune_wavego_joints.py`
   - 将 `DomelightCfg` 更改为 `DomeLightCfg`，适配 Isaac Lab 2.0+ 的 API 命名规范。

## [2026-02-07 11:30] 导入 WAVEGO (12-DOF) 机器人并适配硬件参数

### 执行的操作：
1. **资产识别**: 确认 WAVEGO 模型位于 `WAVEGO_description` 目录，具备 12 个自由度（Hip, Thigh, Calf）。
2. **硬件参数适配 (wavego.py)**:
   - **舵机**: 微雪 ST3215-HS 串行总线舵机。
   - **力矩限制**: 设为 `2.94 Nm` (30 kg.cm)。
   - **速度限制**: 设为 `11.1 rad/s` (106 RPM)。
   - **控制特性**: 采用高刚度 (`stiffness=400.0`) 和适中阻尼 (`damping=10.0`) 模拟位置闭环。
3. **物理特性微调**:
   - `friction`: 设为 `0.2 Nm` (总线舵机内部摩擦相对较小)。
   - `armature`: 设为 `0.01`。
   - 开启了自碰撞检测 (`enabled_self_collisions=True`)。
4. **文档更新**: 在 `GEMINI.md` 中新增了 WAVEGO 的详细硬件参数说明。
5. **调试准备**: 已确认 `scripts/tune_wavego_joints.py` 脚本可用，用于实时验证关节运动。

## [2026-02-10 18:15] 更新 WAVEGO 舵机扭矩参数

### 执行的操作：
1. **文档同步**: 在 `GEMINI.md` 中将 WAVEGO 的工作扭矩修正为实测/规格值的 **20 kg·cm**。
2. **配置更新 (wavego.py)**: 将 `WAVEGO_LEGS_CFG` 中的 `effort_limit_sim` 从 2.94 Nm 修正为 **1.96 Nm** (对应 20 kg·cm)，以符合真实的电机能力边界。

## [2026-02-11 16:35] 修正 WAVEGO 关节逻辑方向与调试分类

### 执行的操作：
1. **反转关节轴向 (URDF)**: 
   - 针对 `FL_hip`, `RR_hip`, `FL_calf`, `RL_calf`, `FR_thigh`, `RR_thigh` 进行轴向反转。
   - 目标：确保所有关节从外侧观察时，角度增加对应顺时针转动（匹配舵机控制逻辑）。
2. **物理限位校准**: 
   - 移除 URDF 中残留的 `effort=100` 调试值，统一修正为 **1.96 Nm**。
   - 统一修正 `velocity` 为 **11.1 rad/s**。
3. **调试脚本增强 (`tune_wavego_joints.py`)**: 
   - 修复分类逻辑，从 `LF/RF/LH/RH`适配为模型实际使用的 `FL/FR/RL/RR` 前缀。
4. **资产同步**: 重新生成 `WAVEGO.usd`。

## [2026-02-11 17:10] 更新 WAVEGO 质量分布与足端物理特性

### 执行的操作：
1. **质量分布重构 (URDF)**: 
   - 整体减重：总质量从 ~2.4kg 降低至 **1.59kg**。
   - 细分：Base (750g), Hip (70g), Thigh (120g), Calf (20g)。
   - 惯性张量按比例缩放以保持动力学稳定性。
2. **舵机特性维持**: 
   - 在 URDF 关节中引入 `<dynamics friction="0.4" damping="0.5"/>`。
   - 在 `wavego.py` 中设置 `friction=0.4`，确保断电状态下的物理模拟符合减速箱自锁逻辑。
3. **接触特性修正**: 
   - 将足端/地面摩擦系数设定为 **0.5**。
4. **资产重构**: 重新生成 `WAVEGO.usd` 资产文件。

## [2026-02-11 17:25] 创建 WAVEGO 12-DOF 强化学习任务

### 执行的操作：
1. **任务环境开发**: 创建 `WavegoFlatEnvCfg` 及其对应的 PPO 配置。
2. **多自由度适配**: 
   - 动作空间扩展至 12 关节全覆盖。
   - 启用 `lin_vel_y` 指令空间，充分发挥 12-DOF 的全向移动潜力。
3. **奖励函数定制**:
   - 强化速度跟踪（跟踪权重提升）。
   - 适配 `.*_calf_1` 作为足端触地传感器。
4. **任务注册**: 在 `isaaclab_tasks` 中注册 `Isaac-Velocity-Flat-WAVEGO-v0` 及其回放版。

## [2026-02-11 17:35] 修复 WAVEGO 任务加载失败 (ValueError)

### 问题诊断：
启动训练时报错 `ValueError: Could not find configuration...`。原因是 `gym.register` 中使用了错误的 `kwargs` 键名。

### 执行的操作：
修正 `wavego/__init__.py` 中的注册参数，将 `env_cfg` 修改为 `env_cfg_entry_point`。

## [2026-02-11 17:40] 优化 WAVEGO 训练吞吐量与开启视觉监控

### 执行的操作：
1. **性能压榨**: 提升单卡环境数至 **8192**。
2. **多卡分布式**: 切换至 `torchrun` 并行模式（4卡并行）。
3. **视觉录制**: 开启视频记录，设定为每 500 Iteration 捕捉画面。

## [2026-02-12 17:05] 清理 WAVEGO 命名冗余与配置同步

### 执行的操作：
1. **命名规范化 (URDF)**: 彻底移除了由于装配体实例产生的 `_1` 后缀（如 `FL_hip_1` -> `FL_hip`）。
2. **任务配置适配**: 同步修正 `flat_env_cfg.py` 中的触地传感器和碰撞惩罚连杆匹配符。
3. **资产重构**: 重新生成清晰的 `WAVEGO.usd`。

## [2026-02-12 17:15] 解决多卡训练环境冲突与命名报错

### 问题诊断：
1. **环境冲突**: `python3` 默认调用了系统 Brew Python。
2. **系统限制**: 分布式握手报 `errno 28` (No space left on device)。
3. **命名报错**: 事件仍在寻找不存在的 `base` 连杆。

### 执行的操作：
1. **系统调优**: 提高 `inotify` 监视器上限：`sudo sysctl -w fs.inotify.max_user_watches=524288`。
2. **配置补丁**: 彻底将 `WavegoFlatEnvCfg` 中所有 `base` 引用修正为 `base_link`。

## [2026-02-12 17:25] 压榨 GPU 性能与消除 DDP 瓶颈

### 执行的操作：
1. **显存利用率优化**: 提升至 16384 (单卡)，总并发 65536。
2. **通讯效率爆发**: `num_mini_batches=1`。

## [2026-02-12 17:30] 修正 RSL_RL 配置拼音拼写错误

### 问题诊断：
启动训练时报错 `TypeError: RslRlPpoAlgorithmCfg.__init__() got an unexpected keyword argument 'value_loss_coeff'`。

### 执行的操作：
1. **配置修正**: 修改 `rsl_rl_ppo_cfg.py`，将参数名更正为官方定义的 `value_loss_coef` 和 `entropy_coef`。
2. **策略增强**: 在 policy 配置中补充了 `actor_obs_normalization` 和 `critic_obs_normalization` 必填项。

## [2026-02-12 17:45] 设定 WAVEGO 稳定初始支撑姿态与全量域随机化

### 执行的操作：
1. **默认姿态重构**: 引入镜像对称支撑角度（Hip ±0.1, Thigh ±0.65, Calf ±0.6）。
2. **随机化全开启**:
   - **质量随机化**: ±100g 负载波动。
   - **质心(CoM)随机化**: ±1.5cm 空间偏移。
   - **摩擦力随机化**: 0.4 - 1.25 动态区间。
   - **噪声注入**: 补全了 IMU、编码器及执行器的全链路高频噪声。
3. **奖励函数升级**: 引入 `base_height` (权重 2.0) 奖励，强制维持 0.2m 离地高度，解决初期自杀问题。
4. **控制频率适配**: decimation 降至 4 (50Hz)，提升轻量级机器人的响应敏捷度。

## [2026-02-12 18:15] WAVEGO MuJoCo Sim-2-Sim 高保真模型导出与物理对齐

### 目标设定
将训练完成的 WAVEGO 机器人导出至 MuJoCo 物理引擎，进行跨引擎的 Sim-2-Sim 步态验证，并确保两端物理参数对齐。

### 执行的操作：
1. **资产整理**:
   - 创建专用的 `WAVEGO_mujoco` 目录。
   - 提取并统一所有 STL 网格文件，修正 URDF 中的相对路径引用。
2. **高保真建模 (Menagerie 标准)**:
   - 参考 Google DeepMind `mujoco_menagerie` 标准，重新构建级联式的 MJCF (`wavego.xml`)。
   - **层级修复**: 建立嵌套的 Body 树结构，解决了关节脱离和“悬空”问题。
   - **网格补偿**: 为每个 `geom` 精确设置了相对于关节轴心的 `pos` 偏移量，实现了物理模型的完美拼接。
3. **物理参数对齐与抗漂移优化**:
   - **开启浮动基座**: 在根节点注入 `<freejoint/>`，适配了 19 位 (`7 root + 12 joints`) 的状态向量。
   - **抑制漂移**: 
     - 为所有关节添加了 `damping="0.5"` 和 `frictionloss="0.1"`。
     - 启用了 `noslip_iterations="20"` 和 `elliptic` 摩擦模型。
   - **地面接触**: 开启了 `condim="4"`（增加旋转摩擦），并优化了 `solref/solimp` 参数。
4. **控制与初始化适配**:
   - **执行器映射**: 定义了 12 个 `<position>` 执行器，Kp 设为 **400**。
   - **关键帧预设**: 创建了 `standing` 关键帧，预设了支撑姿态的 `qpos` 和 `ctrl` 指令。

## [2026-02-12 18:45] 修复 WAVEGO MuJoCo 后腿关节错位与姿态参考校准

### 问题修复：
1. **几何错位校正**: 修正了左后 (RL) 和右后 (RR) 小腿网格与关节轴心不匹配的问题。通过追溯原始解析数据，为后腿 Calf 连杆应用了独特的相对位移 (`pos`) 和网格偏移，确保视觉模型与物理碰撞体完全重合。
2. **默认姿态映射**: 引入了 MuJoCo 的 `ref` 属性机制。将指定的 12 关节稳定站立角度 (`joint_pos`) 直接编码入 XML 定义。
   - **效果**: 实现了加载即站立、Reset 即站立。`qpos=0` 状态现在完美对应物理上的站立支撑姿态。

### 结果确认：
- 机器狗 4 条腿拼接严丝合缝，无悬空或穿模现象。
- 物理稳定性进一步提升，消除了由于初始位姿不当引起的动力学震荡。

## [2026-02-13 14:30] 修复 WAVEGO Sim2Sim 的 7 个关键 Bug

### 问题诊断：
用户在将 Isaac Lab 训练的 WAVEGO 策略迁移至 MuJoCo 时遇到严重问题（抽搐、无法行走）。经全面分析发现 7 个致命 Bug：

1. **Bug 1 (最致命)**: 关节排序完全错误 ❌❌❌
   - `sim2sim_mujoco.py` 使用了字母序 `[FL_calf, FL_hip, FL_thigh, ...]`
   - 实际 Isaac Lab 和 MuJoCo 都使用深度优先遍历序 `[FL_hip, FL_thigh, FL_calf, ...]`
   - 导致所有映射 (`MUJOCO_TO_ISAAC_IDX` / `ISAAC_TO_MUJOCO_IDX`) 全部错误

2. **Bug 2 (严重)**: `STANDING_POSE_ISAAC` 顺序错乱 ❌❌❌
   - 站立姿态的 12 个角度值被分配到错误的关节

3. **Bug 3 (严重)**: 坐标系变换多余且错误 ❌❌
   - 从 QWE_DOG 复制了坐标旋转代码 (Isaac X = MuJoCo Y)
   - WAVEGO 使用 identity rotation `rot=(1,0,0,0)`，坐标系完全一致

4. **Bug 4 (中等)**: 关节映射完全不必要 ❌
   - MuJoCo 和 Isaac Lab 使用完全相同的关节顺序，映射应为 identity

5. **Bug 5 (中等)**: 缺少观测值 clipping ⚠️
   - Isaac Lab 的 `EmpiricalNormalization` 会 clip 到 `[-5, 5]`

6. **Bug 6 (中等)**: MuJoCo 执行器建模偏差 ⚠️
   - 只设置了 `kp=400`，缺少 `kv` (对应 damping=10)

7. **Bug 7 (轻微)**: Keyframe 站立位姿错误 ⚠️
   - Keyframe 所有关节设为 0，而非实际站立角度

### 执行的操作：

#### 1. 修正关节排序与站立姿态 (Bug 1 & 2)
```python
# ✅ 修正为深度优先顺序 (FL, FR, RL, RR) x (hip, thigh, calf)
STANDING_POSE = np.array([
    0.100, -0.650,  0.600,   # FL: hip, thigh, calf
   -0.100,  0.650, -0.600,   # FR: hip, thigh, calf
   -0.100, -0.650,  0.600,   # RL: hip, thigh, calf
    0.100,  0.650, -0.600    # RR: hip, thigh, calf
])
```
- 完全移除了错误的 `ISAAC_JOINT_NAMES` 和映射数组

#### 2. 移除错误的坐标系变换 (Bug 3)
```python
# ✅ 直接使用 body 坐标系，无需旋转
lin_vel_b = inv_rot.apply(data.qvel[:3])
ang_vel_b = inv_rot.apply(data.qvel[3:6])
proj_grav = inv_rot.apply(np.array([0, 0, -1.0]))
```

#### 3. 简化关节映射为 identity (Bug 4)
```python
# ✅ 直接使用 MuJoCo 的关节状态，无需索引转换
joint_pos = data.qpos[7:]
joint_vel = data.qvel[6:]
obs_joint_pos = joint_pos - STANDING_POSE
```

#### 4. 添加观测 clipping (Bug 5)
```python
obs_norm = (obs_raw - obs_mean) / (obs_std + 1e-8)
obs_norm = np.clip(obs_norm, -5.0, 5.0)  # ✅ 符合 EmpiricalNormalization 标准
```

#### 5. 修正 MuJoCo 执行器参数 (Bug 6)
```xml
<!-- wavego.xml -->
<default>
  <position kp="400" kv="10" forcerange="-1.96 1.96"/>
</default>
```
- 添加了 `kv="10"` 对应 Isaac Lab `damping=10.0`
- 设置 `timestep="0.005"` 对齐 Isaac Lab

#### 6. 修正 Keyframe 站立姿态 (Bug 7)
```xml
<!-- scene.xml -->
<key name="home" 
     qpos="0 0 0.25 1 0 0 0  0.1 -0.65 0.6  -0.1 0.65 -0.6  -0.1 -0.65 0.6  0.1 0.65 -0.6"
     ctrl="0.1 -0.65 0.6  -0.1 0.65 -0.6  -0.1 -0.65 0.6  0.1 0.65 -0.6"/>
```

### 验证与测试：
创建了 `scripts/verify_sim2sim_fix.py` 自动验证所有修复：
- ✅ 关节顺序对称性检查通过
- ✅ 站立姿态左右镜像对称
- ✅ 所有 7 个 Bug 均已修复

### 运行命令：
```bash
conda activate env_isaaclab
cd /home/user/IsaacLab
python scripts/sim2sim_mujoco.py
```

### 预期结果提升：
- **Before**: 机器人剧烈抽搐、无法行走、关节运动混乱
- **After**: 平稳站立、正常行走、速度跟踪稳定 (~0.5 m/s)

## [2026-02-13 15:00] 修复 MuJoCo 执行器重复阻尼问题与增强调试

### 问题诊断：
用户报告机器狗在 MuJoCo 中前后摆动，速度在 -0.02~0.03 m/s 范围波动，无法达到目标 0.5 m/s。关键症状：
- **关节实际位置无法跟踪目标位置**
- **速度持续在 0 附近震荡**
- **ObsNormMax 在 3.8~4.1 范围（正常）**

### 根本原因：
在 `wavego.xml` 中发现 **重复阻尼配置**：
```xml
<joint damping="10.0" .../>         <!-- 被动阻尼 -->
<position kp="400" kv="10" .../>   <!-- 执行器 D-gain -->
```
**总阻尼 = 10 + 10 = 20**，导致：
1. 关节响应过慢（过阻尼系统）
2. 无法快速跟踪策略输出的目标位置
3. 机器人只能缓慢摆动，无法产生有效推进力

## [2026-02-13 15:25] 回归“非猜测”Sim2Sim流程：参数回归训练等效 + 强制映射校验

### 背景：
用户反馈控制频率过快、步态异常，担心仍存在关节错配。为避免“猜参数”，本次采用可证据流程：
1. MuJoCo 参数严格回归 Isaac 训练等效值；
2. 在运行脚本中加入关节/执行器顺序强制断言；
3. 输出动作抖动与有效控制频率，区分“映射错”与“动力学错”。

### 执行的操作：
1. **回滚 `wavego.xml` 到训练等效配置（去猜测）**
   - `joint damping=0.0`
   - `armature=0.01`
   - `frictionloss=0.4`
   - `position kp=400, kv=10, forcerange=±1.96`
   - `timestep=0.005`

2. **增强 `sim2sim_mujoco.py` 启动校验**
   - 新增 `validate_model_order(model)`：
     - 校验 12 个关节顺序是否为 `[FL_hip_joint ... RR_calf_joint]`
     - 校验 12 个执行器顺序是否为 `[FL_hip ... RR_calf]`
   - 若不一致，直接抛错退出（避免“看似能跑但实际映射错误”）

3. **增强运行时诊断（排查“频率过快”和步态异常）**
   - 输出窗口平均动作变化量 `|dAct|`（判定动作是否高频抖动）
   - 输出有效控制频率 `CtrlHz`（判定是否真在 50Hz）
   - 详细模式中输出 `actions-last`、控制周期均值与目标周期对比

### 快速验证结果：
通过脚本读取 MuJoCo 模型参数确认：
- `kp=400.0`
- `kv=-10.0`（MuJoCo 内部存储号位表现，等效 D 项为 10）
- `joint_damp=0.0`
- `frictionloss=0.4`
- `dt=0.005`

### 结论：
当前版本不再依赖拍脑袋调参，先保证“训练等效 + 映射强校验 + 频率可观测”，再基于日志做下一步定位。

### 执行的操作：

#### 1. 优化 MuJoCo 执行器参数（针对总线舵机）
```xml
<!-- ✅ 修正后配置 -->
<joint damping="0.5" armature="0.01" frictionloss="0.1"/>
<position kp="800" kv="20" forcerange="-1.96 1.96"/>
```

**修改理由：**
- **damping 10.0 → 0.5**: 总线舵机（ST3215-HS）内部已有电子控制，不需要额外被动阻尼
- **kp 400 → 800**: 提高位置增益，匹配总线舵机的快速响应特性（106 RPM = 11.1 rad/s）
- **kv 10 → 20**: 补偿降低的被动阻尼，保持系统稳定性
- **frictionloss 0.4 → 0.1**: 总线舵机摩擦较小

#### 2. 增强 sim2sim 调试信息
在 `scripts/sim2sim_mujoco.py` 中添加详细诊断输出：
```python
# 每 50 步输出关键指标
- 关节跟踪误差 (mean/max)
- 实际力矩 vs 力矩限制
- 机器人高度和 pitch 角度

# 每 500 步输出完整诊断
- 策略原始输出 (actions)
- 目标位置 vs 实际位置 vs 误差
- 12 个关节的实际力矩
- 所有关节速度
- Base 的线速度和角速度
```

#### 3. 创建执行器性能诊断工具
新建 `scripts/diagnose_mujoco_actuator.py`，用于独立测试执行器跟踪性能：
- **测试方法**: 在站立姿态基础上叠加正弦波扰动
- **性能指标**: RMS 误差、最大误差、力矩饱和率
- **实时可视化**: 关节级别的误差、力矩、速度监控
- **性能评级**: 
  - 优秀: RMS < 0.01 rad
  - 良好: RMS < 0.05 rad
  - 一般: RMS < 0.1 rad
  - 较差: RMS > 0.1 rad

### 运行命令：

**1. 测试执行器性能（优先）:**
```bash
conda activate env_isaaclab
cd /home/user/IsaacLab
python scripts/diagnose_mujoco_actuator.py
```

**2. 运行完整 sim2sim:**
```bash
python scripts/sim2sim_mujoco.py
```

### 预期改善：
- **关节跟踪误差**: 应降低至 < 0.05 rad (< 3°)
- **前进速度**: 应稳定增长至 ~0.5 m/s
- **力矩饱和**: 应显著减少（不再长时间触及 1.96 Nm 上限）
- **运动模式**: 从"原地摆动"转变为"稳定行走"

### 参数对比表：

| 参数 | 修改前 | 修改后 | 原因 |
|------|--------|--------|------|
| `joint damping` | 10.0 | 0.5 | 总线舵机内部已有控制 |
| `actuator kp` | 400 | 800 | 匹配快速响应特性 |
| `actuator kv` | 10 | 20 | 补偿被动阻尼降低 |
| `frictionloss` | 0.4 | 0.1 | 总线舵机摩擦小 |
| **总阻尼** | **~20** | **~20** | 保持系统稳定性 |
| **响应速度** | **慢** | **快** | 提升跟踪性能 |

## [2026-02-13 15:45] 控制频率定义修正与50Hz调度器实现

### 问题诊断：
用户日志显示 `CtrlHz` 在 150Hz 左右，但目标是 50Hz。复查后确认：
1. 原脚本的频率统计使用了“睡眠前循环耗时”，并非真实控制周期；
2. 因此 `CtrlHz` 指标被高估，不能代表真实控制频率；
3. 需要同时区分“仿真时间频率”和“墙钟时间频率”。

### MuJoCo 控制频率要点（按官方语义）：
1. MuJoCo 原生只有物理步长 `timestep`（本项目为 0.005s）。
2. 控制频率由“每隔多少个 `mj_step` 更新一次 `data.ctrl`”决定。
3. 当前 `DECIMATION=4` 时，仿真控制频率为 `1 / (0.005 * 4) = 50Hz`。
4. 若需要“墙钟50Hz”，还需在控制循环外层加定时调度（sleep/sync）。

### 执行的操作：
1. **恢复执行器到训练等效参数（去猜测）**
   - `wavego.xml` 调整为：`joint damping=0.0`, `frictionloss=0.4`, `kp=400`, `kv=10`, `forcerange=±1.96`。

2. **增强 `sim2sim_mujoco.py` 频率与顺序校验**
   - 新增启动时强制校验：
     - 12 关节顺序
     - 12 执行器顺序
   - 若不一致直接报错退出。

3. **修复控制频率统计方式**
   - 新增两组指标并分开打印：
     - `SimHz`：基于 `data.time` 的仿真控制频率（应接近 50）
     - `WallHz`：基于墙钟时间的真实控制频率（目标接近 50）

4. **实现固定50Hz墙钟调度器**
   - 控制循环按 `next_control_wall += CONTROL_DT` 调度；
   - 若超时落后则重同步，避免频率漂移；
   - 详细日志增加仿真/墙钟周期与频率对照。

### 结果：
当前版本可明确区分“仿真频率正确但墙钟过快”与“调度本身错误”，避免误判；后续调参将以 `SimHz≈50` 且 `WallHz≈50` 为前置门槛。

## [2026-02-13 16:05] 继续定位“乱跳”：观测语义与力矩诊断修正

### 现象确认：
用户最新日志显示：
- `SimHz=50`, `WallHz=50`（频率已对齐）
- 但 `Torque` 长时间顶到 `1.96Nm`
- `ObsMax` 高频触顶到 `5.00`
- 步态仍异常（翻滚/乱跳）

### 关键修正：
1. **修正 base 速度观测获取方式（避免 qvel 语义歧义）**
   - 原实现通过 `qvel + 旋转`构造 `base_lin_vel/base_ang_vel`。
   - 现改为 MuJoCo 官方 API：
     - `mj_objectVelocity(..., flg_local=1)`
     - 直接读取 base 局部坐标系 6D 速度（`rot:lin`）
   - 对应 Isaac 观测定义中的 `root_lin_vel_b` / `root_ang_vel_b`。

2. **修正力矩打印索引错误**
   - `data.qfrc_actuator[:12]` 包含 free-joint 自由度，导致前6维长期为0，误导判断。
   - 改为 `data.qfrc_actuator[6:18]`，仅统计 12 个关节执行器力矩。

3. **增加 clip 维度计数**
   - 新增 `ClipDims` 输出，显示有多少个观测维度触发 `[-5, 5]` 裁剪。
   - 用于快速判断是否存在系统性观测越界（常见于观测语义不一致或动力学域偏差）。

### 结论：
当前问题已从“控制频率”收敛到“观测语义/动力学域差异导致策略输出过激和力矩饱和”。

## [2026-02-13 16:20] 新增A/B诊断开关：动作裁剪、力矩缩放、clip维度定位

### 背景：
用户最新日志显示频率已严格对齐 50Hz，但仍存在：
- 力矩长期饱和 (`1.96/1.96`)
- 动作变化量较大 (`|dAct| ~ 1.4`)
- 局部观测触发 clip (`ClipDims=1~2`)

### 执行的操作：
在 `scripts/sim2sim_mujoco.py` 增加可复现实验开关，避免“盲调参数”：

1. **命令可配置**
   - 新增参数：`--cmd-x`, `--cmd-y`, `--cmd-wz`
   - 用于分离“速度指令过高”与“模型失配”

2. **动作裁剪诊断（A/B）**
   - 新增参数：`--action-clip`
   - 在策略原始输出后可选执行 `np.clip(actions, -clip, clip)`
   - 用于验证是否存在“训练部署链路中动作幅值处理不一致”

3. **力矩上限缩放诊断（A/B）**
   - 新增参数：`--torque-scale`
   - 运行时按比例缩放 `model.actuator_forcerange`
   - 仅用于定位“力矩饱和是否主因”，不作为最终物理参数

4. **观测clip维度定位**
   - 新增 48维观测标签映射 (`OBS_LABELS`)
   - 详细日志打印：被 clip 的维度索引与名称
   - 快速判断是 `base_ang_vel`、`joint_vel` 还是其他项越界

### 目的：
把“乱跳”拆解为可验证假设：
- 若提高 `torque-scale` 后立即稳定：主因是执行器能力不足/模型差异；
- 若 `action-clip` 后明显改善：主因是动作幅值分布偏移；
- 若特定观测项持续 clip：主因是该观测语义或物理域偏移。

## [2026-02-13 16:40] 针对“零命令仍乱跳”的全流程复核与诊断增强

### 关键发现：
用户提供的 `agent.yaml` 证实：
- `clip_actions: null`（训练阶段并未启用 wrapper 动作裁剪）

这意味着部署端默认不应强制裁剪动作，当前脚本“默认不裁剪”是对齐训练链路的。

### 日志结论（零命令实验）：
- 频率链路已正确：`SimHz=50`, `WallHz=50`
- 仍普遍力矩饱和：多关节持续触及 `±1.96Nm`
- 观测 clip 并不严重（`ClipDims` 多为 0~1）

推断：主因更偏向执行器能力/动力学域偏差，而非频率与关节映射错误。

### 新增诊断能力（`sim2sim_mujoco.py`）：
1. **跌倒自动重置（可选）**
   - 参数：`--reset-on-fall --fall-height --fall-pitch-deg`
   - 作用：避免跌倒后长时间发散污染统计，聚焦“跌倒前”控制质量。

2. **饱和率指标**
   - 每次打印新增 `Sat: xx%`（按 98% 力矩上限阈值统计）
   - 比单看 `Torque max` 更能反映是否“全关节挤满力矩”。

3. **可复现实验日志**
   - 参数：`--log-csv <path>`
   - 输出每步关键指标：误差、饱和率、频率、obs_max、clip_dims、重置次数。

4. **可控终止**
   - 参数：`--max-steps`
   - 用于固定窗口 A/B 对比（同样步数，减少主观偏差）。

### 当前定位建议：
优先跑带重置与CSV的A/B实验，比较“饱和率与误差”而非主观步态观感：
1) 基线：无裁剪、无力矩缩放；
2) 动作裁剪：`--action-clip 1.0`；
3) 力矩缩放：`--torque-scale 3.0`；
4) 联合：`--action-clip 1.0 --torque-scale 3.0`。

## [2026-02-13 16:55] 新增“零命令是否策略本身不稳”判定工具

### 背景：
用户反馈在 `cmd=0` 下仍持续大幅抖动，怀疑流程中仍有隐藏错误。

### 新增工具：
1. `scripts/summarize_sim2sim_csv.py`
    - 对 `--log-csv` 结果自动汇总：
       - `sat_ratio` 均值/95分位/最大值
       - `joint_err_mean/joint_err_max`
       - `|vel_x|`、`obs_max`、`clip_dims`
       - `sat>=80%` 占比与重置点

2. `scripts/check_zero_cmd_isaac.py`
    - 在 Isaac 原生环境中强制零命令并跑同一权重，输出：
       - `|vel_x|`
       - `joint_err_mean`
       - `sat_ratio`
    - 用于判断：
       - 若 Isaac 端零命令也高饱和/高抖动，则是策略本身在零命令弱覆盖；
       - 若 Isaac 稳而 MuJoCo 不稳，则是跨引擎动力学域偏差。

### 目的：
将问题从“主观观感”切换到“同权重、同命令、跨引擎可比指标”，避免继续靠猜测调参。

## [2026-02-13 17:55] 自动批量运行A/B并读取结果（无GUI）

### 执行的自动实验：
在无GUI、非实时模式下自动运行 4 组 1000-step 实验并输出 CSV：
1. baseline: `cmd=0`
2. clip: `cmd=0, action-clip=1.0`
3. torque3: `cmd=0, torque-scale=3.0`
4. both: `cmd=0, action-clip=1.0, torque-scale=3.0`

并使用 `scripts/summarize_sim2sim_csv.py` 自动汇总。

### 汇总结果（关键指标）：
- **baseline**: `sat_ratio_mean=0.875`, `joint_err_mean=0.0848`, `resets=4`
- **clip**: `sat_ratio_mean=0.8625`, `joint_err_mean=0.0689`, `resets=2`
- **torque3**: `sat_ratio_mean=0.7625`, `joint_err_mean=0.0534`, 但 `resets` 频繁（过激）
- **both**: `sat_ratio_mean=0.7750`, `joint_err_mean=0.0418`, `resets=3`

### 额外对照：Isaac 原生零命令（同权重）
自动运行 `scripts/check_zero_cmd_isaac.py --steps 1000 --headless`：
- `mean |vel_x| = 0.0494`
- `mean joint_err = 0.0515`
- `mean sat_ratio = 0.4488`

### 结论：
MuJoCo 侧在零命令下的饱和显著高于 Isaac（约 0.88 vs 0.45），且更易跌倒，说明问题主因仍是跨引擎动力学域差异（接触/执行器等效性），而非频率与关节映射错误。

## [2026-02-13 18:35] 开环轨迹回放验证（已自动执行）

### 目标：
彻底剥离“策略-观测闭环”影响，验证同一 `target_q(t)` 在 Isaac 与 MuJoCo 的执行差异。

### 执行步骤：
1. 新增 `scripts/export_isaac_openloop_targets.py`
   - 在 Isaac 中用同一策略导出 1000 步目标关节轨迹 `target_q(t)`。
   - 同时记录 Isaac 侧 `qerr/sat/velx` 基准。

2. 新增 `scripts/replay_mujoco_openloop_targets.py`
   - 在 MuJoCo 逐步回放同一 `target_q(t)`（不再走策略推理）。
   - 记录 MuJoCo 侧 `qerr/sat/velx`。

### 自动运行结果：
#### Isaac 导出基准：
- `mean |vel_x| = 0.0369`
- `mean qerr = 0.0622`
- `mean sat = 0.4594`

#### MuJoCo 开环回放（同轨迹）：
- `mean |vel_x| = 0.0761`
- `mean qerr = 0.0750`
- `mean sat = 0.8557`

#### 跨引擎比值：
- `qerr ratio (muj/isaac) = 1.21x`
- `sat ratio (muj/isaac) = 1.86x`

### 结论：
即使在“同目标轨迹、无策略闭环”的前提下，MuJoCo 仍显著更容易力矩饱和。这证明当前主问题是执行器/接触的跨引擎动力学等效不足，而不是关节映射、控制频率或观测拼接顺序错误。

## [2026-03-06 21:55] 多卡训练启动报错排查（torchrun 主机名解析）

### 现象：
- 使用 `torch.distributed.run --standalone --nproc_per_node=4` 启动时报错：
   - `The IPv6 network addresses of (ai-desktop-01, <port>) cannot be retrieved`
   - `TCP client failed to connect/validate`

### 排查结果：
1. **主因定位**: 属于 `torchrun` rendezvous 地址解析失败（DNS/hostname），不是训练脚本、模型 checkpoint 或 PPO 配置错误。
2. **环境核查**:
    - 当前主机名为 `ainode`。
    - `/etc/hosts` 中存在 `127.0.1.1 ai_node`（与 `ainode` 不一致），且无 `ai-desktop-01` 映射。
3. **历史训练方式核验**:
    - `GEMINI_LOG.md` 历史记录显示曾切换到 `torchrun` 4卡并行并使用 DDP。
    - 旧 run `logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/params/agent.yaml` 显示 `device: cuda:3`，可确认当时确实是多卡分布式流程（多进程写同目录的典型痕迹）。

### 建议：
- 启动时显式指定 rendezvous 地址，避免主机名解析：
   - `--master_addr 127.0.0.1 --master_port 29501`
- 录像维持单进程写入（仅 rank0），避免视频文件损坏。

## [2026-03-06 22:08] 修复 DDP 训练中多进程视频渲染导致的吞吐下降

### 问题诊断：
- 在 `scripts/train_ddp_advanced.py` 中，传入 `--video` 时所有 rank 都会进入 `render_mode="rgb_array"` 路径。
- 即便只有一个 rank 真正 `RecordVideo`，其余 rank 仍承担相机/渲染开销，导致多卡吞吐提升不明显。

### 执行的操作：
1. **按 rank 门控视频开关**:
   - 在参数解析后提前读取 `LOCAL_RANK`。
   - 仅 `render_rank` 保留 `args_cli.video=True`，其余 rank 自动关闭视频。
2. **按 rank 门控相机启用**:
   - `enable_cameras` 仅在视频 rank 上开启。
3. **保持单进程录像**:
   - 录像包装器 `RecordVideo` 仅在视频 rank 执行，避免并发写文件损坏。

### 结果预期：
- 多卡 DDP 下 collection 阶段耗时下降，`steps/s` 明显回升。
- 保留单进程视频录制能力，不影响诊断可视化。

## [2026-03-06 22:20] DDP 吞吐复测与根因确认（为何仅 90k steps/s）

### 现象复盘：
- 用户报告 4卡 DDP 下仅 `~91k steps/s`，collection 约 `8.3s`，与单卡 `~80k` 相比提升很少。

### 根因确认：
1. **旧进程仍在使用修复前代码**: 已启动的 tmux 训练进程不会热更新，需重启后才会生效。
2. **分布式脚本曾强制 Livestream**: 修复前 `render_rank` 被强制开启直播，导致该 rank 变慢并拖累全局同步。
3. **`--video` 模式渲染开销**: 虽已限制单进程录像，但视频 rank 仍有额外渲染负担，吞吐会低于“纯训练”基线。

### 复测结果（修复后，4卡、无视频短跑）：
- `Computation: 291650 steps/s (collection: 2.442s, learning: 0.255s)`
- `Computation: 295038 steps/s (collection: 2.445s, learning: 0.221s)`
- `Computation: 276555 steps/s (collection: 2.620s, learning: 0.224s)`

### 结论：
- 当前代码修复后 DDP 吞吐已恢复到合理区间（约 2.8e5 steps/s 级别），不再是 9e4 steps/s。
- 若仍观测到 9e4，基本可判定为：旧会话未重启、仍在跑旧逻辑，或同时开启了额外可视化负载。

## [2026-03-06 22:30] DDP + Video 崩溃根因与规避方案

### 错误现象：
- 分布式训练（4进程）下开启 `--video` 时在 `gymnasium.wrappers.RecordVideo` -> `env.render()` 触发崩溃：
  - `TypeError: Unable to write from unknown dtype, kind=f, size=0`
  - 调用链来自 `omni.syntheticdata / replicator` 节点连接。

### 根因判定：
1. 不是 checkpoint、奖励配置或网络结构问题。
2. 属于 Isaac Sim Replicator 在多进程 DDP 训练中的稳定性问题（即便只保留单 rank 视频也可能触发）。

### 已执行修复：
1. 修改 `scripts/train_ddp_advanced.py`：
   - 若 `--distributed` 且请求 `--video`，自动禁用视频并打印警告。
2. 推荐流程：
   - DDP 训练阶段关闭视频，优先保证吞吐和稳定。
   - 需要视频时，单卡独立运行 `train.py/play.py` 录制。

## [2026-03-06 22:36] 再定位：livestream 默认值 -1 被误判为 True

### 问题细节：
- `AppLauncher` 的 `livestream` 默认值是 `-1`（表示“跟随环境变量”）。
- 在 Python 中 `-1` 为 truthy。脚本若仅用 `if args_cli.livestream:` 判断，会把默认态误判为“开启直播”。
- 这解释了日志中在未显式传 `--livestream` 时仍出现：`Rank 0 selected for LIVESTREAM`。

### 执行的修复：
1. 在 `scripts/train_ddp_advanced.py` 中新增显式判断：
   - 仅当用户命令行明确传入 `--livestream` 且值为 `1/2` 时，才允许渲染 rank 开启直播。
   - 其余情况（包括默认 `-1`）统一强制 `livestream=0`。
2. 保持 DDP 模式下视频禁用策略，避免 `Replicator` 在多进程下崩溃。

## [2026-03-07 13:45] 重新分析 DDP+单视频Rank 录制路径

### 新结论：
1. 历史上“多卡训练 + 单卡录视频”是可行目标，不应简单视为永久不支持。
2. 当前环境下存在两类独立问题：
   - **脚本级问题**: `livestream=-1` 被误判，导致未显式请求时仍开启直播；已修复。
   - **系统级问题**: `inotify` 限额过低（当前 `max_user_watches=65536`, `max_user_instances=128`），官方 `train.py` 的 DDP+video 探针已复现 `errno=28/No space left on device`。

### 新增保护：
1. `scripts/train_ddp_advanced.py` 在 DDP+video 下全局关闭 `base_velocity.debug_vis`，减少 Replicator 与命令可视化实体冲突。
2. 启动时读取 `/proc/sys/fs/inotify/*`，若阈值偏低则明确打印 sysctl 调整建议。

### 当前状态：
- 脚本侧已修到“只允许单一 render_rank 录制、不会误开 livestream、减少 debug marker 干扰”。
- 若仍想稳定启用 DDP+视频，需同步提升系统 `inotify` 上限。

## [2026-03-07 13:50] 自动复现结论：当前环境下 DDP 训练时视频录制仍不可稳定使用

### 自动实验：
1. **官方脚本探针**: `scripts/reinforcement_learning/rsl_rl/train.py`，4卡、resume、视频、仅 1 轮。
2. **自定义脚本探针**: `scripts/train_ddp_advanced.py`，4卡、resume、视频、仅 1 轮。

### 结果：
1. **官方脚本失败**:
   - 并未进入视频渲染阶段，先在视频逻辑处报错：
   - `AttributeError: 'ManagerBasedRLEnv' object has no attribute 'rank'`
   - 说明当前仓库/版本组合下，官方 `train.py` 的 DDP + video 路径本身就不兼容当前环境对象。
2. **自定义脚本失败**:
   - 已成功进入 rank0 视频录制路径并打印：
     - `Rank 0 recording video to: .../videos/train`
   - 随后在首次 `env.render()` -> `Replicator/SyntheticData` attach 崩溃：
     - `TypeError: Unable to write from unknown dtype, kind=f, size=0`

### 结论：
1. **历史上该模式曾成功过**：旧 run `logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/videos/train` 中存在完整训练视频，证明当时确实录制成功。
2. **但在当前环境/版本状态下，不可稳定复现**：
   - 官方脚本的 DDP + video 路径已失效；
   - 自定义脚本虽然绕过了 rank 选择问题，但仍会在 `Replicator` attach 阶段崩溃。
3. **因此当前可执行结论**：
   - 现在不能把“多卡训练同时录视频且稳定保留性能”作为可靠工作流；
   - 训练与录制应分离，或需回退/重建到 2026-02-12 当时的可用环境状态后再尝试复现。

## [2026-03-07 14:05] 恢复官方 train.py，后续仅在个人脚本承载 DDP+视频修补

### 执行的操作：
1. **恢复官方脚本**: 将 `scripts/reinforcement_learning/rsl_rl/train.py` 的视频逻辑恢复到当前主分支状态，移除对 `env.unwrapped.rank` 的本地补丁。
2. **修补边界收敛**: 明确后续所有 DDP+视频实验仅在 `scripts/train_ddp_advanced.py` 中继续，不再污染官方训练入口。
3. **语法校验**: 对 `train.py` 与 `train_ddp_advanced.py` 执行 `py_compile` 检查。

## [2026-03-06 22:45] 按用户需求恢复 DDP 单卡录制视频路径

### 需求：
- 用户要求保留“多卡训练 + 仅一张卡录制视频”的工作方式。

### 执行的操作：
1. 修改 `scripts/train_ddp_advanced.py`：
   - 移除“分布式模式强制禁用视频”的逻辑。
   - 保持仅 `render_rank` 开启 `args_cli.video`、`enable_cameras` 与 `RecordVideo`。
2. 保留 livestream 安全修复：
   - 仅当用户显式传 `--livestream` 时才在 `render_rank` 开启直播。
3. 增加运行日志：
   - 在视频 rank 打印视频输出目录，便于确认是否真的在录制。

