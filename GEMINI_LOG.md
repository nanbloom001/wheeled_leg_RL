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
4. **环境维护**: 执行了 `./isaaclab.sh --install` 以确保 Conda 环境中的 `isaaclab` 核心库更新并可用。

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
3. **静摩擦增强**: 为彻底消除漂移，一度尝试将 URDF friction 设为 `5.0`。

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
2. **PhysX 稳定性修复**: 大幅增加了 `gpu_collision_stack_size` 和其他 GPU 缓冲区容量，解决了自碰撞导致的缓冲区溢出崩溃。
3. **环境兼容性修正**: 
   - 禁用了 `base_com` 事件（解决了 `base` 连杆命名冲突）。
   - 在平坦地形任务中移除了 `height_scanner` 及相关的观测项（解决了实体丢失错误）。
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