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

### 问题诊断：
调试脚本无法识别 Config 中以正则形式定义的 `joint_pos`（如 `shank_.*`），导致滑块初始值被强制归零并覆盖了机器人的默认姿态。

### 执行的操作：
修改 `tune_qwe_dog_joints.py`，引入 `re` 模块进行正则匹配，确保滑块能正确加载配置文件的初始角度。

