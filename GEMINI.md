# Isaac Lab 项目背景

**说明：根据用户要求，本项目上下文中的后续对话将默认使用中文进行。请你将每次执行完指令后简要记录对文件仓库或环境做了哪些更改追加到GEMINI_LOG.md中，不允许覆盖已有记录，只能追加或者修改。**


## 项目仓库地址: https://github.com/isaac-sim/IsaacLab

## Isaac Sim 官方文档地址: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html

## Isaac Lab 官方文档地址: https://isaac-sim.github.io/IsaacLab/main/index.html

## 机器人硬件参数 (Sim-to-Real 参考)

*   **舵机型号**: MG996R (模拟/数字舵机)
*   **单体重量**: 55g
*   **工作扭矩**: 13 kg·cm (约 1.27 Nm) @ 6.0V
*   **反应转速**: 53-62 RPM (约 5.5 - 6.5 rad/s) -> 仿真取 6.3 rad/s
*   **死区设定**: 4微秒
*   **控制特性**: 仅支持位置控制 (Position Control)，无力矩回传，高减速比带来显著的自锁效应（高静摩擦）。

## 机器人硬件参数 - WAVEGO (12-DOF)

*   **舵机型号**: 微雪 ST3215-HS (串行总线舵机)
*   **单体重量**: 52g (规格参考)
*   **工作扭矩**: 20 kg·cm (约 1.96 Nm) @ 12V
*   **反应转速**: 106 RPM (约 11.1 rad/s)
*   **Kt 常数**: 8.3 kg.cm/A
*   **控制特性**: 12-bit 高精度磁编码位置控制，支持参数回传。

## 项目概述

**Isaac Lab** 是一个专为机器人研究设计的 GPU 加速开源框架，涵盖强化学习 (RL)、模仿学习 (IL) 和运动规划等领域。它构建在 **NVIDIA Isaac Sim** 之上，利用了其强大的物理和传感器仿真能力。

核心特性：
-   **GPU 加速：** 优化仿真速度，专为学习任务设计。
-   **统一工作流：** 支持强化学习、模仿学习以及仿真到现实 (Sim-to-Real) 的迁移。
-   **模块化设计：** 分离为核心库、资产、强化学习接口和任务扩展。

## 目录结构

### 源代码目录 (`source/`)
核心逻辑组织在多个 Python 包（扩展）中：
-   `isaaclab`：包含核心功能的主要库。
-   `isaaclab_assets`：预配置的资产，如机器人、传感器和道具。
-   `isaaclab_rl`：针对强化学习框架（如 rsl_rl, sb3）的接口。
-   `isaaclab_tasks`：基准任务环境集合。
-   `isaaclab_mimic`：模仿学习工具 (mimic)。
-   `isaaclab_contrib`：社区贡献的扩展。

### 脚本目录 (`scripts/`)
包含各种工作流的可执行脚本：
-   `demos/`：展示功能的简单示例（如加载机器人、遥操作）。
-   `environments/`：用于列出环境和运行智能体的脚本。
-   `reinforcement_learning/`：RL 智能体的训练和回放脚本。
-   `tools/`：工具脚本，用于资产转换、数据集处理等。

### 配置
-   `apps/`：Isaac Sim 应用程序的配置文件 (`.kit`)。
-   `docker/`：用于容器化部署的 Dockerfile 和 compose 文件。
-   `docs/`：Sphinx 文档源文件。

## 构建与安装

本项目依赖于已安装或可用的 **Isaac Sim**。

1.  **环境设置：**
    使用提供的包装脚本创建管理环境（Conda 或 UV）：
    ```bash
    # 创建 Conda 环境（默认名称：env_isaaclab）
    ./isaaclab.sh --conda [env_name]

    # 或者 创建 UV 环境
    ./isaaclab.sh --uv [env_name]
    ```
    *注意：激活环境后会创建一个 `isaaclab` 别名指向 `isaaclab.sh`。*

2.  **安装：**
    安装包及其依赖项：
    ```bash
    ./isaaclab.sh --install
    ```
    这将安装 `source/` 中的扩展以及额外的强化学习依赖。

## 运行代码

**请勿直接运行 `python`。** 请使用 `./isaaclab.sh` 包装器（如果环境已激活，可使用 `isaaclab` 别名），以确保加载正确的 Python 解释器和环境变量（包括 Isaac Sim 路径）。

### 运行 Python 脚本
**需要先激活conda环境: env_isaaclab**
```bash
./isaaclab.sh -p path/to/script.py
# 示例：
./isaaclab.sh -p scripts/demos/arms.py
```

### 使用模拟器可执行文件运行
如果需要运行完整的模拟器可执行文件（通常用于 GUI 或特定扩展）：
```bash
./isaaclab.sh -s path/to/script.py
```

### 运行 Docker
```bash
./isaaclab.sh --docker [args]
```

## 开发与测试

-   **格式化与 Lint 检查：**
    运行 pre-commit 钩子（如 Ruff 等）来格式化代码：
    ```bash
    ./isaaclab.sh --format
    ```

-   **测试：**
    运行所有 pytest 测试：
    ```bash
    ./isaaclab.sh --test
    ```

-   **文档：**
    构建 Sphinx 文档：
    ```bash
    ./isaaclab.sh --docs
    ```

-   **VS Code：**
    生成 VS Code 设置（工作区设置）：
    ```bash
    ./isaaclab.sh --vscode
    ```

## 开发规范

-   **代码风格：** 遵循 PEP 8 指南，由 `ruff` 强制执行。配置位于 `pyproject.toml`。
-   **类型检查：** 使用 `pyright`。
-   **文档字符串：** 使用 Google 风格的文档字符串。
-   **许可证：** 大部分使用 BSD-3-Clause，部分组件使用 Apache 2.0。
