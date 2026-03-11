# 01 — Jetson Nano B01 环境配置指南

> 适用：NVIDIA Jetson Nano B01 (4GB)  
> 目标：配置能以 50 Hz 运行 WAVEGO 12-DOF 策略推理的完整环境。

---

## 目录

1. [方案选择：原生 vs Docker](#1-方案选择)
2. [方案 A：原生环境配置（推荐）](#2-方案a原生环境配置推荐)
3. [方案 B：Docker 容器配置](#3-方案bdocker-容器配置)
4. [验证清单](#4-验证清单)

---

## 1. 方案选择

| 对比项 | 原生环境 | Docker 容器 |
|---|---|---|
| 复杂度 | 中 | 高 |
| 性能开销 | 无 | 极低（约 1-2%） |
| 可复现性 | 依赖手动配置 | 镜像标准化 |
| 串口/I2C 访问 | 直接 | 需 `--device` 映射 |
| GPU 访问 | 直接 | 需 `nvidia-docker` |
| **推荐度** | **⭐⭐⭐ 首选** | 适合多机器/CI 场景 |

**建议**：首次部署使用**原生环境**（方案 A），降低调试难度。确认跑通后再考虑 Docker 化。

---

## 2. 方案 A：原生环境配置（推荐）

### 2.1 系统确认

```bash
# 确认 JetPack 版本
cat /etc/nv_tegra_release
# 应输出 R32 (JetPack 4.6) 或 R35 (JetPack 5.x)

# 确认 GPU
nvidia-smi  # Nano 上可能无 nvidia-smi，用以下替代
cat /proc/device-tree/model
# 应输出 "NVIDIA Jetson Nano Developer Kit"

# 确认 CUDA
nvcc --version
# 应输出 CUDA 10.2 (JP4.6) 或 CUDA 11.4 (JP5.x)
```

**验收标准**：上述三条命令均有合理输出，无报错。

### 2.2 安装 Miniforge (Conda)

Jetson Nano (aarch64) 不能用 Anaconda/Miniconda x86 包，需用 **Miniforge**：

```bash
# 下载 Miniforge (aarch64)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh

# 安装
chmod +x Miniforge3-Linux-aarch64.sh
./Miniforge3-Linux-aarch64.sh -b -p $HOME/miniforge3

# 初始化
$HOME/miniforge3/bin/conda init bash
source ~/.bashrc
```

**验收标准**：`conda --version` 输出版本号。

### 2.3 创建部署环境

```bash
# 创建 Python 3.8 环境（JetPack 4.6 最高稳定支持）
conda create -n wavego_deploy python=3.8 -y
conda activate wavego_deploy

# 安装基础依赖
pip install numpy==1.24.4
pip install pyserial==3.5
pip install pyyaml==6.0.1
```

### 2.4 安装 ONNX Runtime (GPU)

**JetPack 4.6 (CUDA 10.2)**：

```bash
# 方法 1: 从 Jetson Zoo 下载预编译包
# 查找最新地址: https://elinux.org/Jetson_Zoo#ONNX_Runtime
wget https://nvidia.box.com/shared/static/zostg6agm00fb6t5uisw51qi6khdvber.whl \
  -O onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl
pip install onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl
```

**JetPack 5.x (CUDA 11.4)**：

```bash
# JetPack 5 可直接 pip 安装
pip install onnxruntime-gpu
```

**如果找不到预编译包，回退到 CPU 版本**：

```bash
pip install onnxruntime==1.17.0
```

> **注意**：CPU 版本对 48→12 维的小网络推理延迟约 0.5-2ms，50 Hz 控制完全够用。GPU 版本延迟约 0.3-1ms，提升不显著。WAVEGO 的策略网络很小（~25 万参数），CPU 推理即可满足实时性。

**验收标准**：

```bash
python -c "import onnxruntime as ort; print(ort.get_device()); print(ort.__version__)"
# 输出 'GPU' 或 'CPU' + 版本号
```

### 2.5 配置 STM32 USB CDC 串口

> **关键**：IMU (WT61C) 和舵机 (STS3215) 都通过 STM32 中转，Jetson Nano 上只需一个 USB CDC 串口，**无需 I2C / smbus2**。

```bash
# 安装串口权限工具和查看工具
sudo apt-get install -y cu

# 将当前用户加入 dialout 组（串口读写权限）
sudo usermod -a -G dialout $USER
# 重新登录后生效
logout
```

**创建 udev 规则以固定设备名**（避免每次插拔 ttyACM 编号变化）：

```bash
sudo tee /etc/udev/rules.d/99-wavego.rules << 'EOF'
# STM32F407 USB CDC (VID=0483, PID=5740)
SUBSYSTEM=="tty", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", \
  SYMLINK+="wavego_stm32", MODE="0666", GROUP="dialout"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
```

**验收标准**：

```bash
# 插入 STM32 USB 线后
ls /dev/ttyACM*             # 应显示 /dev/ttyACM0 (或 ACM1)
ls -la /dev/wavego_stm32    # udev 符号链接应存在
stat -c "%a %G" /dev/ttyACM0  # 应为 666 dialout

# 快速测试串口通信
python -c "
import serial, time
s = serial.Serial('/dev/ttyACM0', 115200, timeout=0.1)
time.sleep(0.1)
print('串口已打开:', s.is_open)
s.close()
"
```


### 2.6 串口基础功能测试

```bash
# 运行 STM32Bridge 快速自测
cd ~/wavego_deploy  # 或 部署目录
python scripts/stm32_bridge.py --port /dev/ttyACM0 --duration 5
# 期望: 输出 Roll/Pitch/Pos 读数，无报错
```



### 2.7 (可选) 安装 TensorRT

如果需要 TensorRT 加速（详见 `04_MODEL_EXPORT_AND_OPTIMIZATION.md`）：

```bash
# JetPack 已预装 TensorRT，确认版本
python3 -c "import tensorrt; print(tensorrt.__version__)"
# JetPack 4.6: TensorRT 8.0.x
# JetPack 5.x: TensorRT 8.5.x

# 在 conda 环境中可能需要设置路径
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
```

---

## 3. 方案 B：Docker 容器配置

> 仅在需要多机标准化部署或原生配置受阻时使用。

### 3.1 安装 nvidia-docker

```bash
# JetPack 4.6 已预装 Docker，确认版本
docker --version

# 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 3.2 Dockerfile

参见 `docker/Dockerfile.jetson`：

```dockerfile
FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3
# JetPack 5.x 使用: FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3

RUN pip3 install --upgrade pip && \
  pip3 install pyserial==3.5 pyyaml==6.0.1

WORKDIR /app
COPY sim2real_test/ /app/sim2real_test/
COPY logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/exported/ /app/model/

CMD ["python3", "/app/sim2real_test/scripts/wavego_inference.py"]
```

### 3.3 运行容器（映射 STM32 USB CDC 设备）

```bash
docker run --runtime nvidia \
  --device /dev/ttyACM0 \
  -v ~/wavego_deploy/config:/app/config \
  wavego_deploy:latest
```

**注意事项**：

- `--device /dev/ttyACM0`：映射 STM32 USB CDC 串口（**不需要** `--device /dev/i2c-1`）
- 不需要 `--privileged`（仅映射特定设备即可）
- 容器内需确认串口设备路径与宿主机一致


---

## 4. 验证清单

完成环境配置后，逐项验证：

| # | 验证项 | 命令 | 预期结果 | ✅ |
|---|---|---|---|---|
| 1 | Python 版本 | `python --version` | 3.8.x | |
| 2 | NumPy | `python -c "import numpy; print(numpy.__version__)"` | ≥1.24 | |
| 3 | ONNX Runtime | `python -c "import onnxruntime; print(onnxruntime.__version__)"` | ≥1.10 | |
| 4 | PySerial | `python -c "import serial; print(serial.__version__)"` | ≥3.5 | |
| 5 | **STM32 串口设备** | `ls /dev/ttyACM*` | `/dev/ttyACM0` 存在 | |
| 6 | **串口读写权限** | `stat -c "%a" /dev/ttyACM0` | `666` | |
| 7 | **STM32 通信** | `python scripts/stm32_bridge.py --duration 3` | 输出 IMU 数据 | |
| 8 | ONNX Runtime 推理 | `python -c "import onnxruntime as ort; s=ort.InferenceSession('config/wavego_policy.onnx'); print('OK')"` | `OK` | |

> ⚠️ **不再需要** `i2cdetect`、`smbus2` 或 `adafruit-*` 库。IMU 数据通过 STM32 USB CDC 回传。

**全部通过后，进入 `02_DEPLOYMENT_PIPELINE.md`。**

