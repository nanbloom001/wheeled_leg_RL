# 在 Jetson Nano (Ubuntu 18.04) 上部署 Isaac Lab 策略指南

这份指南旨在帮助您将从 Isaac Lab 导出的 ONNX 策略模型部署到 NVIDIA Jetson Nano 上。考虑到 Nano 的硬件限制和 Ubuntu 18.04 的软件环境，我们将采用最稳妥的 **ONNX Runtime** 方案。

## 0. 前置条件

*   **硬件**: Jetson Nano (4GB 版本推荐)
*   **系统**: JetPack 4.6 (Ubuntu 18.04 LTS)
*   **模型**: 导出的 `policy.onnx` 文件 (输入维度: 44, 输出维度: 8)
*   **下位机**: 已连接至 USB 串口 (如 `/dev/ttyUSB0`)

---

## 1. 环境配置 (Jetson Nano)

Jetson Nano 默认的 Python 版本是 3.6，这对于现代深度学习库来说太老了。我们强烈建议**不要**升级系统 Python，而是使用 `python3` (3.6) 配合特定版本的库，或者使用 Virtualenv。

### 1.1 更新系统与基础工具
```bash
sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-base libopenmpi-dev libomp-dev
pip3 install --upgrade pip setuptools wheel
```

### 1.2 安装核心依赖 (Numpy, Serial)
注意：Python 3.6 支持的 Numpy 版本有限，不能直接装最新的。
```bash
# 安装 Numpy (1.19.x 是 Python 3.6 的稳定选择)
pip3 install "numpy<1.20"

# 安装串口通信库
pip3 install pyserial
```

### 1.3 安装 ONNX Runtime (GPU 加速版)
这是最难的一步。Jetson 无法直接通过 `pip install onnxruntime-gpu` 安装。
我们需要下载 NVIDIA 官方或社区提供的预编译 `.whl` 包。

*   **JetPack 4.6 (Python 3.6) 推荐包**:
    *   ONNX Runtime 1.10.0 或 1.11.0 是比较稳定的版本。

**下载并安装:**
```bash
# 下载适用于 JetPack 4.6 + Python 3.6 的 ONNX Runtime GPU 1.10.0
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl

# 安装
pip3 install onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl
```
*如果下载链接失效，请搜索 "Jetson Zoo ONNX Runtime" 寻找替代链接。*

### 1.4 验证安装
```bash
python3 -c "import onnxruntime; print(onnxruntime.get_device())"
# 输出应包含 'GPU' 字样
```

---

## 2. 部署代码 (`inference.py`)

将以下代码保存为 `inference.py`，并将您的 `policy.onnx` 放在同一目录下。

```python
import time
import numpy as np
import onnxruntime as ort
import serial
import struct

class RobotPolicy:
    def __init__(self, model_path, use_cuda=True):
        # 初始化 ONNX Session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to CPU...")
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"Model loaded. Input: {self.input_name} {self.input_shape}")

    def predict(self, obs):
        """
        执行推理
        :param obs: numpy array, shape (44,)
        :return: actions, numpy array, shape (8,)
        """
        # 增加 Batch 维度 (44,) -> (1, 44)
        obs_batch = obs.astype(np.float32).reshape(1, -1)
        
        # 推理
        result = self.session.run([self.output_name], {self.input_name: obs_batch})
        
        # 移除 Batch 维度
        return result[0].squeeze()

class SerialInterface:
    def __init__(self, port='/dev/ttyUSB0', baud=115200):
        try:
            self.ser = serial.Serial(port, baud, timeout=0.01)
            print(f"Serial opened on {port}")
        except Exception as e:
            print(f"Failed to open serial: {e}")
            self.ser = None

    def read_obs(self):
        """
        [需要用户实现] 从下位机读取传感器数据并转换为观测向量
        返回: np.array shape (44,)
        """
        # 这里只是伪代码，请根据您的通信协议修改
        if not self.ser: return np.zeros(44)
        
        # 示例: 假设下位机发送 44 个 float
        # data = self.ser.read(...)
        # obs = struct.unpack('44f', data)
        
        # 这是一个 Dummy Observation 用于测试
        obs = np.zeros(44, dtype=np.float32)
        
        # 重要：模拟重力向量 (通常 IMU 是最关键的)
        # obs[6:9] = np.array([0, 0, -1]) 
        
        return obs

    def send_action(self, action):
        """
        [需要用户实现] 将动作发送给下位机
        action: np.array shape (8,) - 关节目标位置 (弧度)
        """
        if not self.ser: return
        
        # 示例: 简单的包头 + 8个float + 校验
        # payload = struct.pack('8f', *action)
        # self.ser.write(b'\xAA\x55' + payload)
        pass

def main():
    # 1. 加载模型
    policy = RobotPolicy("policy.onnx")
    
    # 2. 初始化串口
    robot_io = SerialInterface()
    
    # 3. 控制循环配置
    control_freq = 20.0 # Hz
    dt = 1.0 / control_freq
    
    print("Starting control loop...")
    
    try:
        while True:
            loop_start = time.time()
            
            # --- Step A: 获取观测 ---
            obs = robot_io.read_obs()
            
            # --- Step B: 神经网络推理 ---
            # 这里的 obs 已经是原始物理数值，无需手动 Normalize，因为 ONNX 模型内部包含了 Normalization 层
            action = policy.predict(obs)
            
            # --- Step C: 发送指令 ---
            # 注意: action 通常是归一化的或相对值，可能需要根据 Isaac Lab 的 Action Scaling 进行缩放
            # 如果 Action Scaling 已经在 Sim 中处理，这里得到的通常是 "关节目标位置"
            robot_io.send_action(action)
            
            # --- Step D: 维持频率 ---
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"Warning: Loop overrun! Took {elapsed*1000:.1f}ms")
                
    except KeyboardInterrupt:
        print("Stopping...")

if __name__ == "__main__":
    main()
```

---

## 3. 关键注意事项

### 3.1 Opset Version 问题
Isaac Lab 默认导出 **Opset 18**。
*   如果 ONNX Runtime 1.10 报错 `Invalid Opset Version`，说明版本太老不支持 Opset 18。
*   **解决方案**:
    1.  **升级 ONNX Runtime**: 尝试寻找更新的 whl 包（如 1.14+）。
    2.  **降级 Opset**: 在 Isaac Lab 的 `exporter.py` 中，将 `opset_version = 18` 改为 `11` 或 `12`，然后重新运行 `play.py` 导出。

### 3.2 Action Scaling (动作缩放)
Isaac Lab 的输出通常是经过缩放的。
*   检查您的 Config (`flat_env_linear_fast_cfg.py`) 中的 `actions.joint_pos.scale`。
*   如果 scale 是 `0.25`，那么网络输出 `1.0` 代表物理上的 `0.25 rad`。
*   **实机代码修正**:
    ```python
    real_action = network_output * action_scale + default_joint_pos
    ```
    您必须在 `send_action` 函数中手动加上这个计算，除非导出的模型已经包含了这部分（通常不包含）。

### 3.3 Observation Normalization (观测归一化)
*   好消息：Isaac Lab 导出的 ONNX **包含**了 Normalization 层。
*   您只需要给网络输入**真实的物理量**（如速度 m/s，角度 rad，角速度 rad/s）。
*   **不要**在 Python 代码里再做 `(obs - mean) / std`。

### 3.4 串口权限
如果运行脚本报错 `Permission denied: '/dev/ttyUSB0'`：
```bash
sudo usermod -a -G dialout $USER
# 然后重启或重新登录
```
或者临时解决：
```bash
sudo chmod 777 /dev/ttyUSB0
```

---

## 4. 故障排查

*   **错误**: `ImportError: libcublas.so.xx: cannot open shared object file`
    *   **解决**: 确保您的 `LD_LIBRARY_PATH` 包含了 CUDA 库路径。
    *   `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64`

*   **错误**: `onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL : Load model...`
    *   **解决**: 通常是 Opset 版本不兼容。请回退到 Isaac Lab 修改导出 Opset 为 11。

*   **延迟过高**:
    *   去掉所有的 `print` 语句。
    *   确保 `Serial` 的 `timeout` 设置得很小（如 0.01）。
