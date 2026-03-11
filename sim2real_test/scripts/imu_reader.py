"""
[DEPRECATED] 旧 IMU 读取模块 (MPU6050 / BNO055 I2C)。

实际硬件：本项目使用 **维特智能 WT61C 六轴 IMU**，
通过 UART 连接 STM32，STM32 通过 USB CDC 回传 Jetson Nano。

Jetson Nano 上无需直接导入此文件。
请改用 STM32Bridge.get_state() 获取 IMU 数据：

    from stm32_bridge import STM32Bridge
    bridge = STM32Bridge(port="/dev/ttyACM0")
    state = bridge.get_state()
    imu_roll   = state["imu_roll"]   # deg
    imu_pitch  = state["imu_pitch"]  # deg
    imu_gyro   = state["imu_gyro"]   # deg/s, shape (3,)
    imu_accel  = state["imu_accel"]  # m/s², shape (3,)

保留此文件仅供参考，不参与任何实际部署流程。
如需将来某种单体 IMU 直接连接 Jetson Nano，再考虑启用此文件。
"""

from __future__ import annotations

import argparse
import time

import numpy as np



class MPU6050Reader:
    """MPU6050 I2C 读取 (需要 smbus2 库)。"""

    def __init__(self, bus: int = 1, address: int = 0x68):
        self.address = address
        self._gyro_scale = 250.0 / 32768.0 * (np.pi / 180.0)  # ±250 dps → rad/s
        self._accel_scale = 2.0 / 32768.0 * 9.81               # ±2g → m/s²

        # --- 取消注释以启用实际硬件 ---
        # import smbus2
        # self.bus = smbus2.SMBus(bus)
        # # 唤醒 MPU6050
        # self.bus.write_byte_data(self.address, 0x6B, 0x00)
        # # 设置陀螺仪量程 ±250 dps
        # self.bus.write_byte_data(self.address, 0x1B, 0x00)
        # # 设置加速度计量程 ±2g
        # self.bus.write_byte_data(self.address, 0x1C, 0x00)
        # # 设置低通滤波 ~44Hz
        # self.bus.write_byte_data(self.address, 0x1A, 0x03)

        print(f"[MPU6050] Init bus={bus}, addr=0x{address:02X}")

    def _read_raw_word(self, reg: int) -> int:
        """读取 16-bit 有符号寄存器值。"""
        # high = self.bus.read_byte_data(self.address, reg)
        # low = self.bus.read_byte_data(self.address, reg + 1)
        # val = (high << 8) | low
        # return val - 65536 if val >= 32768 else val
        return 0  # 占位

    def read_gyroscope(self) -> np.ndarray:
        """返回 [wx, wy, wz] rad/s (传感器坐标系)。"""
        gx = self._read_raw_word(0x43) * self._gyro_scale
        gy = self._read_raw_word(0x45) * self._gyro_scale
        gz = self._read_raw_word(0x47) * self._gyro_scale
        return np.array([gx, gy, gz], dtype=np.float32)

    def read_accelerometer(self) -> np.ndarray:
        """返回 [ax, ay, az] m/s² (传感器坐标系)。"""
        ax = self._read_raw_word(0x3B) * self._accel_scale
        ay = self._read_raw_word(0x3D) * self._accel_scale
        az = self._read_raw_word(0x3F) * self._accel_scale
        return np.array([ax, ay, az], dtype=np.float32)


class BNO055Reader:
    """BNO055 I2C 读取 (需要 adafruit-circuitpython-bno055 库)。"""

    def __init__(self, bus: int = 1, address: int = 0x28):
        # --- 取消注释以启用实际硬件 ---
        # import board
        # import busio
        # import adafruit_bno055
        # i2c = busio.I2C(board.SCL, board.SDA)
        # self.sensor = adafruit_bno055.BNO055_I2C(i2c, address=address)
        print(f"[BNO055] Init bus={bus}, addr=0x{address:02X}")

    def read_gyroscope(self) -> np.ndarray:
        """返回 [wx, wy, wz] rad/s。"""
        # gyro = self.sensor.gyro  # 返回 (x, y, z) rad/s
        # return np.array(gyro, dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    def read_quaternion(self) -> np.ndarray:
        """返回四元数 [w, x, y, z]。"""
        # quat = self.sensor.quaternion  # (w, x, y, z)
        # if quat[0] is None:
        #     return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # return np.array(quat, dtype=np.float32)
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def read_accelerometer(self) -> np.ndarray:
        """返回 [ax, ay, az] m/s²。"""
        # accel = self.sensor.linear_acceleration
        # return np.array(accel, dtype=np.float32)
        return np.array([0.0, 0.0, -9.81], dtype=np.float32)


# ==============================================================================
# 坐标系变换（根据实际 IMU 安装方向修改）
# ==============================================================================

def imu_to_body_transform(
    gyro_imu: np.ndarray,
    accel_imu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    将 IMU 传感器坐标系数据转换为机器人机体坐标系 (X-前, Y-左, Z-上)。

    *** 这个函数必须根据 IMU 的实际安装朝向来修改！***

    常见变换:
    - IMU X 朝前, Y 朝左, Z 朝上 (与机体系一致): 无需变换
    - IMU X 朝右, Y 朝前: gyro_body = [-gy, gx, gz]
    - IMU 倒装 (Z 朝下): gyro_body = [gx, -gy, -gz]
    """
    # 默认：假设 IMU 安装方向与机体坐标系一致
    gyro_body = gyro_imu.copy()
    accel_body = accel_imu.copy()
    return gyro_body, accel_body


# ==============================================================================
# 从加速度计估算 projected_gravity
# ==============================================================================

def accel_to_projected_gravity(accel_body: np.ndarray) -> np.ndarray:
    """
    从加速度计读数估算 projected_gravity。

    静态条件下：accel ≈ -g_body (加速度计测量的是反作用力)
    因此 projected_gravity ≈ -accel / |accel|

    注意：运动中加速度叠加，精度降低。
    """
    norm = np.linalg.norm(accel_body)
    if norm < 0.1:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return -accel_body / norm


# ==============================================================================
# 测试
# ==============================================================================

def test_imu(imu_type: str = "mpu6050", duration: float = 5.0):
    """自检模式：打印 IMU 数据。"""
    if imu_type == "mpu6050":
        imu = MPU6050Reader()
    else:
        imu = BNO055Reader()

    print(f"\n=== IMU Test ({imu_type}) for {duration}s ===")
    print("Format: gyro_body(3) | proj_gravity(3) | freq")

    t_start = time.perf_counter()
    count = 0

    while time.perf_counter() - t_start < duration:
        t0 = time.perf_counter()

        gyro_imu = imu.read_gyroscope()
        accel_imu = imu.read_accelerometer()

        gyro_body, accel_body = imu_to_body_transform(gyro_imu, accel_imu)
        proj_grav = accel_to_projected_gravity(accel_body)

        count += 1
        elapsed = time.perf_counter() - t_start

        if count % 100 == 0:
            hz = count / elapsed
            print(
                f"[{elapsed:6.2f}s] "
                f"gyro=[{gyro_body[0]:+6.3f}, {gyro_body[1]:+6.3f}, {gyro_body[2]:+6.3f}] "
                f"grav=[{proj_grav[0]:+5.2f}, {proj_grav[1]:+5.2f}, {proj_grav[2]:+5.2f}] "
                f"hz={hz:.0f}"
            )

        # 限制读取速率
        dt = time.perf_counter() - t0
        if dt < 0.005:  # 最高 200 Hz
            time.sleep(0.005 - dt)

    total_hz = count / (time.perf_counter() - t_start)
    print(f"\nDone: {count} samples, avg {total_hz:.0f} Hz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--imu-type", choices=["mpu6050", "bno055"], default="mpu6050")
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()

    if args.test:
        test_imu(args.imu_type, args.duration)
