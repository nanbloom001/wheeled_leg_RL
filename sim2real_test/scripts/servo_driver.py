"""
[DEPRECATED] 旧 STS3215 舵机驱动占位实现。

实际硬件：本项目使用 **STM32F407** 作为透传网关，
通过 USB CDC (/dev/ttyACM0) 与 Jetson Nano 通信。

Jetson Nano 上无需直接嫹 STS3215 总线，也无需此文件。
请改用 STM32Bridge.send_servo_targets() 发送指令：

    from stm32_bridge import STM32Bridge
    bridge = STM32Bridge(port="/dev/ttyACM0")
    bridge.send_servo_targets(target_raw_0_4095, speed=0, time_ms=0)

协议细节以当前固件实测为准（不同固件分支可能存在字段顺序/校验差异）。
详见当前分支固件源码与 stm32_bridge.py 兼容实现。
保留此文件仅供参考。
"""

from __future__ import annotations

import argparse
import struct
import time

import numpy as np

# 12 个关节名称（servo/MuJoCo DFS 顺序）
SERVO_JOINT_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]

# 默认站立位（servo 顺序, rad）
DEFAULT_POS_SERVO = np.array([
    0.1, -0.65, 0.6,     # FL
    -0.1, 0.65, -0.6,    # FR
    -0.1, -0.65, 0.6,    # RL
    0.1, 0.65, -0.6,     # RR
], dtype=np.float32)

# 关节限位（servo 顺序, rad）
JOINT_LIMITS_LOW_SERVO = np.array([
    -0.5236, -2.0944, -2.0944,
    -0.5236, -2.0944, -2.0944,
    -0.5236, -2.0944, -2.0944,
    -0.5236, -2.0944, -2.0944,
], dtype=np.float32)

JOINT_LIMITS_HIGH_SERVO = np.array([
    0.5236, 2.0944, 2.0944,
    0.5236, 2.0944, 2.0944,
    0.5236, 2.0944, 2.0944,
    0.5236, 2.0944, 2.0944,
], dtype=np.float32)


class ST3215Driver:
    """
    ST3215 串行总线舵机驱动。

    [DEPRECATED] 此类已废弃，请改用 STM32Bridge。

    通信架构:
        Jetson Nano → USB CDC (/dev/ttyACM0) → STM32F407 → 串行总线 → 12× STS3215

    实际协议包头: 0xA5 0x5A (不是 0xAA 0x55)
        发送: [0xA5, 0x5A, LEN, CMD, DATA..., XOR_CHECKSUM]
        接收: [0xA5, 0x5A, LEN, CMD, DATA..., XOR_CHECKSUM]
    """

    # 命令码（根据实际下位机协议定义）
    CMD_SET_POSITIONS = 0x01
    CMD_GET_POSITIONS = 0x02
    CMD_GET_VELOCITIES = 0x03
    CMD_PING = 0x10

    def __init__(self, port: str = "/dev/ttyACM0", baud: int = 115200, timeout: float = 0.01):
        self._port = port
        self._baud = baud
        self._ser = None

        try:
            import serial
            self._ser = serial.Serial(port, baud, timeout=timeout)
            print(f"[ST3215] Serial opened: {port} @ {baud}")
        except ImportError:
            print("[ST3215] pyserial not installed, running in dummy mode")
        except Exception as e:
            print(f"[ST3215] Failed to open serial: {e}, running in dummy mode")

    def _checksum(self, data: bytes) -> int:
        return sum(data) & 0xFF

    def _send_packet(self, cmd: int, payload: bytes) -> bool:
        """发送一个数据包。"""
        if self._ser is None:
            return False
        # [DEPRECATED] 旧包头 0xAA 0x55 已废弃；实际固件使用 0xA5 0x5A
        # 请使用 stm32_bridge.STM32Bridge.send_servo_targets() 代替
        header = bytes([0xA5, 0x5A, len(payload), cmd])  # 修正为实际包头
        checksum = bytes([self._checksum(header + payload)])
        self._ser.write(header + payload + checksum)
        return True

    def _recv_packet(self, expected_len: int, timeout: float = 0.05) -> bytes | None:
        """接收一个数据包。"""
        if self._ser is None:
            return None
        t0 = time.perf_counter()
        buf = b""
        total_len = 4 + expected_len + 1  # header(4) + payload + checksum(1)
        while time.perf_counter() - t0 < timeout:
            chunk = self._ser.read(total_len - len(buf))
            if chunk:
                buf += chunk
            if len(buf) >= total_len:
                break
        if len(buf) < total_len:
            return None
        # 验证帧头
        if buf[0] != 0xA5 or buf[1] != 0x5A:  # 实际固件包头
            return None
        payload = buf[4:4 + expected_len]
        return payload

    def ping(self) -> bool:
        """测试下位机是否响应。"""
        self._send_packet(self.CMD_PING, b"")
        resp = self._recv_packet(1)
        if resp is not None:
            print("[ST3215] Ping OK")
            return True
        else:
            print("[ST3215] Ping failed (no response)")
            return False

    def send_positions(self, positions: np.ndarray) -> None:
        """
        发送 12 个关节目标位置 (servo 顺序, 单位 rad)。

        Args:
            positions: shape (12,), float32, 单位 rad
        """
        assert positions.shape == (12,), f"Expected (12,), got {positions.shape}"

        # 安全限幅
        positions = np.clip(positions, JOINT_LIMITS_LOW_SERVO, JOINT_LIMITS_HIGH_SERVO)

        # 打包为字节
        payload = struct.pack("12f", *positions.astype(np.float32))
        self._send_packet(self.CMD_SET_POSITIONS, payload)

    def read_positions(self) -> np.ndarray:
        """
        读取 12 个关节当前位置 (servo 顺序, 单位 rad)。

        Returns:
            shape (12,), float32
        """
        self._send_packet(self.CMD_GET_POSITIONS, b"")
        resp = self._recv_packet(12 * 4)  # 12 个 float32
        if resp is not None and len(resp) == 48:
            return np.array(struct.unpack("12f", resp), dtype=np.float32)
        # 失败时返回默认位
        return DEFAULT_POS_SERVO.copy()

    def read_velocities(self) -> np.ndarray:
        """读取 12 个关节当前速度 (servo 顺序, 单位 rad/s)。"""
        self._send_packet(self.CMD_GET_VELOCITIES, b"")
        resp = self._recv_packet(12 * 4)
        if resp is not None and len(resp) == 48:
            return np.array(struct.unpack("12f", resp), dtype=np.float32)
        return np.zeros(12, dtype=np.float32)

    def go_to_default(self, speed: float = 0.5) -> None:
        """缓慢移动到默认站立位。"""
        print("[ST3215] Moving to default position...")
        current = self.read_positions()
        steps = 50
        for i in range(steps + 1):
            alpha = i / steps
            target = (1 - alpha) * current + alpha * DEFAULT_POS_SERVO
            self.send_positions(target)
            time.sleep(0.02)
        print("[ST3215] At default position.")

    def close(self):
        if self._ser is not None:
            self._ser.close()
            print("[ST3215] Serial closed.")


def test_single_joint(driver: ST3215Driver, joint_name: str, offset: float):
    """测试单个关节: 从默认位偏移 offset rad，等 2 秒，回到默认位。"""
    idx = SERVO_JOINT_NAMES.index(joint_name)
    print(f"\nTesting {joint_name} (servo index {idx}): offset={offset:.3f} rad ({np.degrees(offset):.1f}°)")

    target = DEFAULT_POS_SERVO.copy()
    target[idx] += offset
    target = np.clip(target, JOINT_LIMITS_LOW_SERVO, JOINT_LIMITS_HIGH_SERVO)

    print("  Sending offset position...")
    driver.send_positions(target)
    time.sleep(2.0)

    print("  Returning to default...")
    driver.send_positions(DEFAULT_POS_SERVO.copy())
    time.sleep(1.0)
    print("  Done.")


def test_read_positions(driver: ST3215Driver, duration: float = 10.0):
    """持续读取并打印关节位置。"""
    print(f"\nReading positions for {duration}s (move joints by hand)...")
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < duration:
        pos = driver.read_positions()
        line = "  ".join(f"{SERVO_JOINT_NAMES[i][:6]}={pos[i]:+6.3f}" for i in range(12))
        print(f"[{time.perf_counter()-t_start:5.1f}s] {line}")
        time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--test", action="store_true", help="Ping test")
    parser.add_argument("--single-joint", type=str, default=None, help="Test single joint by name")
    parser.add_argument("--offset", type=float, default=0.1, help="Offset for single-joint test (rad)")
    parser.add_argument("--read-positions", action="store_true", help="Read and print positions")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--go-default", action="store_true", help="Move to default position")
    args = parser.parse_args()

    driver = ST3215Driver(port=args.port, baud=args.baud)

    try:
        if args.test:
            driver.ping()
        if args.go_default:
            driver.go_to_default()
        if args.single_joint:
            test_single_joint(driver, args.single_joint, args.offset)
        if args.read_positions:
            test_read_positions(driver, args.duration)
    finally:
        driver.close()
