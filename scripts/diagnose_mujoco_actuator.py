#!/usr/bin/env python3
"""
MuJoCo 执行器性能诊断工具
测试关节是否能正确跟踪目标位置
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "WAVEGO_mujoco/scene.xml"

# 站立姿态
STANDING_POSE = np.array([
    0.100, -0.650,  0.600,   # FL: hip, thigh, calf
   -0.100,  0.650, -0.600,   # FR: hip, thigh, calf
   -0.100, -0.650,  0.600,   # RL: hip, thigh, calf
    0.100,  0.650, -0.600    # RR: hip, thigh, calf
])

# 测试目标：在站立姿态基础上叠加正弦扰动
def get_test_target(t, amplitude=0.2, frequency=0.5):
    """生成测试目标位置 (正弦波叠加)"""
    offset = amplitude * np.sin(2 * np.pi * frequency * t)
    target = STANDING_POSE.copy()
    # 只测试前腿的 thigh 关节
    target[1] += offset  # FL_thigh
    target[4] -= offset  # FR_thigh (镜像)
    return target

def main():
    print("=" * 80)
    print("MuJoCo 执行器性能诊断")
    print("=" * 80)
    
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    
    # 打印当前配置
    print("\n当前执行器配置:")
    print(f"  timestep: {model.opt.timestep}")
    print(f"  actuator_gainprm (kp): {model.actuator_gainprm[0]}")
    print(f"  actuator_biasprm (kv): {model.actuator_biasprm[0]}")
    print(f"  joint_damping: {model.dof_damping[6]}")  # 第一个非root关节
    print(f"  joint_frictionloss: {model.dof_frictionloss[6]}")
    print(f"  forcerange: {model.actuator_forcerange[0]}")
    
    # 初始化到站立姿态
    data.qpos[2] = 0.25
    data.qpos[7:] = STANDING_POSE.copy()
    mujoco.mj_forward(model, data)
    
    print("\n开始测试...")
    print("按 ESC 退出\n")
    
    count = 0
    start_time = time.time()
    
    # 统计变量
    errors_log = []
    torques_log = []
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and count < 1000:
            t = time.time() - start_time
            
            # 设置目标位置
            target = get_test_target(t)
            data.ctrl[:] = target
            
            # 仿真
            mujoco.mj_step(model, data)
            viewer.sync()
            
            count += 1
            
            if count % 20 == 0:  # 每 0.1s 打印一次
                # 计算误差
                joint_error = data.qpos[7:] - data.ctrl[:]
                rms_error = np.sqrt(np.mean(joint_error**2))
                max_error = np.max(np.abs(joint_error))
                
                # 获取力矩
                qfrc_actuator = data.qfrc_actuator[:12]
                max_torque = np.max(np.abs(qfrc_actuator))
                
                # 记录
                errors_log.append(rms_error)
                torques_log.append(max_torque)
                
                # 检查饱和
                saturated = np.sum(np.abs(qfrc_actuator) > 1.90)  # 接近 1.96 Nm
                
                print(f"[t={t:5.1f}s] RMS误差: {rms_error:.4f} rad | "
                      f"最大误差: {max_error:.4f} rad | "
                      f"最大力矩: {max_torque:.2f} Nm | "
                      f"饱和数: {saturated}/12")
                
                # 详细输出（每 1s）
                if count % 200 == 0:
                    print(f"\n{'─'*80}")
                    print(f"关节级诊断 (t={t:.1f}s):")
                    joint_names = ["FL_hip", "FL_thigh", "FL_calf", 
                                   "FR_hip", "FR_thigh", "FR_calf",
                                   "RL_hip", "RL_thigh", "RL_calf",
                                   "RR_hip", "RR_thigh", "RR_calf"]
                    for i, name in enumerate(joint_names):
                        err = joint_error[i]
                        torque = qfrc_actuator[i]
                        vel = data.qvel[6+i]
                        status = "⚠️" if abs(torque) > 1.90 else "✓"
                        print(f"  {name:12s}: 误差={err:+.4f} rad, "
                              f"力矩={torque:+.3f} Nm, 速度={vel:+.2f} rad/s {status}")
                    print(f"{'─'*80}\n")
            
            time.sleep(0.005)  # 匹配 timestep
    
    # 最终统计
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    if errors_log:
        print(f"\n跟踪性能统计:")
        print(f"  平均 RMS 误差: {np.mean(errors_log):.4f} rad ({np.mean(errors_log)*180/np.pi:.2f}°)")
        print(f"  最大 RMS 误差: {np.max(errors_log):.4f} rad ({np.max(errors_log)*180/np.pi:.2f}°)")
        print(f"  平均最大力矩: {np.mean(torques_log):.3f} Nm")
        print(f"  峰值力矩: {np.max(torques_log):.3f} / 1.96 Nm")
        
        # 判断
        avg_error = np.mean(errors_log)
        if avg_error < 0.01:
            print("\n✅ 执行器性能: 优秀 (误差 < 0.01 rad)")
        elif avg_error < 0.05:
            print("\n✅ 执行器性能: 良好 (误差 < 0.05 rad)")
        elif avg_error < 0.1:
            print("\n⚠️  执行器性能: 一般 (误差 < 0.1 rad) - 建议调整 kp/kv")
        else:
            print("\n❌ 执行器性能: 较差 (误差 > 0.1 rad) - 需要优化参数")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
