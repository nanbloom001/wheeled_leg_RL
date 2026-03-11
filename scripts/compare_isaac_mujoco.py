#!/usr/bin/env python3
"""
Step 2: 对比 Isaac Lab 和 MuJoCo 的配置差异
基于 Step 1 提取的真实数据进行对比
"""

import json
import numpy as np
import mujoco

def load_ground_truth():
    """加载 Isaac Lab 真实数据"""
    try:
        with open("isaac_lab_ground_truth.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ 错误: 未找到 isaac_lab_ground_truth.json")
        print("   请先运行: python scripts/extract_isaac_lab_data.py --headless")
        exit(1)

def main():
    print("=" * 80)
    print("Isaac Lab ↔ MuJoCo 配置对比")
    print("=" * 80)
    
    # 加载 Isaac Lab 真实数据
    gt = load_ground_truth()
    
    # 加载 MuJoCo 模型
    model = mujoco.MjModel.from_xml_path("WAVEGO_mujoco/scene.xml")
    data = mujoco.MjData(model)
    
    # 获取 MuJoCo 关节名称
    mujoco_joints = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name != "root":  # 跳过 root freejoint
            mujoco_joints.append(name)
    
    print("\n[1] 关节顺序对比:")
    print("=" * 80)
    print(f"{'Index':<8} {'Isaac Lab':<25} {'MuJoCo':<25} {'匹配?':<10}")
    print("-" * 80)
    
    all_match = True
    for i in range(12):
        isaac_name = gt["joint_names"][i]
        mujoco_name = mujoco_joints[i]
        match = isaac_name == mujoco_name
        all_match = all_match and match
        status = "✅" if match else "❌"
        print(f"{i:<8} {isaac_name:<25} {mujoco_name:<25} {status}")
    
    if all_match:
        print("\n✅ 关节顺序完全匹配！")
    else:
        print("\n❌ 关节顺序不匹配！需要重新映射！")
        return
    
    print("\n[2] 站立姿态对比:")
    print("=" * 80)
    isaac_default = np.array(gt["default_joint_pos"])
    mujoco_default = model.key_qpos[0][7:19]  # keyframe "home" 的关节位置
    
    print(f"{'Joint':<20} {'Isaac Lab':<15} {'MuJoCo':<15} {'差异':<15}")
    print("-" * 80)
    
    max_diff = 0
    for i, name in enumerate(gt["joint_names"]):
        diff = abs(isaac_default[i] - mujoco_default[i])
        max_diff = max(max_diff, diff)
        status = "✅" if diff < 0.01 else "⚠️"
        print(f"{name:<20} {isaac_default[i]:+.6f}      {mujoco_default[i]:+.6f}      "
              f"{diff:.6f} {status}")
    
    if max_diff < 0.01:
        print(f"\n✅ 站立姿态匹配良好！(最大差异: {max_diff:.6f} rad)")
    else:
        print(f"\n⚠️ 站立姿态存在差异！(最大差异: {max_diff:.6f} rad)")
    
    print("\n[3] 执行器参数对比:")
    print("=" * 80)
    print("Isaac Lab 配置:")
    print("  stiffness (kp): 400.0")
    print("  damping (总): 10.0")
    print("  action_scale: 0.25")
    print("  effort_limit: 1.96 Nm")
    print()
    print("MuJoCo 当前配置:")
    print(f"  kp (gainprm[0]): {model.actuator_gainprm[0, 0]:.1f}")
    print(f"  joint damping: {model.dof_damping[6]:.2f}")
    print(f"  frictionloss: {model.dof_frictionloss[6]:.2f}")
    print(f"  forcerange: [{model.actuator_forcerange[0, 0]:.2f}, {model.actuator_forcerange[0, 1]:.2f}] Nm")
    print(f"  timestep: {model.opt.timestep}")
    
    # 关键参数检查
    issues = []
    if abs(model.actuator_gainprm[0, 0] - 400) > 50:
        issues.append("⚠️ kp 与 Isaac Lab 差异较大")
    if abs(model.actuator_forcerange[0, 1] - 1.96) > 0.01:
        issues.append("⚠️ forcerange 与 Isaac Lab 不一致")
    if abs(model.opt.timestep - 0.005) > 0.001:
        issues.append("⚠️ timestep 与 Isaac Lab 不一致")
    
    if issues:
        print("\n配置问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ 执行器参数配置合理")
    
    print("\n[4] 观测空间验证:")
    print("=" * 80)
    obs_mean = np.array(gt["obs_mean"])
    obs_std = np.array(gt["obs_std"])
    
    print(f"观测维度: {len(obs_mean)}")
    print(f"归一化均值范围: [{obs_mean.min():.3f}, {obs_mean.max():.3f}]")
    print(f"归一化标准差范围: [{obs_std.min():.3f}, {obs_std.max():.3f}]")
    
    # 检查是否有异常值
    if np.any(obs_std < 0.01):
        print("⚠️ 警告: 某些观测维度的标准差接近 0，可能导致数值问题")
        low_std_idx = np.where(obs_std < 0.01)[0]
        print(f"   低标准差维度: {low_std_idx}")
    
    print("\n[5] 动作空间验证:")
    print("=" * 80)
    print(f"动作维度: 12")
    print(f"动作缩放: {gt['action_scale']}")
    print(f"策略原始输出范围: {gt['action_range']}")
    print(f"目标位置范围: {gt['target_range']}")
    
    # 检查动作范围是否合理
    action_min, action_max = gt['action_range']
    if abs(action_min) > 10 or abs(action_max) > 10:
        print("⚠️ 警告: 策略输出范围异常大，可能导致不稳定")
    else:
        print("✅ 策略输出范围正常")
    
    print("\n" + "=" * 80)
    print("对比完成！")
    print("=" * 80)
    
    # 生成诊断报告
    print("\n📊 诊断总结:")
    if all_match and max_diff < 0.01 and not issues:
        print("  ✅ 所有配置匹配良好，可以进行 sim2sim 测试")
    else:
        print("  ⚠️ 发现配置差异，建议先修复后再进行 sim2sim")

if __name__ == "__main__":
    main()
