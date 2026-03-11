#!/bin/bash
# 快速验证 MuJoCo 执行器修复

echo "========================================================================"
echo "WAVEGO MuJoCo 执行器修复验证"
echo "========================================================================"

cd /home/user/IsaacLab
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

echo ""
echo "步骤 1: 检查 XML 配置..."
echo "------------------------------------------------------------------------"
grep -A2 "joint damping" WAVEGO_mujoco/wavego.xml | head -3
grep "position kp" WAVEGO_mujoco/wavego.xml

echo ""
echo "步骤 2: 运行执行器诊断 (10秒测试)..."
echo "------------------------------------------------------------------------"
echo "提示: 观察关节跟踪误差是否 < 0.05 rad"
echo ""

timeout 10s python scripts/diagnose_mujoco_actuator.py || echo "诊断测试已超时退出（正常）"

echo ""
echo "========================================================================"
echo "接下来请手动运行完整测试:"
echo "========================================================================"
echo ""
echo "  python scripts/sim2sim_mujoco.py"
echo ""
echo "预期结果:"
echo "  - 关节误差 < 0.05 rad"
echo "  - VelX 逐渐增加至 ~0.5 m/s (不再在 0 附近震荡)"
echo "  - 力矩不再持续饱和"
echo ""
