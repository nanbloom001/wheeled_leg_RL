#!/usr/bin/env python3
"""
简单的 MuJoCo 场景查看器 — 无策略、无诊断，仅可视化场景
"""

import os
from pathlib import Path
import mujoco
from mujoco import viewer

def main():
    # 设置工作目录为脚本所在的父目录（ProjectRoot）
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    # 加载 MuJoCo 场景
    scene_path = "WAVEGO_mujoco/scene.xml"
    
    print(f"当前工作目录: {os.getcwd()}")
    print(f"尝试加载: {scene_path}")
    
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene file not found: {scene_path}")
    
    # 从 XML 加载模型
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    
    print(f"✓ 成功加载模型: {model.nq} dofs, {len(model.body_names)} bodies")
    print("启动交互式查看器...")
    print("  - 鼠标拖动旋转视图")
    print("  - 滚轮缩放")
    print("  - 右键菜单打开选项")
    print("  - 'H' 键查看帮助")
    print("  - ESC 或关闭窗口退出")
    
    # 启动查看器（会一直运行直到用户关闭窗口）
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            # 实时仿真循环（内部自动渲染）
            mujoco.mj_step(model, data)
            
            # 定时回调，可选
            if data.time >= 300:  # 300秒后停止（可去掉）
                break

if __name__ == "__main__":
    main()
