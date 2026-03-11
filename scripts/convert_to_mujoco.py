import mujoco
import os
import re

# 定义路径
urdf_path = "WAVEGO_mujoco/wavego_for_mujoco.urdf"
mjcf_path = "WAVEGO_mujoco/wavego.xml"

print(f"Loading and pre-processing URDF: {urdf_path}")

# 预处理：修正 URDF 中的网格路径，确保 MuJoCo 能找到 meshes 文件夹
with open(urdf_path, 'r') as f:
    content = f.read()

# 将任何路径格式统一替换为 meshes/filename.stl
content = re.sub(r'filename="[^"]*/([^/]+\.[sS][tT][lL])"', r'filename="meshes/\1"', content)

with open(urdf_path, 'w') as f:
    f.write(content)

try:
    # 加载 URDF
    # MuJoCo 的 from_xml_path 会自动处理相关的资产路径
    model = mujoco.MjModel.from_xml_path(urdf_path)
    
    # 保存为 MJCF XML
    mujoco.mj_saveLastXML(mjcf_path, model)
    print(f"Successfully converted to MJCF: {mjcf_path}")
    
    # 验证是否可以生成数据
    data = mujoco.MjData(model)
    print("Verification successful: Model and Data structures created.")
except Exception as e:
    print(f"Error during MuJoCo conversion: {e}")
