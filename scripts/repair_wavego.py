
import os
import re

# File paths
input_file = "WAVEGO_description/WAVEGO_description/urdf/WAVEGO.xacro"
output_file = "WAVEGO_description/WAVEGO_description/urdf/WAVEGO.urdf"

# Joint name mapping based on parent/child links
# Parent -> Child -> New Name
joint_map_by_links = {
    ("base_link", "FL_hip_1"): "FL_hip_joint",
    ("FL_hip_1", "FL_thigh_1"): "FL_thigh_joint",
    ("FL_thigh_1", "FL_calf_1"): "FL_calf_joint",
    
    ("base_link", "FR_hip_1"): "FR_hip_joint",
    ("FR_hip_1", "FR_thigh_1"): "FR_thigh_joint",
    ("FR_thigh_1", "FR_calf_1"): "FR_calf_joint",
    
    ("base_link", "RL_hip_1"): "RL_hip_joint",
    ("RL_hip_1", "RL_thigh_1"): "RL_thigh_joint",
    ("RL_thigh_1", "RL_calf_1"): "RL_calf_joint",
    
    ("base_link", "RR_hip_1"): "RR_hip_joint",
    ("RR_hip_1", "RR_thigh_1"): "RR_thigh_joint",
    ("RR_thigh_1", "RR_calf_1"): "RR_calf_joint",
}

with open(input_file, "rb") as f:
    content = f.read().decode("utf-8", errors="ignore")

# 1. Remove xacro includes and namespaces
content = re.sub(r'<xacro:include.*/>', '', content)
content = content.replace('<robot name="WAVEGO" xmlns:xacro="http://www.ros.org/wiki/xacro">', '<robot name="WAVEGO">')

# 2. Fix STL paths
# package://WAVEGO_description/meshes/ -> ../meshes/
content = content.replace("package://WAVEGO_description/meshes/", "../meshes/")

# 3. Fix joint names
# We need to find each joint block, check its parent/child, and rename it.
def rename_joints(match):
    joint_block = match.group(0)
    parent = re.search(r'<parent link="([^"]+)"/>', joint_block).group(1)
    child = re.search(r'<child link="([^"]+)"/>', joint_block).group(1)
    
    new_name = joint_map_by_links.get((parent, child), f"joint_{parent}_{child}")
    
    # Replace the name attribute in the joint tag
    joint_block = re.sub(r'name="[^"]+"', f'name="{new_name}"', joint_block, count=1)
    return joint_block

content = re.sub(r'<joint.*?>.*?</joint>', rename_joints, content, flags=re.DOTALL)

# 4. Remove any remaining xacro tags (like <xacro:property>)
content = re.sub(r'<xacro:.*?>', '', content)

with open(output_file, "w") as f:
    f.write(content)

print(f"Successfully created {output_file}")
