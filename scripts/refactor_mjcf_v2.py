import re

with open('WAVEGO_mujoco/wavego.xml', 'r') as f:
    content = f.read()

# 1. 移除之前 sed 可能造成的乱序标签
content = re.sub(r'<body name="base_link".*?>', '', content)
content = re.sub(r'<freejoint name="root"/>', '', content)
content = content.replace('</body>', '', content.count('</body>') - 12) # 仅保留末尾的一个闭合

# 2. 重新寻找 worldbody 位置并插入标准的 floating base 结构
worldbody_start = content.find('<worldbody>') + len('<worldbody>')
# 注入 base_link body, freejoint 和根惯量
header = """
    <body name="base_link" pos="0 0 0.25">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="0.75" diaginertia="0.001 0.001 0.001"/>"""

content = content[:worldbody_start] + header + content[worldbody_start:]

# 3. 为所有关节添加 damping="0.5" 以消除震荡，并设置 ref 以对齐初始姿态
# 我们希望 0 弧度就是支撑姿态，所以设置 ref
content = content.replace('joint name="FL_hip_joint"', 'joint name="FL_hip_joint" damping="0.5" stiffness="0" arm_ref="0.0"')
content = content.replace('joint name="FL_thigh_joint"', 'joint name="FL_thigh_joint" damping="0.5" stiffness="0" ref="0.65"')
content = content.replace('joint name="FL_calf_joint"', 'joint name="FL_calf_joint" damping="0.5" stiffness="0" ref="-0.6"')

content = content.replace('joint name="FR_hip_joint"', 'joint name="FR_hip_joint" damping="0.5" stiffness="0" arm_ref="0.0"')
content = content.replace('joint name="FR_thigh_joint"', 'joint name="FR_thigh_joint" damping="0.5" stiffness="0" ref="0.65"')
content = content.replace('joint name="FR_calf_joint"', 'joint name="FR_calf_joint" damping="0.5" stiffness="0" ref="-0.6"')

content = content.replace('joint name="RL_hip_joint"', 'joint name="RL_hip_joint" damping="0.5" stiffness="0" arm_ref="0.0"')
content = content.replace('joint name="RL_thigh_joint"', 'joint name="RL_thigh_joint" damping="0.5" stiffness="0" ref="0.65"')
content = content.replace('joint name="RL_calf_joint"', 'joint name="RL_calf_joint" damping="0.5" stiffness="0" ref="-0.6"')

content = content.replace('joint name="RR_hip_joint"', 'joint name="RR_hip_joint" damping="0.5" stiffness="0" arm_ref="0.0"')
content = content.replace('joint name="RR_thigh_joint"', 'joint name="RR_thigh_joint" damping="0.5" stiffness="0" ref="0.65"')
content = content.replace('joint name="RR_calf_joint"', 'joint name="RR_calf_joint" damping="0.5" stiffness="0" ref="-0.6"')

# 4. 在文件末尾（最后一个 body 之后）闭合根 body
last_body_idx = content.rfind('</body>')
content = content[:last_body_idx+7] + "
    </body>" + content[last_body_idx+7:]

with open('WAVEGO_mujoco/wavego.xml', 'w') as f:
    f.write(content)
print("Re-refactored wavego.xml with joint damping and initial pose references.")
