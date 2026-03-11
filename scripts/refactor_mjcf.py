import re

with open('WAVEGO_mujoco/wavego.xml', 'r') as f:
    lines = f.readlines()

new_lines = []
in_worldbody = False
processed_base = False

for line in lines:
    if '<worldbody>' in line:
        in_worldbody = True
        new_lines.append(line)
        # 插入 base_link body 和 freejoint
        new_lines.append('    <body name="base_link" pos="0 0 0">
')
        new_lines.append('      <freejoint name="root"/>
')
        new_lines.append('      <inertial pos="0 0 0" mass="0.75" diaginertia="0.001 0.001 0.001"/>
')
        continue
    
    if in_worldbody and '</worldbody>' in line:
        new_lines.append('    </body>
') # 关闭 base_link body
        new_lines.append(line)
        in_worldbody = False
        continue
    
    new_lines.append(line)

with open('WAVEGO_mujoco/wavego.xml', 'w') as f:
    f.writelines(new_lines)

print("Successfully refactored wavego.xml with a floating base_link.")
