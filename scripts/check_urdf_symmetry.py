
import xml.etree.ElementTree as ET
import sys

def parse_urdf(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    joints = []
    for joint in root.findall('joint'):
        name = joint.get('name')
        if joint.get('type') == 'fixed':
            continue
            
        origin = joint.find('origin')
        rpy = origin.get('rpy') if origin is not None else "0 0 0"
        xyz = origin.get('xyz') if origin is not None else "0 0 0"
        
        axis = joint.find('axis')
        axis_xyz = axis.get('xyz') if axis is not None else "1 0 0"
        
        joints.append({
            "name": name,
            "rpy": rpy,
            "axis": axis_xyz
        })
    
    # 打印表格
    print(f"{'Joint Name':<25} | {'RPY (Origin)':<30} | {'Axis':<15}")
    print("-" * 75)
    
    # 按腿分组打印
    groups = ["qian_zuo", "qian_you", "hou_zuo", "hou_you"]
    for group in groups:
        print(f"--- {group} ---")
        for j in joints:
            if group in j["name"]:
                print(f"{j['name']:<25} | {j['rpy']:<30} | {j['axis']:<15}")

if __name__ == "__main__":
    parse_urdf("/home/user/IsaacLab/qwe.SLDASM/urdf/qwe.SLDASM.urdf")
