
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

def parse_origin(joint):
    origin = joint.find('origin')
    xyz = np.array([float(x) for x in origin.get('xyz').split()])
    rpy = np.array([float(x) for x in origin.get('rpy').split()])
    return xyz, rpy

def get_joint_data(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = {}
    for j in root.findall('joint'):
        data[j.get('name')] = parse_origin(j)
    return data

def main():
    old_file = "/home/user/IsaacLab/qwe_old.SLDASM/urdf/qwe.SLDASM.urdf"
    curr_file = "/home/user/IsaacLab/qwe.SLDASM/urdf/qwe.SLDASM.urdf"
    
    old_data = get_joint_data(old_file)
    curr_data = get_joint_data(curr_file)
    
    legs = ["qian_zuo", "qian_you", "hou_zuo", "hou_you"]
    
    print(f"{'Joint':<20} | {'Old XYZ':<25} | {'Old RPY':<25} | {'Curr XYZ':<25} | {'Curr RPY':<25}")
    print("-" * 130)
    
    for leg in legs:
        small = f"small_{leg}_joint"
        # 注意：last 关节名字可能不规律，这里尝试匹配
        last = None
        for name in old_data.keys():
            if f"last_{leg}" in name:
                last = name
                break
        
        if not last: continue

        # Small
        s_old_xyz, s_old_rpy = old_data[small]
        s_curr_xyz, s_curr_rpy = curr_data[small]
        
        print(f"{small:<20} | {str(s_old_xyz):<25} | {str(s_old_rpy):<25} | {str(s_curr_xyz):<25} | {str(s_curr_rpy):<25}")
        
        # Last
        l_old_xyz, l_old_rpy = old_data[last]
        l_curr_xyz, l_curr_rpy = curr_data[last]
        
        print(f"{last:<20} | {str(l_old_xyz):<25} | {str(l_old_rpy):<25} | {str(l_curr_xyz):<25} | {str(l_curr_rpy):<25}")
        
        # 计算理论上的 Last Position 变化
        # P_last_world = P_small + R_small * P_last_local
        # 我们假设 Parent (Big Link) 没变（其实变了，但我们只关心相对关系）
        # 如果我们希望 Last Link 相对于 Big Link 的位置不变：
        # P_old = P_small_old + R_small_old * P_last_old
        # P_new = P_small_new + R_small_new * P_last_new
        # 我们希望 P_old == P_new
        # 所以 P_last_new = R_small_new.inv() * (P_small_old + R_small_old * P_last_old - P_small_new)
        
        R_old = Rotation.from_euler('xyz', s_old_rpy).as_matrix()
        R_new = Rotation.from_euler('xyz', s_curr_rpy).as_matrix()
        
        target_last_pos = R_new.T @ (s_old_xyz + R_old @ l_old_xyz - s_curr_xyz)
        
        print(f"  -> Calculated Target Last XYZ: {target_last_pos}")
        print(f"  -> Current Last XYZ:           {l_curr_xyz}")
        print(f"  -> Diff:                       {np.linalg.norm(target_last_pos - l_curr_xyz):.4f}")
        print("-" * 130)

if __name__ == "__main__":
    main()
