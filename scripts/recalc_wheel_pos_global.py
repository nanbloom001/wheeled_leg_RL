
import numpy as np
import xml.etree.ElementTree as ET

def get_rpy_mat(rpy_str):
    r, p, y = [float(x) for x in rpy_str.split()]
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def get_xyz(xyz_str):
    return np.array([float(x) for x in xyz_str.split()])

class RobotKinematics:
    def __init__(self, urdf_path):
        self.root = ET.parse(urdf_path).getroot()
        self.joints = {j.get('name'): j for j in self.root.findall('joint')}
        
    def get_transform(self, joint_name):
        j = self.joints[joint_name]
        origin = j.find('origin')
        return get_xyz(origin.get('xyz')), get_rpy_mat(origin.get('rpy'))

def main():
    old_urdf = "/home/user/IsaacLab/qwe_old.SLDASM/urdf/qwe.SLDASM.urdf"
    new_urdf = "/home/user/IsaacLab/qwe.SLDASM/urdf/qwe.SLDASM.urdf"
    
    old_k = RobotKinematics(old_urdf)
    new_k = RobotKinematics(new_urdf)
    
    legs = ["qian_zuo", "qian_you", "hou_zuo", "hou_you"]
    
    print("--- Global Recalculation ---")
    
    for leg in legs:
        # 1. 确定关节名称 (处理 Last 关节命名的不一致)
        # Old URDF names
        o_big = f"big_{leg}_joint"
        o_small = f"small_{leg}_joint"
        o_last = next((n for n in old_k.joints if f"last_{leg}" in n), None)
        
        # New URDF names (same names)
        n_big = o_big
        n_small = o_small
        n_last = o_last
        
        # 2. 计算 Old Last Joint 在 Base Frame 下的位置 (Target)
        # T_base_last = T_base_big * T_big_small * T_small_last
        
        t_b_xyz, t_b_rot = old_k.get_transform(o_big)
        t_s_xyz, t_s_rot = old_k.get_transform(o_small)
        t_l_xyz, _       = old_k.get_transform(o_last)
        
        # P_last_in_base = P_big + R_big * (P_small + R_small * P_last)
        p_target = t_b_xyz + t_b_rot @ (t_s_xyz + t_s_rot @ t_l_xyz)
        
        # 3. 计算 New Small Joint 在 Base Frame 下的姿态
        nt_b_xyz, nt_b_rot = new_k.get_transform(n_big)
        nt_s_xyz, nt_s_rot = new_k.get_transform(n_small)
        
        # New Small Position in Base
        p_small_new_in_base = nt_b_xyz + nt_b_rot @ nt_s_xyz
        
        # New Small Rotation in Base
        # R_base_small = R_base_big * R_big_small
        r_small_new_in_base = nt_b_rot @ nt_s_rot
        
        # 4. 逆推 New Last Joint 应该在 New Small Frame 的哪里
        # P_target = P_small_new_in_base + R_small_new_in_base * P_new_local
        # P_new_local = R_inv * (P_target - P_small_new_in_base)
        
        p_new_local = r_small_new_in_base.T @ (p_target - p_small_new_in_base)
        
        print(f"Leg {leg}:")
        print(f"  Target Global Pos: {p_target}")
        print(f"  New Local XYZ:     {p_new_local[0]:.6f} {p_new_local[1]:.6f} {p_new_local[2]:.6f}")

if __name__ == "__main__":
    main()
