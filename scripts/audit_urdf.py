
import xml.etree.ElementTree as ET
import sys

def parse_vec(s):
    return [float(x) for x in s.split()]

def compare(name_a, name_b, vec_a, vec_b, label):
    # 简单的对比打印
    print(f"{label:<15} | {name_a[-8:]:<10}: {vec_a} | {name_b[-8:]:<10}: {vec_b}")

def audit_urdf(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    joints = {j.get('name'): j for j in root.findall('joint')}
    links = {l.get('name'): l for l in root.findall('link')}
    
    pairs = [
        ("big_qian_zuo", "big_qian_you"),
        ("small_qian_zuo", "small_qian_you"),
        ("last_qian_zuo", "last_qian_you"),
        # ("big_hou_zuo", "big_hou_you"), # 只要看一组就能发现规律
    ]
    
    print("\n=== URDF Symmetry Audit ===")
    
    for left, right in pairs:
        print(f"\n--- Comparing {left} vs {right} ---")
        
        # 1. Joint Origin
        j_l = joints[left + "_joint"]
        j_r = joints[right + "_joint"]
        
        xyz_l = parse_vec(j_l.find('origin').get('xyz'))
        xyz_r = parse_vec(j_r.find('origin').get('xyz'))
        rpy_l = parse_vec(j_l.find('origin').get('rpy'))
        rpy_r = parse_vec(j_r.find('origin').get('rpy'))
        
        compare(left, right, xyz_l, xyz_r, "Joint XYZ")
        compare(left, right, rpy_l, rpy_r, "Joint RPY")
        
        # 2. Link Inertial Origin
        l_l = links[left]
        l_r = links[right]
        
        in_xyz_l = parse_vec(l_l.find('inertial').find('origin').get('xyz'))
        in_xyz_r = parse_vec(l_r.find('inertial').find('origin').get('xyz'))
        
        compare(left, right, in_xyz_l, in_xyz_r, "Inertial XYZ")
        
        # 3. Visual Origin
        v_xyz_l = parse_vec(l_l.find('visual').find('origin').get('xyz'))
        v_xyz_r = parse_vec(l_r.find('visual').find('origin').get('xyz'))
        
        compare(left, right, v_xyz_l, v_xyz_r, "Visual XYZ")

if __name__ == "__main__":
    audit_urdf("/home/user/IsaacLab/qwe.SLDASM/urdf/qwe.SLDASM.urdf")
