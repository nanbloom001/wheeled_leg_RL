
import xml.etree.ElementTree as ET
import sys

def list_joints(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    print("All joints found in URDF:")
    for j in root.findall('joint'):
        print(j.get('name'))

if __name__ == "__main__":
    list_joints("/home/user/IsaacLab/qwe.SLDASM/urdf/qwe.SLDASM.urdf")
