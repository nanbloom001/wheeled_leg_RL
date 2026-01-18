
from scipy.spatial.transform import Rotation
import numpy as np

def main():
    # 当前状态 (基于 Isaac Lab 的 rot=(0.5, 0.5, 0.5, 0.5))
    # 导致: Robot Up = World X+, Robot Front = World Y-
    
    # 我们想要的目标:
    # Robot Up -> World Z+
    # Robot Front -> World X+
    
    # 原始 CAD 的坐标系定义 (推测):
    # 既然 rot=(0.5, 0.5, 0.5, 0.5) 对应 XYZ->YZX 变换
    # 这意味着:
    # New X (背) = Old Y
    # New Y (左) = Old Z
    # New Z (头?) = Old X
    
    # 让我们不要推测原始 CAD，直接找"从当前错误姿态到正确姿态"的修正旋转
    # 当前: X_curr (背), Y_curr (头反向?)
    # 目标: Z_target (背), X_target (头)
    
    # 需要的旋转:
    # 把 World X+ 转到 World Z+ (绕 Y 轴 -90 度)
    # 把 World Y- 转到 World X+ (绕 Z 轴 +90 度)
    
    # 让我们尝试直接定义 "Body Frame in World Frame" 的旋转矩阵
    # 假设 Body Frame 定义: x=前, y=左, z=上
    # 原始 CAD 可能是: x=上, y=前, z=右 (或者类似的乱序)
    
    # 现在的现象:
    # 身体的上(背) 指向了 World X
    # 身体的前(头) 指向了 World -Y
    # 身体的左     指向了 World -Z (由右手定则 X x -Y = -Z)
    
    # 我们要建立一个旋转 R，使得:
    # R * [1, 0, 0] (原CAD的X轴) = [0, 0, 1] (变成 World Z/上)
    # R * [0, 1, 0] (原CAD的Y轴) = [-1, 0, 0] (变成 World -X/后?) -> 不对，头朝Y-说明CAD里Y是反向的头?
    
    # 让我们试错法:
    # 现在的 rot=(0.5, 0.5, 0.5, 0.5) 显然不对。
    # 让我们回到单位四元数 (1, 0, 0, 0)
    # 如果设为 (1, 0, 0, 0)，也就是不旋转:
    # 请告诉我此时的姿态？
    # 
    # 或者，我们直接基于"背朝X, 头朝-Y"这个现状进行修正。
    # 我们需要把 X 轴转到 Z 轴 (绕Y -90)，把 -Y 轴转到 X 轴 (绕Z +90)。
    pass

if __name__ == "__main__":
    # 直接生成几种常用组合供测试
    rots = {
        "Identity": (1.0, 0.0, 0.0, 0.0),
        "Rot X 90": (0.7071, 0.7071, 0.0, 0.0),
        "Rot X -90": (0.7071, -0.7071, 0.0, 0.0),
        "Rot Y 90": (0.7071, 0.0, 0.7071, 0.0),
        "Rot Y -90": (0.7071, 0.0, -0.7071, 0.0), # 之前用过，解决了侧卧但可能方向反了
        "Rot Z 90": (0.7071, 0.0, 0.0, 0.7071),
        "Rot Z -90": (0.7071, 0.0, 0.0, -0.7071),
    }
    # 还有一些复合的
    # 比如 (0.5, -0.5, 0.5, -0.5)
    
    print("Use manual trial for reliability.")
