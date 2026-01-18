
from scipy.spatial.transform import Rotation
import numpy as np

# 1. 解决侧卧: 绕 Y 轴 +90 度
r1 = Rotation.from_euler('y', 90, degrees=True)

# 2. 解决朝向 (X=左右 -> X=前后): 绕 Z 轴 -90 度
# 注意旋转顺序：是相对于世界系还是局部系？
# 假设原始模型：前=Y, 上=X, 右=Z (侧卧且朝Y)
# 目标：前=X, 上=Z, 左=Y

# 让我们直接定义映射：
# Old X -> New Z (上)
# Old Y -> New X (前)
# Old Z -> New Y (左)

# 这种旋转矩阵是：
# [0, 1, 0]
# [0, 0, 1]
# [1, 0, 0]

mat = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
])

quat = Rotation.from_matrix(mat).as_quat()
# scipy returns (x, y, z, w), Isaac needs (w, x, y, z)
print(f"Correct Quat (w,x,y,z): {quat[3]:.4f}, {quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}")
