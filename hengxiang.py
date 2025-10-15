import numpy as np

# 优化后的 X (2x4)
X = np.array([
    [1, -16, 2, 6],  # 异常值 20，其他元素幅值接近 5~6
    [-2, 8, -1, -9]   # 异常值 20 分散在另一列
], dtype=float)

# 优化后的 W (4x3)
W = np.array([
    [2, 1, -2],
    [1, -1, -1],  # 异常值 10
    [2, -1, -2],
    [-1, -1, 1]    # 异常值 8
], dtype=float)

# 4x4 Hadamard 矩阵
H4 = np.array([
    [1,  1,  1,  1],
    [1, -1,  1, -1],
    [1,  1, -1, -1],
    [1, -1, -1,  1]
], dtype=float)

R = H4 / 2.0   # 归一化正交矩阵
R_inv = R.T    # 逆矩阵

# 旋转操作
XR = X @ R
RinvW = R_inv @ W
XR_RinvW = XR @ RinvW
XW = X @ W

# 打印结果
np.set_printoptions(precision=2, suppress=True)

print("X (2x4):\n", X, "\n")
print("W (4x3):\n", W, "\n")
print("R (4x4) - 正交 Hadamard:\n", R, "\n")
print("XR = X @ R:\n", XR, "\n")
print("R^-1 W = R.T @ W:\n", RinvW, "\n")
print("XW = X @ W:\n", XW, "\n")
print("XR @ R^-1 W:\n", XR_RinvW, "\n")
