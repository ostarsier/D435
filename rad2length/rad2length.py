import numpy as np

def rotation_matrix_x(angle_rad):
    """
    生成绕X轴逆时针旋转的3x3旋转矩阵。
    angle_rad: 旋转角度（弧度）。
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array(,
        [0, c, -s],
        [0, s, c])

def rotation_matrix_y(angle_rad):
    """
    生成绕Y轴逆时针旋转的3x3旋转矩阵。
    angle_rad: 旋转角度（弧度）。
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([c, 0, s],
        ,
        [-s, 0, c])

# 1. 定义初始点 A=[a,b,c]
# 假设点 A 为 [1, 2, 3] 作为示例
a, b, c = 1, 2, 3
point_A = np.array([a, b, c])
print(f"初始点 A: {point_A}\n")

# 定义旋转角度
# 假设绕X轴旋转 30 度，绕Y轴旋转 45 度
# 将角度转换为弧度，因为 numpy 的三角函数需要弧度
angle_alpha_deg = 30
angle_beta_deg = 45
angle_alpha_rad = np.deg2rad(angle_alpha_deg)
angle_beta_rad = np.deg2rad(angle_beta_deg)

print(f"绕X轴旋转角度 (alpha): {angle_alpha_deg} 度 ({angle_alpha_rad:.4f} 弧度)")
print(f"绕Y轴旋转角度 (beta): {angle_beta_deg} 度 ({angle_beta_rad:.4f} 弧度)\n")

# 2. 第一次旋转：绕X轴旋转
R_x = rotation_matrix_x(angle_alpha_rad)
print("绕X轴旋转矩阵 R_x(alpha):\n", R_x)

# 计算绕X轴旋转后的中间点 A'
point_A_prime = R_x @ point_A
print(f"\n绕X轴旋转后的中间点 A': {point_A_prime}")

# 3. 第二次旋转：绕Y轴旋转
R_y = rotation_matrix_y(angle_beta_rad)
print("\n绕Y轴旋转矩阵 R_y(beta):\n", R_y)

# 4. 计算两次旋转的组合旋转矩阵
# 由于是先绕X轴旋转，再绕Y轴旋转，且是对点进行主动变换，
# 矩阵乘法顺序为 R_total = R_y @ R_x
R_total = R_y @ R_x
print("\n组合旋转矩阵 R_total = R_y(beta) * R_x(alpha):\n", R_total)

# 5. 计算旋转后的点在基坐标系中的坐标
# 最终点 A'' = R_total @ point_A
point_A_double_prime = R_total @ point_A
print(f"\n旋转后的点在基坐标系中的坐标 A'': {point_A_double_prime}")

# 验证中间点 A' 再次绕Y轴旋转是否得到相同结果
# point_A_double_prime_check = R_y @ point_A_prime
# print(f"\n验证：中间点 A' 再次绕Y轴旋转结果: {point_A_double_prime_check}")