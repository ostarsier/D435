import numpy as np

"""
       我有相机坐标系下的坐标(x,y,z),其坐标系X朝右，Z朝前，Y朝下。
       我如何把坐标系转换为，X朝前，Y朝左，Z朝上
       例如坐标(1,2,3) 应转换为(3,-1,-2)
       """
R = np.array([[0, 0, 1],
              [-1, 0, 0],
              [0, -1, 0]])

p_cam = np.array([1, 2, 3])  # (x_C, y_C, z_C)
p_new = R @ p_cam  # (x_N, y_N, z_N)
print(p_new)  # [ 3 -1 -2]
"""
我有相机坐标系下的坐标(x,y,z),X朝前，Y朝左，Z朝上
我想平移(a,b,c)，
"""
a=0.075
b=0
c=0.4
p_zero=p_new+np.array([a, b, c])