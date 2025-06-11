#coding:utf-8

from diff_tool import diff_2D, diff_3D
import numpy as np

if __name__ == "__main__":
    
    coords_2d = np.array([
        [0, 0],  # 节点0
        [1, 0],  # 节点1
        [0.5, np.sqrt(3)/2]  # 节点2（等边三角形）
    ])

    A_2d = np.array([
    [1, -1, 0],   # 杆0 连接 节点0 -> 节点1
    [1, 0, -1],   # 杆1 连接 节点0 -> 节点2
    [0, 1, -1]    # 杆2 连接 节点1 -> 节点2
])
    
    C_2d = np.eye(3) 

    f_2d = np.array([0, 0, 0, 0, 0, -100])  # [f0_x, f0_y, f1_x, f1_y, f2_x, f2_y]

    fixed = [0, 1]

    u_2d = diff_2D(f_2d, A_2d, C_2d, coords_2d, fixed)

    print("Displacement:",u_2d)

    # 3D示例
    coords_3d = np.array([
    [0, 0, 0],  # 节点0
    [1, 0, 0],  # 节点1
    [0, 1, 0],  # 节点2
    [0, 0, 1],  # 节点3
])
    A_3d = np.array([
        [1, -1, 0, 0],  # 杆0 连接 0 -> 1
        [1, 0, -1, 0],  # 杆1 连接 0 -> 2
        [1, 0, 0, -1],  # 杆2 连接 0 -> 3
        [0, 1, -1, 0],  # 杆3 连接 1 -> 2
        [0, 1, 0, -1],  # 杆4 连接 1 -> 3
        [0, 0, 1, -1]   # 杆5 连接 2 -> 3
    ])
    C_3d = np.diag([1, 1, 1,1,1,1])
    f_3d = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100])  # 节点3处施加-z方向力
    fixed = [0,1,2]  

    u_3d = diff_3D(f_3d, A_3d, C_3d, coords_3d, fixed)
    print("Displacement:",u_3d)