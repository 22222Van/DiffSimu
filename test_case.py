# coding:utf-8

from diff_tool import diff_solve, visualize_truss
import numpy as np

if __name__ == "__main__":
    coords_2d = np.array([
        [0, 0],  # 节点0
        [1, 0],  # 节点1
        [0.5, np.sqrt(3)/2]  # 节点2（等边三角形）
    ])

    A_2d = [[0, 1],
            [0, 2],
            [1, 2]]
    C_2d = [1, 1, 1]

    f_2d = np.array([
        [0, 0],
        [0, 0],
        [0, -0.1]
    ])

    fixed = [0, 1]

    u_2d = diff_solve(f_2d, A_2d, C_2d, coords_2d, fixed)
    visualize_truss(coords_2d, u_2d, A_2d, f_2d, fixed)

    # 3D示例
    coords_3d = np.array([
        [0, 0, 0],  # 节点0
        [1, 0, 0],  # 节点1
        [0, 1, 0],  # 节点2
        [0, 0, 1],  # 节点3
    ])
    A_3d = np.array([
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 3]
    ])
    C_3d = [1, 1, 1, 1, 1, 1]
    f_3d = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, -0.1]
    ])  # 节点3处施加-z方向力
    fixed = [0, 1, 2]

    u_3d = diff_solve(f_3d, A_3d, C_3d, coords_3d, fixed)
    visualize_truss(coords_3d, u_3d, A_3d, f_3d, fixed)
