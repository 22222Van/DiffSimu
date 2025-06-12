# coding:utf-8

from typing import Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import ArrayLike, NDArray


def diff_solve(
    forces: ArrayLike,
    edges: ArrayLike,
    stiffness: ArrayLike,
    coords: ArrayLike,
    fixed: Optional[Sequence[int]] = None
) -> NDArray[np.floating]:
    """
    Solve 2D Truss Problem (A'CAu = f) using node coordinates

    m: node number
    n: truss number
    dim: dimension number

    param:
        coords: (m, dim) array of node coordinates [x, y]
        forces: (m, dim) force vector [[f1_x, ...], ..., [fm_x, ...]]
        edges: (n, 2) connectivity matrix
        stiffness: (n, ) stiffness coefficients
        fixed: fixed node indices, None makes matrix singular

    return:
        (m, dim) displacement vector [[u1_x, ...], ..., [um_x, ...]]
    """
    forces = np.array(forces, dtype=np.float64)
    edges = np.array(edges, dtype=np.int32)
    stiffness = np.array(stiffness, dtype=np.float64)
    coords = np.array(coords, dtype=np.float64)

    m, dim = coords.shape
    n = edges.shape[0]  # number of trusses

    if forces.shape != coords.shape:
        raise ValueError(
            f"f must have shape same as coords, got {forces.shape=}, {coords.shape=}"
        )

    # 验证坐标数组形状
    if edges.shape != (n, 2):
        raise ValueError(
            f"edges must have shape ({n}, 2), got {edges.shape}"
        )

    if stiffness.shape != (n,):
        raise ValueError(
            f"stiffness must have shape ({n},), got {stiffness.shape}"
        )

    C = np.diag(stiffness)

    forces = forces.ravel()

    A = np.zeros((n, dim * m))

    for i in range(n):
        # 找到杆件的起始节点和结束节点
        start_node, end_node = edges[i]

        # 获取起始和结束节点坐标
        start_coords = coords[start_node]  # shape: (dim,)
        end_coords = coords[end_node]      # shape: (dim,)

        # 计算方向向量和长度
        direction = end_coords - start_coords  # shape: (dim,)
        L = np.linalg.norm(direction)

        if L < 1e-10:
            print(
                f'[Warning] length of edge {start_node}->{end_node} is too small.'
            )

        unit_dir = direction / L  # 单位方向向量（任意维）

        # 填充A矩阵
        A[i, dim * start_node: dim * (start_node + 1)] = unit_dir
        A[i, dim * end_node: dim * (end_node + 1)] = -unit_dir

    # 计算刚度矩阵
    K = A.T @ C @ A

    # 处理固定节点
    if fixed is None:
        fixed = []

    fixed_dofs = [dim * node + d for node in fixed for d in range(dim)]
    active = [i for i in range(dim * m) if i not in fixed_dofs]

    # 求解简化系统
    K_reduced = K[active, :][:, active]
    f_reduced = forces[active]

    u_reduced = np.linalg.solve(K_reduced, f_reduced)

    # 组装完整位移向量
    u = np.zeros(dim * m)
    u[active] = u_reduced

    return u.reshape((m, dim))


def visualize_truss(
    coords,
    displacement,
    edges,
    forces,
    fixed,
    force_scale=0.1,
    color_orig='gray',
    color_def='blue',
    color_force='red',
    color_fixed='black'
):
    new_coords = coords + displacement
    dim = coords.shape[1]
    if dim == 2:
        plt.figure()
        # 原始结构：虚线
        for e in edges:
            plt.plot(coords[e, 0], coords[e, 1],
                     linestyle='--', marker='o', color=color_orig)
        # 变形后结构：实线
        for e in edges:
            plt.plot(new_coords[e, 0], new_coords[e, 1],
                     linestyle='-', marker='o', color=color_def)
        # 受力箭头
        for i, f in enumerate(forces):
            if np.linalg.norm(f) > 0:
                plt.quiver(
                    new_coords[i, 0],
                    new_coords[i, 1],
                    f[0],
                    f[1],
                    angles='xy',
                    scale_units='xy',
                    scale=1/force_scale,
                    color=color_force
                )
        # 固定节点
        fixed_coords = coords[fixed]
        plt.scatter(
            fixed_coords[:, 0],
            fixed_coords[:, 1],
            marker='x',
            color=color_fixed,
            s=200
        )
        plt.axis('equal')
        plt.title('2D Truss Visualization')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # 原始结构
        for e in edges:
            ax.plot(
                coords[e, 0],
                coords[e, 1],
                coords[e, 2],
                linestyle='--',
                marker='o',
                color=color_orig
            )
        # 变形后结构
        for e in edges:
            ax.plot(
                new_coords[e, 0],
                new_coords[e, 1],
                new_coords[e, 2],
                linestyle='-',
                marker='o',
                color=color_def
            )
        # 受力箭头
        for i, f in enumerate(forces):
            length = np.linalg.norm(f)
            if length > 0:
                ax.quiver(
                    new_coords[i, 0],
                    new_coords[i, 1],
                    new_coords[i, 2],
                    f[0],
                    f[1],
                    f[2],
                    length=length*force_scale,
                    color=color_force
                )
        # 固定节点
        fixed_coords = coords[fixed]
        ax.scatter(
            fixed_coords[:, 0],
            fixed_coords[:, 1],
            fixed_coords[:, 2],
            marker='x',
            color=color_fixed,
            s=200
        )
        ax.set_title('3D Truss Visualization')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    else:
        print("Only 2D and 3D supported.")
