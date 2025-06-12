# coding:utf-8

from typing import Optional, Union, Sequence
import numpy as np

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
    forces = np.array(forces)
    edges = np.array(edges)
    stiffness = np.array(stiffness)
    coords = np.array(coords)

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
