#coding:utf-8

from typing import Optional, Union, Sequence
import numpy as np

from numpy.typing import NDArray

def diff_2D(
    f: NDArray[np.floating],
    A: NDArray[np.floating],
    C: NDArray[np.floating],
    coords: NDArray[np.floating],  # 节点坐标数组，形状为(m, 2)
    fixed: Optional[Union[NDArray[np.floating], Sequence[int]]] = None
) -> NDArray[np.floating]:
    """
    Solve 2D Truss Problem (A'CAu = f) using node coordinates
    
    m: node number
    n: truss number
    
    param:
        coords: (m, 2) array of node coordinates [x, y]
        f: (2*m, 1) force vector [f1_x, f1_y, ..., fm_x, fm_y]
        A: (n, m) connectivity matrix
        C: (n, n) stiffness coefficients matrix
        fixed: fixed node indices (0-based), None makes matrix singular
    
    return:
        u: (2*m, 1) displacement vector [u1_x, u1_y, ..., um_x, um_y]
    """
    m = A.shape[1]
    n = A.shape[0]
    
    # 验证坐标数组形状
    if coords.shape != (m, 2):
        raise ValueError(f"coords must have shape ({m}, 2), got {coords.shape}")
    
    A_2d = np.zeros((n, 2 * m))
    
    for i in range(n):
        # 找到杆件的起始节点和结束节点
        row = A[i]
        start_idx = np.where(row == 1)[0]
        end_idx = np.where(row == -1)[0]
        
        if len(start_idx) != 1 or len(end_idx) != 1:
            raise ValueError(f"Row {i} of A must contain exactly one 1 and one -1")
        
        start_node = start_idx[0]
        end_node = end_idx[0]
        
        # 获取节点坐标
        x1, y1 = coords[start_node]
        x2, y2 = coords[end_node]
        
        # 计算方向向量和长度
        dx = x2 - x1
        dy = y2 - y1
        L = np.sqrt(dx**2 + dy**2)
        
        if L < 1e-10:  # 避免除以零
            c, s = 0.0, 0.0
        else:
            c = dx / L  # x方向的方向余弦
            s = dy / L  # y方向的方向余弦
        
        # 填充A_2d矩阵
        A_2d[i, 2*start_node] = c
        A_2d[i, 2*start_node + 1] = s
        A_2d[i, 2*end_node] = -c
        A_2d[i, 2*end_node + 1] = -s
    
    # 计算刚度矩阵
    K = A_2d.T @ C @ A_2d
    
    # 处理固定节点
    if fixed is None:
        fixed = []
    
    # 固定节点转换为自由度索引
    fixed_dofs = []
    for node in fixed:
        fixed_dofs.append(2 * node)
        fixed_dofs.append(2 * node + 1)
    
    active = [i for i in range(2 * m) if i not in fixed_dofs]
    
    # 求解简化系统
    K_reduced = K[active, :][:, active]
    f_reduced = f[active]
    
    u_reduced = np.linalg.solve(K_reduced, f_reduced)
    
    # 组装完整位移向量
    u = np.zeros(2 * m)
    u[active] = u_reduced
    
    return u

def diff_3D(
    f: NDArray[np.floating],
    A: NDArray[np.floating],
    C: NDArray[np.floating],
    coords: NDArray[np.floating],  # 节点坐标数组，形状为(m, 3)
    fixed: Optional[Union[NDArray[np.floating], Sequence[int]]] = None
) -> NDArray[np.floating]:
    """
    Solve 3D Truss Problem (A'CAu = f) using node coordinates
    
    m: node number
    n: truss number
    
    param:
        coords: (m, 3) array of node coordinates [x, y, z]
        f: (3*m, 1) force vector [f1_x, f1_y, f1_z, ..., fm_x, fm_y, fm_z]
        A: (n, m) connectivity matrix
        C: (n, n) stiffness coefficients matrix
        fixed: fixed node indices (0-based), None makes matrix singular
    
    return:
        u: (3*m, 1) displacement vector [u1_x, u1_y, u1_z, ..., um_x, um_y, um_z]
    """
    m = A.shape[1]
    n = A.shape[0]
    
    # 验证坐标数组形状
    if coords.shape != (m, 3):
        raise ValueError(f"coords must have shape ({m}, 3), got {coords.shape}")
    
    A_3d = np.zeros((n, 3 * m))
    
    for i in range(n):
        # 找到杆件的起始节点和结束节点
        row = A[i]
        start_idx = np.where(row == 1)[0]
        end_idx = np.where(row == -1)[0]
        
        if len(start_idx) == 0 or len(end_idx) == 0:
            raise ValueError(f"Row {i} of A must contain exactly one 1 and one -1")
        
        start_node = start_idx[0]
        end_node = end_idx[0]
        
        # 获取节点坐标
        x1, y1, z1 = coords[start_node]
        x2, y2, z2 = coords[end_node]
        
        # 计算方向向量和长度
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        L = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if L < 1e-10:  # 避免除以零
            cx, cy, cz = 0.0, 0.0, 0.0
        else:
            cx = dx / L  # x方向的方向余弦
            cy = dy / L  # y方向的方向余弦
            cz = dz / L  # z方向的方向余弦
        
        # 填充A_3d矩阵
        A_3d[i, 3*start_node] = cx
        A_3d[i, 3*start_node + 1] = cy
        A_3d[i, 3*start_node + 2] = cz
        
        A_3d[i, 3*end_node] = -cx
        A_3d[i, 3*end_node + 1] = -cy
        A_3d[i, 3*end_node + 2] = -cz
    
    # 计算刚度矩阵
    K = A_3d.T @ C @ A_3d
    
    # 处理固定节点
    if fixed is None:
        fixed = []
    
    # 固定节点转换为自由度索引
    fixed_dofs = []
    for node in fixed:
        fixed_dofs.extend([3*node, 3*node+1, 3*node+2])
    
    active = [i for i in range(3 * m) if i not in fixed_dofs]
    
    # 求解简化系统
    K_reduced = K[active, :][:, active]
    f_reduced = f[active]
    
    u_reduced = np.linalg.solve(K_reduced, f_reduced)
    
    # 组装完整位移向量
    u = np.zeros(3 * m)
    u[active] = u_reduced
    
    return u