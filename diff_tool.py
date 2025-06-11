from typing import Optional, Union, Sequence
import numpy as np

from numpy.typing import NDArray

def diff_2D(
    f: NDArray[np.floating],
    A: NDArray[np.floating],
    C: NDArray[np.floating],
    theta: NDArray[np.floating],
    is_rad: bool,
    fixed: Optional[Union[NDArray[np.floating], Sequence[int]]] = None
) -> NDArray[np.floating]:
    """
    Solve 2D Truss Problem (A'CAu = f)

    m: node number
    n: truss number

    param:
        theta: (n,1), the angle between truss and horizontal, should be 0~90 degree
        is_rad: if theta is rad , True; if theta is degree, False
        f : (2*m,1) force decomposed into horizontal force and vertical force , e.g. f_1^H, f_2^V
        A : (n, m) connectivity
        C : (n, n) coefficents
        fixed: fixed node id , should be in range(m); when None, the matrix becomes singular!

    return:
        u : (2*m,1), displacement decomposed into horizontal displacement and vertical displacement , e.g. u_1^H, u_2^V
    """
    m = A.shape[1]
    n = A.shape[0]

    # Convert degree to rad.
    if not is_rad:
        theta_rad = np.deg2rad(theta)
    else:
        theta_rad = theta

    # Direction cosines for each truss
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)

    A_2d = np.zeros((n, 2 * m))

    for i in range(n):
        start_node = np.argmax(A[i] == 1)
        end_node = np.argmax(A[i] == -1)

        A_2d[i, 2*start_node] = c[i]
        A_2d[i, 2*start_node + 1] = s[i]

        A_2d[i, 2*end_node] = -c[i]
        A_2d[i, 2*end_node + 1] = -s[i]

    # Stiffness Matrix
    K = A_2d.T @ C @ A_2d

    if fixed is None:
        fixed = []  # Singular!

    active = [i for i in range(2 * m) if i not in fixed]

    K_reduced = K[active, :][:, active]
    f_reduced = f[active]

    u_reduced = np.linalg.solve(K_reduced, f_reduced)

    u = np.zeros(2 * m)
    u[active] = u_reduced

    return u



def diff_3D(
    f: NDArray[np.floating],
    A: NDArray[np.floating],
    C: NDArray[np.floating],
    theta_xy: NDArray[np.floating],
    theta_xz: NDArray[np.floating],
    is_rad: bool,
    fixed: Optional[Union[NDArray[np.floating], Sequence[int]]] = None
) -> NDArray[np.floating]:
    """
    Solve 3D Truss Problem (A'CAu = f)

    m: node number
    n: truss number

    param:
        theta_xy: (n,1), the angle between truss and xy-plane (elevation)
        theta_xz: (n,1), the angle between truss's xy-projection and x-axis (azimuth)
        is_rad: if angles are in radians, True; if degrees, False
        f: (3*m,1) force decomposed into x, y, z components (f_1^x, f_1^y, f_1^z, ...)
        A: (n, m) connectivity matrix
        C: (n, n) coefficients matrix
        fixed: fixed node indices (0-based), should be in range(m); None makes matrix singular

    return:
        u: (3*m,1), displacement decomposed into x, y, z components (u_1^x, u_1^y, u_1^z, ...)
    """
    m = A.shape[1]
    n = A.shape[0]

    # Convert degree to rad if needed
    if not is_rad:
        theta_xy_rad = np.deg2rad(theta_xy)
        theta_xz_rad = np.deg2rad(theta_xz)
    else:
        theta_xy_rad = theta_xy
        theta_xz_rad = theta_xz

    # Direction cosines for each truss
    c_xy = np.cos(theta_xy_rad)
    s_xy = np.sin(theta_xy_rad)
    c_xz = np.cos(theta_xz_rad)
    s_xz = np.sin(theta_xz_rad)

    # Calculate direction components (x,y,z)
    cx = s_xy * c_xz  # x component
    cy = s_xy * s_xz  # y component
    cz = c_xy         # z component

    A_3d = np.zeros((n, 3 * m))

    for i in range(n):
        start_node = np.argmax(A[i] == 1)
        end_node = np.argmax(A[i] == -1)

        # Start node components
        A_3d[i, 3*start_node] = cx[i]
        A_3d[i, 3*start_node + 1] = cy[i]
        A_3d[i, 3*start_node + 2] = cz[i]

        # End node components
        A_3d[i, 3*end_node] = -cx[i]
        A_3d[i, 3*end_node + 1] = -cy[i]
        A_3d[i, 3*end_node + 2] = -cz[i]

    # Stiffness Matrix
    K = A_3d.T @ C @ A_3d

    if fixed is None:
        fixed = []  # Singular matrix

    # Convert node indices to degree of freedom indices
    fixed_dofs = []
    for node in fixed:
        fixed_dofs.extend([3*node, 3*node+1, 3*node+2])

    active = [i for i in range(3 * m) if i not in fixed_dofs]

    K_reduced = K[active, :][:, active]
    f_reduced = f[active]

    u_reduced = np.linalg.solve(K_reduced, f_reduced)

    u = np.zeros(3 * m)
    u[active] = u_reduced

    return u
