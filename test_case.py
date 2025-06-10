from diff_tool import diff_2D,diff_3D
import numpy as np

if __name__ == "__main__":
    # Example truss with 2 nodes (m=2) and 1 truss (n=1)
    A = np.array([[1, -1]])  # Truss connects node 1 to node 2
    C = np.array([[1000]])   # Stiffness coefficient
    theta = np.array([45])   # 45 degree angle
    f = np.array([0, 0, 0, -100])  # Vertical force at node 2

    u = diff_2D(f, A, C, theta, is_rad=False, fixed=[1])
    print("Displacements:", u)  # Singular matrix

    # Example truss with 3 nodes and 2 truss
    A = np.array([[1, -1, 0], [0, 1, -1]])
    C = np.array([[100, 0], [0, 100]])
    theta = np.array([60, -60])
    f = np.array([0, 0, 0, -50, 0, 0])
    u = diff_2D(f, A, C, theta, is_rad=False, fixed=[0, 1])
    # [ 0.00000000e+00  0.00000000e+00 -4.04256138e+16  2.33397390e+16 -7.03687442e+15  4.26167367e+16]
    print("Displacements:", u)


    A = np.array([
        [1, -1, 0, 0],   
        [0, 1, -1, 0],   
        [0, 0, 1, -1],   
        [1, 0, 0, -1],  
        [0, 1, 0, -1]    
    ])
    C = np.diag([1000]*5)  
    theta_xy = np.array([30, 30, 30, 45, 45])  
    theta_xz = np.array([0, 0, 0, 30, 45])     

   
    f = np.array([0, 0, 0,   
                0, 0, -20,   
                0, 0, 0,    
                0, 0, 0])    

   
    fixed = [0, 2,3] 

    u = diff_3D(f, A, C, theta_xy, theta_xz, is_rad=False, fixed=fixed)
    print("Displacements:", u)

    import matplotlib.pyplot as plt


    # 桁架连接数据
    A = np.array([
        [1, -1, 0, 0],   # 桁架1-2
        [0, 1, -1, 0],   # 桁架2-3
        [0, 0, 1, -1],   # 桁架3-4
        [1, 0, 0, -1],   # 对角桁架1-4
        [0, 1, 0, -1]    # 对角桁架2-4
    ])

    # 假设所有桁架长度为1单位长度(简化计算)
    length = 1

    # 计算节点坐标
    def calculate_positions(theta_xy, theta_xz, length=1):
        theta_xy_rad = np.deg2rad(theta_xy)
        theta_xz_rad = np.deg2rad(theta_xz)
        
        x = length * np.sin(theta_xy_rad) * np.cos(theta_xz_rad)
        y = length * np.sin(theta_xy_rad) * np.sin(theta_xz_rad)
        z = length * np.cos(theta_xy_rad)
        
        return x, y, z

    # 计算相邻节点相对位置
    pos12 = calculate_positions(30, 0, length)
    pos23 = calculate_positions(30, 0, length)
    pos34 = calculate_positions(30, 0, length)
    pos14 = calculate_positions(45, 30, length)
    pos24 = calculate_positions(45, 45, length)

    # 计算绝对坐标
    node1 = np.array([0, 0, 0])
    node2 = node1 + pos12
    node3 = node2 + pos23
    node4 = node3 + pos34

    # 修正对角桁架坐标
    node4_alt1 = node1 + pos14
    node4_alt2 = node2 + pos24
    node4 = (node4 + node4_alt1 + node4_alt2) / 3  # 取平均值

    # 绘制3D图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制桁架
    connections = [
        (node1, node2), (node2, node3), (node3, node4),
        (node1, node4), (node2, node4)
    ]

    colors = ['red', 'green', 'blue', 'orange', 'purple']
    labels = ['Truss 1-2', 'Truss 2-3', 'Truss 3-4', 'Diagonal 1-4', 'Diagonal 2-4']

    for (start, end), color, label in zip(connections, colors, labels):
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                color=color, linewidth=2, marker='o', markersize=6, label=label)

    # 标记节点
    nodes = [node1, node2, node3, node4]
    for i, node in enumerate(nodes, 1):
        ax.text(node[0], node[1], node[2], f'Node {i}', fontsize=12)

    # 设置图形属性
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Truss Structure Visualization', fontsize=14)
    ax.legend(loc='upper right')

    # 调整视角
    ax.view_init(elev=30, azim=45)

    # 显示网格和等比例
    ax.grid(True)
    ax.set_box_aspect([1,1,1])  # 保持等比例

    plt.tight_layout()
    plt.show()