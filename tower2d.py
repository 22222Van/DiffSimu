import numpy as np

from diff_tool import diff_solve, visualize_truss

import numpy as np
from diff_tool import diff_solve, visualize_truss

SQ2 = np.sqrt(2)

F =(0.001, -0.05) 
C = 1                 
MAX_FLOOR = 10       

points = []    
edges = []     
force = []     
stiffs = []    


#F1: (-SQ2, 0) (SQ2,0) (0,0) 
#F2: (1,2) (-1,2)
#F3: (-SQ2, 4) (SQ2,4) (0,4) 
#F4: (1,6) (-1,6)
#Edges: (0,2),(1,2),(0,4),(1,3),(3,4),(2,4),(2,3)
#......


for i in range(MAX_FLOOR):
    height = 2 * i

    if i % 2 == 0:  
        if i == 0:
            n0 = (-SQ2, height)
            n2 = (SQ2, height)
            n1 = (0, height)
            points.extend([n0, n1, n2])
            edges.extend([
                (0, 1),  # (-SQ2, 0) to (0, 0)
                (1, 2),  # (SQ2, 0) to (0, 0)
            ])
            stiffs.extend([C]*2)
        else:
            n0 = (-SQ2, height)
            n2 = (SQ2, height)
            n1 = (0, height)
            points.extend([n0, n1, n2])
            edge_indices = [
                (2.5*i, 2.5*i+1),
                (2.5*i+1, 2.5*i+2),
                (2.5*i, 2.5*i-2),
                (2.5*i+1, 2.5*i-1),
                (2.5*i+1, 2.5*i-2),
                (2.5*i+2, 2.5*i-1)
            ]
            edges.extend([(int(a), int(b)) for a, b in edge_indices])
            stiffs.extend([C]*6)
        force.extend([F]*3)
    else:  
        n1 = (1, height)
        n0 = (-1, height)
        points.extend([n0, n1])
        edge_indices = [
            (2.5*i+0.5, 2.5*i+1.5),
            (2.5*i+0.5, 2.5*i-2.5),
            (2.5*i+0.5, 2.5*i-1.5),
            (2.5*i+1.5, 2.5*i-1.5),
            (2.5*i+1.5, 2.5*i-0.5)
        ]
        edges.extend([(int(a), int(b)) for a, b in edge_indices])
        stiffs.extend([C]*5)
        force.extend([F]*2)



fixed = [0, 1,2]  


displacements = diff_solve(force, edges, stiffs, points, fixed)
visualize_truss(points, displacements, edges, force, fixed)