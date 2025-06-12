import numpy as np

from diff_tool import diff_solve, visualize_truss

SQ2 = np.sqrt(2)
# F = (0, 0, 0)
F = (0.001, 0, -0.05)
C = 1
MAX_FLOOR = 10

points = []
edges = []
force = []
stiffs = []

for i in range(MAX_FLOOR):
    j = i-1
    height = 2*i
    if i % 2 == 0:
        points.append((1, 1, height))
        points.append((-1, 1, height))
        points.append((-1, -1, height))
        points.append((1, -1, height))
        if i != 0:
            edges.append((4*j, 4*i))
            edges.append((4*j, 4*i+1))
            edges.append((4*j+1, 4*i+1))
            edges.append((4*j+1, 4*i+2))
            edges.append((4*j+2, 4*i+2))
            edges.append((4*j+2, 4*i+3))
            edges.append((4*j+3, 4*i+3))
            edges.append((4*j+3, 4*i))
    else:
        points.append((0, SQ2, height))
        points.append((-SQ2, 0, height))
        points.append((0, -SQ2, height))
        points.append((SQ2, 0, height))
        if i != 0:
            edges.append((4*j, 4*i+3))
            edges.append((4*j, 4*i))
            edges.append((4*j+1, 4*i))
            edges.append((4*j+1, 4*i+1))
            edges.append((4*j+2, 4*i+1))
            edges.append((4*j+2, 4*i+2))
            edges.append((4*j+3, 4*i+2))
            edges.append((4*j+3, 4*i+3))
    edges.append((4*i, 4*i+1))
    edges.append((4*i+1, 4*i+2))
    edges.append((4*i+2, 4*i+3))
    edges.append((4*i+3, 4*i))
    for _ in range(4):
        force.append(F)
    for _ in range(4):
        stiffs.append(C)
    if i != 0:
        for _ in range(8):
            stiffs.append(C)

edges.append((4*MAX_FLOOR-4, 4*MAX_FLOOR-2))
edges.append((4*MAX_FLOOR-3, 4*MAX_FLOOR-1))
stiffs.append(C)
stiffs.append(C)

fixed = [0, 1, 2, 3]
displacements = diff_solve(force, edges, stiffs, points, fixed)
visualize_truss(points, displacements, edges, force, fixed)
