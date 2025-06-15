# DiffSimu

**SI114: Computational Science and Engeneering Project**

DiffSimu is a Python tool for finite element analysis and visualization of 2D and 3D truss structures. 

## Features

- Static analysis of 2D/3D truss structures
- Customizable nodes, bars, stiffness, external forces, and constraints
- Visualization of results (Matplotlib)

## Dependencies

- numpy
- matplotlib

Install dependencies:
```sh
pip install numpy matplotlib
```

## File Structure

- `diff_tool.py`: Core algorithms and visualization functions
- `test_case.py`: 2D and 3D truss test cases
- `tower2d.py`: 2D tower truss example
- `tower3d.py`: 3D tower truss example

## Quick Start

Run 2D and 3D test cases:

```sh
python test_case.py
```

Run the 2D tower truss example:

```sh
python tower2d.py
```

Run the 3D tower truss example:

```sh
python tower3d.py
```

## Usage

See [`diff_solve`](diff_tool.py) and [`visualize_truss`](diff_tool.py) for core function documentation.

### 2D Example

```python
import numpy as np
from diff_tool import diff_solve, visualize_truss

coords = np.array([
    [0, 0],
    [1, 0],
    [0.5, np.sqrt(3)/2]
])
edges = [[0, 1], [0, 2], [1, 2]]
stiffness = [1, 1, 1]
forces = np.array([
    [0, 0],
    [0, 0],
    [0, -0.1]
])
fixed = [0, 1]

u = diff_solve(forces, edges, stiffness, coords, fixed)
visualize_truss(coords, u, edges, forces, fixed)
```


