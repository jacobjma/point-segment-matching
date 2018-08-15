import time

from psm.graph.clockwise import clockwise_traversal
from psm.build import lattice_traversal
from psm.graph.graphutils import find_clockwise
from psm.graph.traversal_slow import clockwise_traversal as clockwise_traversal_slow

a = [1, 0]
b = [0, 1]

structures = lattice_traversal(a, b, max_depth=3)
points = structures.points
adjacency = structures.adjacency
clockwise = find_clockwise(points, adjacency)
edge = (0, 1)

N = 10000

start = time.time()
for i in range(N):
    traversal = clockwise_traversal_slow(edge, adjacency, clockwise)
end = time.time()
print(end - start)

start = time.time()
for i in range(N):
    traversal = clockwise_traversal(edge, adjacency, clockwise)
end = time.time()
print(end - start)
