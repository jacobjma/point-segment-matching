import random
import time

from psm.graph.subgraph_isomorphism import subgraph_isomorphism

from psm.build import lattice_traversal
from psm.graph.graphutils import subgraph, find_clockwise
from psm.graph.isomorphism_slow import subgraph_isomorphism as subgraph_isomorphism_slow
from psm.graph.traversal_slow import clockwise_traversal

a = [1, 0]
b = [0, 1]
N = 100

structures = lattice_traversal(a, b, max_depth=3)
points = structures.points
adjacency = structures.adjacency
clockwise = find_clockwise(points, adjacency)

i = 0
n = 10
selected = set((0,))
while len(selected) < n:
    i = random.sample(adjacency[i], 1)[0]
    selected.add(i)

selected = list(selected)

subgraph_points = points[selected]
subgraph_adjacency = subgraph(adjacency, selected)

root = 0
edge = (root, random.sample(subgraph_adjacency[root], 1)[0])

subgraph_clockwise = find_clockwise(subgraph_points, subgraph_adjacency)
traversal = clockwise_traversal(edge, subgraph_adjacency, subgraph_clockwise)

subgraph_points = subgraph_points[traversal]
subgraph_adjacency = subgraph(subgraph_adjacency, traversal)

# import matplotlib.pyplot as plt
# plt.plot(points[:,0],points[:,1],'o')
# plt.plot(subgraph_points[:,0],subgraph_points[:,1],'x')
# plt.show()

start = time.time()
for i in range(N):
    isomorphisms = subgraph_isomorphism(adjacency, clockwise, subgraph_adjacency)
end = time.time()
print((end - start) / N)

start = time.time()
for i in range(N):
    isomorphisms = subgraph_isomorphism_slow(adjacency, clockwise, subgraph_adjacency)
end = time.time()
print((end - start) / N)
