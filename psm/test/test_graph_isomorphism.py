import numpy as np
import matplotlib.pyplot as plt
from psm.build import lattice_traversal
from psm import plotutils
from psm.graph.graphutils import subgraph, find_clockwise
#from psm.graph.isomorphism import subgraph_isomorphisms
from subgraph_isomorphism import subgraph_isomorphisms

a = [1, 0]
b = [0, 1]

structures = lattice_traversal(a, b, max_depth=3)
points = structures.points
adjacency = structures.adjacency
clockwise = find_clockwise(points, adjacency)

selected = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12]

subgraph_points = points[selected]
subgraph_adjacency = subgraph(adjacency, selected)

isomorphisms = subgraph_isomorphisms(adjacency, clockwise, subgraph_adjacency)

subgraph_adjacency = subgraph(adjacency, isomorphisms[0])

print(isomorphisms)
print(len(isomorphisms))
print(np.array(isomorphisms))

fig, axes = plt.subplots(1, 2)
plotutils.graph_embedding(points, adjacency, ax=axes[0])

plotutils.graph_embedding(subgraph_points, subgraph_adjacency, ax=axes[1])

for i, point in enumerate(points):
    axes[0].annotate('{}'.format(i), xy=point)

for i, point in enumerate(subgraph_points):
    axes[1].annotate('{}'.format(i), xy=point)

plt.show()
