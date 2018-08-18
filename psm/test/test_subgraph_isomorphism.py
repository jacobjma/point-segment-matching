import matplotlib.pyplot as plt

from psm import plotutils
from psm.build import lattice_traversal
from psm.graph.graphutils import subgraph, find_clockwise
from psm.graph.traversal_slow import clockwise_traversal
from psm.graph.isomorphism_slow import subgraph_isomorphism

import random

# TODO : Make test more comprehensive

a = [1, 0]
b = [0, 1]
show_plots = True

structures = lattice_traversal(a, b, max_depth=5)
points = structures.points
adjacency = structures.adjacency
clockwise = find_clockwise(points, adjacency)

i = 0
n = 20
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

isomorphisms = subgraph_isomorphism(adjacency, clockwise, subgraph_adjacency)

if not show_plots:
    assert len(isomorphisms) >= 4
    quit()

isomorphic_points = points[isomorphisms[0]]
isomorphic_adjacency = subgraph(adjacency, isomorphisms[0])

fig, axes = plt.subplots(1, 3)
plotutils.graph_embedding(points, adjacency, ax=axes[0])

plotutils.graph_embedding(subgraph_points, subgraph_adjacency, ax=axes[1])

plotutils.graph_embedding(isomorphic_points, isomorphic_adjacency, ax=axes[2])

for i, point in enumerate(points):
    axes[0].annotate('{}'.format(i), xy=point)

for i, point in enumerate(subgraph_points):
    axes[1].annotate('{}'.format(i), xy=point)

for i, point in enumerate(isomorphic_points):
    axes[2].annotate('{}'.format(i), xy=point)

plt.show()
