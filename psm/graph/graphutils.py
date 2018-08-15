import itertools

import numpy as np
from scipy.sparse.csgraph import connected_components

from psm.utils import flatten, labels2groups


def adjacency2edges(adjacency):
    edges = [frozenset(itertools.product([i], vertices)) for i, vertices in enumerate(adjacency)]
    return set(flatten(edges))


def edges2adjacency(edges, num_nodes):
    adjacency = [set() for i in range(num_nodes)]

    for edge in edges:
        adjacency[edge[0]].add(edge[1])
        adjacency[edge[1]].add(edge[0])
    return adjacency


def adjacency2matrix(adjacency):
    matrix = np.zeros((len(adjacency),) * 2, dtype=bool)
    for i, adjacent in enumerate(adjacency):
        for j in adjacent:
            matrix[i, j] = True
            matrix[j, i] = True
    return matrix


def connected_pieces(adjacency):
    connected = connected_components(adjacency2matrix(adjacency))[1]
    return labels2groups(connected, sort_by_counts=True)


def subgraph(adjacency, indices):
    if type(indices) is np.ndarray:
        if indices.dtype is np.dtype('bool'):
            if len(indices) != len(adjacency):
                raise ValueError()

            indices = np.where(indices)[0]

        return [set(np.where(indices == j)[0][0] for j in adjacency[i] if j in indices) for i in indices]
    else:
        return [set(indices.index(j) for j in adjacency[i] if j in indices) for i in indices]


def find_clockwise(points, adjacency):
    """Create a dict mapping a directed edge to the next edge
    adjacent its tail going in the clockwise direction.
    """

    clockwise = {}
    for i, point in enumerate(points):
        adjacent = list(adjacency[i])

        adjacent_points = points[adjacent] - point
        angle = np.arctan2(adjacent_points[:, 0], adjacent_points[:, 1])
        adjacent = [adjacent[j] for j in np.argsort(angle)]

        for j, k in enumerate(adjacent):
            clockwise[(i, adjacent[j - 1])] = (i, k)

    return clockwise
