import numpy as np
from scipy.sparse.csgraph import connected_components

from psm.utils import labels2groups
import numba


def adjacency_array_to_list_of_sets(array, num_adjacent):
    return [set(adjacent[:l]) for adjacent, l in zip(array, num_adjacent)]


def adjacency_list_of_sets_to_array(list_of_sets):
    num_adjacent = np.array([len(adjacent) for adjacent in list_of_sets])
    adjacency = np.full((len(list_of_sets), max(num_adjacent)), -1, dtype=np.int32)
    for i, adjacent in enumerate(list_of_sets):
        adjacency[i, :num_adjacent[i]] = list(adjacent)
    return adjacency, num_adjacent




@numba.njit
def adjacency_to_edges(adjacency, lengths):
    n = len(adjacency)
    edges = np.zeros((n * adjacency.shape[1], 2), dtype=np.int32)
    k = 0
    for i in range(n):
        for j in adjacency[i][:lengths[i]]:
            if i > j:
                edges[k, 0] = i
                edges[k, 1] = j
                k += 1
    return edges[:k]


@numba.njit
def edges_to_adjacency(edges, num_nodes, max_adjacent=10):
    adjacency = np.full((num_nodes, max_adjacent), -1)
    insertion = np.zeros(num_nodes, dtype=np.int32)
    for i in range(len(edges)):
        n0 = edges[i][0]
        n1 = edges[i][1]
        adjacency[n0, insertion[n0]] = n1
        insertion[n0] += 1
        adjacency[n1, insertion[n1]] = n0
        insertion[n1] += 1
    return adjacency[:, :np.max(insertion)], insertion


@numba.njit
def adjacency_to_matrix(adjacency, lengths):
    n = len(adjacency)
    matrix = np.full((n, n), True)
    for i in range(n):
        for j in adjacency[i][:lengths[i]]:
            matrix[i, j] = True
            matrix[j, i] = True
    return matrix


def adjacency2edges(adjacency):
    edges = set()
    for i, adjacent in enumerate(adjacency):
        for j in adjacent:
            edges.add(frozenset((i, j)))

    return [list(edge) for edge in edges]


def edges2adjacency(edges, num_nodes):
    adjacency = [set() for _ in range(num_nodes)]

    for edge in edges:
        edge = list(edge)
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
    return [frozenset(indices.index(j) for j in adjacency[i] if j in indices) for i in indices]


# def subgraph(adjacency, indices):
#     if type(indices) is np.ndarray:
#         if indices.dtype is np.dtype('bool'):
#             if len(indices) != len(adjacency):
#                 raise ValueError()
#
#             indices = np.where(indices)[0]
#
#         return [set(np.where(indices == j)[0][0] for j in adjacency[i] if j in indices) for i in indices]
#     else:
#         return [set(indices.index(j) for j in adjacency[i] if j in indices) for i in indices]


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
