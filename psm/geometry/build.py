import itertools
import numbers

import numpy as np

from psm.graph.geometric import urquhart
from psm.graph.graphutils import find_clockwise, subgraph
from psm.graph.traversal import clockwise_traversal_with_depth, clockwise_traversal_with_depth_and_skips
from psm.register import MatchGraph
from psm.structures import Structures
from psm.utils import flatten


def build_lattice_points(a, b, max_index, basis=None):
    a = np.array(a)
    b = np.array(b)

    if basis is None:
        basis = np.array([[0., 0.]])
    else:
        basis = np.array(basis, dtype=float)

    h = range(-max_index, max_index + 1)
    h, k = np.meshgrid(h, h)
    indices = np.stack((k, h), axis=-1)

    points = np.sum(indices[:, :, None, :] * np.stack((a, b), axis=-1)[None, None, :, :], axis=-1)

    points = points[:, :, None, :] + basis[None, None, :, :]

    return points.reshape((np.prod(points.shape[:3]), 2))


def build_lattice_dict(max_index, basis_size):
    h = range(-max_index, max_index + 1)

    indices = {}
    for i, (h, k) in enumerate(itertools.product(h, h)):
        indices[(h, k)] = (basis_size * i, basis_size * i + 1)

    return indices


def lattice_traversal(a, b, basis=None, radius=None, skips=None, max_depth=None, graph_func=None, rmsd_calc=None,
                      tol=1e-6):
    if graph_func is None:
        graph_func = urquhart

    if basis is None:
        basis = np.array([[0., 0.]])
    else:
        basis = np.array(basis, dtype=float)

    max_index = 11

    points = build_lattice_points(a, b, max_index, basis)

    lattice_dict = build_lattice_dict(max_index, len(basis))

    adjacency = graph_func(points)

    clockwise = find_clockwise(points, adjacency)

    root = lattice_dict[(0, 0)][0]

    traversals = []
    for i in adjacency[root]:
        edge = (root, i)

        if skips is None:
            traversal, _ = clockwise_traversal_with_depth(edge, adjacency, clockwise, max_depth)
        else:
            traversal, _ = clockwise_traversal_with_depth_and_skips(edge, adjacency, clockwise, max_depth, skips)

        if radius is not None:
            order = {j: i for i, j in enumerate(traversal)}
            keep = np.where(np.linalg.norm(points[traversal], axis=1) <= radius)[0]
            traversal = [traversal[i] for i in sorted(keep, key=order.get)]

        traversals.append(traversal)

    structures = Structures(points, traversals, adjacency)

    if len(traversals) > 1:
        rmsd_calc = MatchGraph(transform='rigid', pivot='front', isomorphism=True)

        rmsd_calc.register(structures, structures, progress_bar=False)

        return rmsd_calc.principal_structures(len(adjacency[0]), tol)
    else:
        return structures


def regular_polygons(sidelength, sides):
    points = np.zeros((np.sum(sides), 2))
    segments = []
    adjacency = []

    i = 0
    for n in sides:
        L = sidelength / (2 * np.sin(np.pi / n))

        points[i:i + n, 0] = np.cos(np.arange(n) * 2 * np.pi / n) * L
        points[i:i + n, 1] = np.sin(np.arange(n) * 2 * np.pi / n) * L

        segments.append(range(i, i + n))
        adjacency += [set((i + (j - 1) % n, i + (j + 1) % n)) for j in range(n)]

        i += n

    return Structures(points, segments, adjacency)
