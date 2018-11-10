import itertools

import numpy as np

from psm.segments import Segments
from psm.graph.graphutils import find_clockwise
from psm.graph.traversal_slow import clockwise_traversal_with_depth


def build_lattice_points(a, b, max_index, basis=None):
    # TODO: Docstring
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
    # TODO: Docstring
    h = range(-max_index, max_index + 1)

    indices = {}
    for i, (h, k) in enumerate(itertools.product(h, h)):
        indices[(h, k)] = (basis_size * i, basis_size * i + 1)

    return indices


def lattice(a, b, max_index, basis=None):
    # TODO: Docstring
    return Segments(build_lattice_points(a, b, max_index, basis))


def lattice_segment(a, b, max_depth, min_alpha=0, basis=None, max_index=None):
    # TODO: Docstring
    if basis is None:
        basis = np.array([[0., 0.]])
    else:
        basis = np.array(basis, dtype=float)

    if max_index is None:
        max_index = 11

    segments = lattice(a=a, b=b, max_index=max_index, basis=basis)

    segments.build_graph(min_alpha)

    root = build_lattice_dict(max_index, len(basis))[(0, 0)][0]
    edge = (root, next(iter(segments.adjacency[root])))

    clockwise = find_clockwise(segments.points, segments.adjacency)

    traversal, _ = clockwise_traversal_with_depth(edge, segments.adjacency, clockwise, max_depth)

    segments._indices = [traversal]

    return segments


def regular_polygon_points(sidelength, n):
    points = np.zeros((n, 2))

    L = sidelength / (2 * np.sin(np.pi / n))

    points[:, 0] = np.cos(np.arange(n) * 2 * np.pi / n) * L
    points[:, 1] = np.sin(np.arange(n) * 2 * np.pi / n) * L

    return points


def regular_polygon(sidelength, n, include_center=False):
    points = regular_polygon_points(sidelength, n)
    adjacency = [set(((j - 1) % n, (j + 1) % n)) for j in range(n)]
    segments = [range(n)]

    if include_center is True:
        points = np.vstack(([[0, 0]], points))
        adjacency = [set(range(1, n + 1))] + [set((((j - 1) % n)+1, ((j + 1) % n)+1)) for j in range(n)]
        for adjacent in adjacency[1:]:
            adjacent.add(0)
        segments = [range(n + 1)]

    return Segments(points, segments, adjacency)


def regular_polygons(sidelength, n_sides):
    # TODO: Docstring

    segments = Segments()
    for n in n_sides:
        segments.extend(regular_polygon(sidelength, n))

    return segments
