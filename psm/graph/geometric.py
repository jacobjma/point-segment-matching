import itertools
from collections import defaultdict

import numpy as np
import scipy.spatial
import sklearn.cluster
import sklearn.neighbors

from psm.graph.graphutils import edges2adjacency


def _ccw(a, b, c):
    return np.cross(b - a, c - a) <= 0


def gabriel(points):
    """Return the Gabriel Graph as an adjacency list."""

    def intersect(a, b):
        return ((_ccw(a[0], b[0], b[1]) != _ccw(a[1], b[0], b[1])) &
                (_ccw(a[0], a[1], b[0]) != _ccw(a[0], a[1], b[1])))

    simplices = scipy.spatial.Delaunay(points).simplices

    edges = np.array([list(edge) for edge in _simplex_edges(simplices)])

    vor = scipy.spatial.Voronoi(points)

    to_keep = np.zeros(len(edges), dtype=bool)
    for i, edge in enumerate(edges):
        a = vor.regions[vor.point_region[edge[0]]]
        b = vor.regions[vor.point_region[edge[1]]]

        common = np.intersect1d(a, b, assume_unique=True)

        if intersect(vor.vertices[common], points[edge]):
            to_keep[i] = True

    return edges2adjacency(edges[to_keep], len(points))


def _simplex_edges(simplices):
    edges = [[frozenset((simplex[i - 1], simplex[i])) for i in range(3)] for simplex in simplices]
    return set(itertools.chain.from_iterable(edges))


def urquhart(points):
    """Return the Urquhart Graph as an adjacency list."""

    simplices = scipy.spatial.Delaunay(points).simplices
    edges = _simplex_edges(simplices)
    lengths = {edge: np.linalg.norm(np.diff([points[i] for i in edge], axis=0), axis=1)[0] for edge in edges}

    to_remove = set()
    for simplex in simplices:
        simplex_edges = [frozenset(tuple(np.take(simplex, [k, k + 1], mode='wrap'))) for k in range(3)]
        longest = simplex_edges[np.argmax([lengths[edge] for edge in simplex_edges])]

        to_remove.add(longest)

    edges = edges - to_remove

    edges = np.array([list(edge) for edge in edges])

    return edges2adjacency(edges, len(points))


def _calc_circumcenter(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    a = np.linalg.det([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]])
    bx = -np.linalg.det([[x1 ** 2 + y1 ** 2, y1, 1],
                         [x2 ** 2 + y2 ** 2, y2, 1],
                         [x3 ** 2 + y3 ** 2, y3, 1]])

    by = np.linalg.det([[x1 ** 2 + y1 ** 2, x1, 1],
                        [x2 ** 2 + y2 ** 2, x2, 1],
                        [x3 ** 2 + y3 ** 2, x3, 1]])
    x0 = -bx / (2 * a)
    y0 = -by / (2 * a)
    return np.array([x0, y0])


def _simplex_edges(simplices):
    edges = []
    for s in simplices:
        for i in range(len(s)):
            a = s[i]
            b = s[(i + 1) % 3]
            edges += [(a, b)]
    return np.array(edges)


def _order_exterior_vertices(simplices):
    edges = _simplex_edges(simplices).tolist()

    tally = defaultdict(list)
    for i, item in enumerate(edges):
        tally[tuple(sorted(item))].append(i)

    edges = {edges[locs[0]][0]: edges[locs[0]][1] for locs
             in tally.values() if len(locs) == 1}

    order = [list(edges.keys())[0]]
    while len(order) < len(edges):
        order += [edges[order[-1]]]

    return order


def circumcenter_clustering(points, k=.5, min_points=2, return_faces=False):
    simplices = scipy.spatial.Delaunay(points).simplices

    vertices = np.array([_calc_circumcenter(*points[s]) for s in simplices])

    nbrs = sklearn.cluster.NearestNeighbors(n_neighbors=2).fit(points)
    distances, indices = nbrs.kneighbors(points)

    eps = k * np.median(distances[:, 1])
    estimator = sklearn.neighbors.DBSCAN(eps=eps, min_samples=min_points)

    labels = estimator.fit_predict(vertices)

    faces = []
    for i in range(labels.max() + 1):
        v = np.where(labels == i)[0]
        order = _order_exterior_vertices(simplices[v])
        faces += [order]

    adjacency = [set() for i in range(len(points))]
    for face in faces:
        for i, j in enumerate(face):
            adjacency[j].add(face[i - 1])

    if return_faces:
        return adjacency, faces
    else:
        return adjacency
