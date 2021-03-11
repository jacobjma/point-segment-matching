from collections import defaultdict

import numpy as np
import scipy.spatial
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy import optimize

from psm.graph.graphutils import edges2adjacency
from psm.utils import flatten, set_difference
from psm.graph.faces import find_faces, find_outer_face
import numba


def _calc_angle(p, r, q):
    rp = p - r
    rq = q - r
    return np.arccos(np.dot(rp, rq) /
                     (np.linalg.norm(rp) * np.linalg.norm(rq)))


@numba.njit
def triangle_angles(p, r, q):
    a = np.sqrt(np.sum((r - q) ** 2, axis=1))
    b = np.sqrt(np.sum((q - p) ** 2, axis=1))
    c = np.sqrt(np.sum((p - r) ** 2, axis=1))
    a2 = a ** 2
    b2 = b ** 2
    c2 = c ** 2
    angles = np.zeros((len(p), 3))
    angles[:, 0] = np.arccos((b2 + c2 - a2) / (2 * b * c))
    angles[:, 1] = np.arccos((a2 + c2 - b2) / (2 * a * c))
    angles[:, 2] = np.pi - angles[:, 0] - angles[:, 1]
    return angles


@numba.njit
def stable_delaunay_cluster(points, simplices, neighbors, in_hull, threshold):
    angles = triangle_angles(points[simplices[:, 0]], points[simplices[:, 1]], points[simplices[:, 2]])
    labels = np.full(len(simplices), -1, np.int32)

    # assign all labels to faces in clustered with the outer face of the triangulation
    for i in in_hull:
        if labels[i] == -1:
            alpha = np.inf
            for j in np.where(neighbors[i] == -1)[0]:
                alpha = min(np.pi - angles[i][j], alpha)

            if alpha < threshold:
                queue = [i]
                while queue:
                    i = queue.pop()
                    for j in neighbors[i]:
                        if j != -1:
                            if labels[j] == -1:
                                k = set_difference(simplices[i], simplices[j])
                                l = set_difference(simplices[j], simplices[i])
                                alpha = np.pi - (angles[i][simplices[i] == k][0] + angles[j][simplices[j] == l][0])

                                if alpha < threshold:
                                    labels[j] = 0
                                    queue += [j]

    # assign labels to all other faces
    max_label = 1
    for i in range(0, len(simplices)):
        if labels[i] == -1:
            labels[i] = max_label
            queue = [i]
            while queue:
                i = queue.pop()
                for j in neighbors[i]:
                    if labels[j] == -1:
                        k = set_difference(simplices[i], simplices[j])
                        l = set_difference(simplices[j], simplices[i])
                        alpha = np.pi - (angles[i][simplices[i] == k][0] + angles[j][simplices[j] == l][0])

                        if alpha < threshold:
                            labels[j] = max_label
                            queue += [j]

            max_label += 1

    return labels


# @numba.njit
def delaunay_edge_stability(points, simplices):
    # TODO : Performance critical, implement in Cython

    alphas = defaultdict(lambda: np.pi)
    for i, simplex in enumerate(simplices):
        for j in range(3):
            p = simplex[j - 1]
            r = simplex[j - 2]
            q = simplex[j]

            alphas[frozenset((p, q))] -= _calc_angle(*points[[p, r, q]])

    return dict(alphas)


def estimate_min_alpha(alphas, n_neighbors, exclude=None):
    if exclude is None:
        exclude = []

    node_alphas = defaultdict(list)
    for edge, alpha in alphas.items():
        edge = list(edge)
        node_alphas[edge[0]].append(alpha)
        node_alphas[edge[1]].append(alpha)

    node_alphas = [sorted(values) for node, values in node_alphas.items() if node not in exclude]

    def fun(alpha):
        return - sum([n_neighbors == sum(i > alpha for i in j) for j in node_alphas])

    minimum = optimize.brute(fun, ranges=((0, np.pi),))

    return minimum[0]


def stable_delaunay(points, min_alpha=0, max_aspect=np.inf):
    simplices = scipy.spatial.Delaunay(points).simplices

    if max_aspect is not None:
        ar = np.array([_aspect_ratio(*points[s]) for s in simplices])
        simplices = simplices[ar < max_aspect]

    edges = _simplex_edges(simplices)

    alphas = delaunay_edge_stability(points, simplices)

    edges = [edge for edge in edges if alphas[edge] >= min_alpha]

    adjacency = edges2adjacency(edges, len(points))

    return adjacency, alphas


def _ccw(a, b, c):
    return np.cross(b - a, c - a) <= 0


def gabriel(points):
    """Return the Gabriel graph as an adjacency list."""

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
    return set(flatten(edges))


def urquhart(points):
    """Return the Urquhart graph as an adjacency list."""

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


def _aspect_ratio(A, B, C):
    a = np.linalg.norm(A - B)
    b = np.linalg.norm(B - C)
    c = np.linalg.norm(C - A)
    s = (a + b + c) / 2
    return a * b * c / (8 * (s - a) * (s - b) * (s - c))


def delaunay(points, max_aspect=None):
    """Return the Delaunay graph as an adjacency list."""
    simplices = scipy.spatial.Delaunay(points).simplices

    if max_aspect is not None:
        ar = np.array([_aspect_ratio(*points[s]) for s in simplices])
        simplices = simplices[ar < max_aspect]

    edges = _simplex_edges(simplices)
    edges = np.array([list(edge) for edge in edges])

    return edges2adjacency(edges, len(points))


def knn(points, k, condition='bilateral'):
    """Return the k-nearest neighbors graph as an adjacency list."""
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    _, groups = nbrs.kneighbors(points)
    groups = groups[:, 1:]

    adjacency = [set(list(group)) for group in groups]

    if condition is 'bilateral':
        for i, adjacent in enumerate(adjacency):
            for j in adjacent:
                if i not in adjacency[j]:
                    adjacency[i] = adjacency[i] - {j}
    elif condition is 'unilateral':
        for i, adjacent in enumerate(adjacency):
            for j in adjacent:
                adjacency[j].add(i)
    else:
        raise ValueError()

    return adjacency


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


#@numba.njit
def _directed_simplex_edges(simplices):
    edges = [[(simplex[i - 1], simplex[i]) for i in range(3)] for simplex in simplices]
    return flatten(edges)


def _order_exterior_vertices(simplices):
    edges = _directed_simplex_edges(simplices)

    tally = defaultdict(list)
    for i, item in enumerate(edges):
        tally[tuple(sorted(item))].append(i)

    edges = {edges[locs[0]][0]: edges[locs[0]][1] for locs in tally.values() if len(locs) == 1}

    order = [list(edges.keys())[0]]
    while len(order) < len(edges):
        order += [edges[order[-1]]]

    return order


def circumcenter_clustering(points, tol=.5, min_points=2, return_faces=False):
    # TODO: docstring
    simplices = scipy.spatial.Delaunay(points).simplices

    vertices = np.array([_calc_circumcenter(*points[s]) for s in simplices])

    nbrs = NearestNeighbors(n_neighbors=2).fit(points)
    distances, indices = nbrs.kneighbors(points)

    eps = tol * np.median(distances[:, 1])
    estimator = DBSCAN(eps=eps, min_samples=min_points)

    labels = estimator.fit_predict(vertices)

    faces = []
    for i in range(labels.max() + 1):
        v = np.where(labels == i)[0]
        order = _order_exterior_vertices(simplices[v])
        faces += [order]

    adjacency = [set() for _ in range(len(points))]
    for face in faces:
        for i, j in enumerate(face):
            adjacency[j].add(face[i - 1])

    if return_faces:
        return adjacency, faces
    else:
        return adjacency
