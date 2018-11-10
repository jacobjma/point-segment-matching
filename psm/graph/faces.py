import itertools

import numpy as np
from scipy.sparse.csgraph import connected_components

from psm.graph.graphutils import adjacency2matrix, adjacency2edges, find_clockwise
from psm.utils import labels2groups


def face_adjacency(faces):
    """ Returns the adjacency of a set of faces.
    Two faces are considered adjacent if they share one or more edges.
    """

    faces = [set([frozenset((face[i - 1], face[i])) for i in range(len(face))]) for face in faces]

    adjacency = [set() for i in range(len(faces))]
    for i, face in enumerate(faces):

        for j, other_face in enumerate(faces[:i]):

            if len(set(face).intersection(other_face)) > 0:
                adjacency[i].add(j)
                adjacency[j].add(i)

    return adjacency


def connected_faces(faces):
    adjacency = face_adjacency(faces)

    m = adjacency2matrix(adjacency)
    connected = connected_components(m)[1]

    return labels2groups(connected, sort_by_counts=True)


def find_outer_face(points, adjacency, faces):
    """Return the index of the outer face from a list of faces."""

    leftmost = np.where(points[:, 0] == np.min(points[:, 0]))[0]
    leftmost = leftmost[np.argmin(points[leftmost, 1])]

    adjacent = list(adjacency[leftmost])

    angles = np.arctan2(points[adjacent][:, 1] - points[leftmost, 1],
                        points[adjacent][:, 0] - points[leftmost, 0])

    other = adjacent[np.argmax((angles - np.pi / 2) % (2 * np.pi))]

    pair = '|' + str(leftmost) + '|' + str(other) + '|'

    for i, face in enumerate(faces):
        if pair in ('|' + '|'.join(map(str, face + [face[0]])) + '|'):
            return i

    raise RuntimeError('Outer face not found.')


def find_faces(points, adjacency, remove_outer_face=True):
    """Return the faces of a plane graph."""

    clockwise = find_clockwise(points, adjacency)

    edges = adjacency2edges(adjacency)
    edges = [list(itertools.permutations(edge)) for edge in edges]
    edges = set(edge for sublist in edges for edge in sublist)

    faces = []
    while edges:
        edge = edges.pop()
        faces.append([edge[0]])

        while 1:
            edge = clockwise[tuple(reversed(edge))]

            if edge in edges:
                faces[-1].append(edge[0])
                edges.remove(edge)
            else:
                break

    if remove_outer_face:
        outer_face = find_outer_face(points, adjacency, faces)
        del faces[outer_face]

    return faces


def traverse_perimeter(points, adjacency):
    current = np.where(points[:, 0] == np.min(points[:, 0]))[0]
    current = current[np.argmin(points[current, 1])]

    prev_angle = np.pi / 2
    traversal = []
    while 1:
        traversal.append(current)
        adjacent = adjacency[current]

        try:
            if len(adjacent) > 1:
                adjacent = list(adjacent - {traversal[-2]})
            else:
                adjacent = list(adjacent)
        except IndexError:
            adjacent = list(adjacent)

        if len(adjacent) > 1:
            angles = np.arctan2(points[adjacent][:, 1] - points[current, 1],
                                points[adjacent][:, 0] - points[current, 0])

            angle_diffs = angles - prev_angle

            if np.any(angle_diffs == 0.):
                max_idx = np.where(angle_diffs == 0.)[0][0]
            else:
                max_idx = np.argmax(angle_diffs % (2 * np.pi))
        else:
            max_idx = 0

        prev_angle = np.arctan2(points[current, 1] - points[adjacent[max_idx], 1],
                                points[current, 0] - points[adjacent[max_idx], 0])
        current = adjacent[max_idx]

        try:
            if current == traversal[1]:
                if traversal[0] == traversal[-1]:
                    break
        except IndexError:
            pass

    return traversal[:-1]


def _reduce_perimeter(perimeter):
    keep = []
    for i in range(len(perimeter)):
        A = perimeter[i]
        P = perimeter[i - 1]
        B = perimeter[i - 2]

        det = (P[0] - A[0]) * (B[1] - A[1]) - (P[1] - A[1]) * (B[0] - A[0])

        if det >= 0.:
            keep.append((i - 1) % len(perimeter))

    return keep


def _perimeter2hull(perimeter):
    new_perimeter = perimeter.copy()
    old_hull = range(len(perimeter))

    if len(perimeter) < 4:
        return old_hull

    while 1:
        hull = _reduce_perimeter(new_perimeter)
        new_perimeter = perimeter[hull].copy()

        hull = [old_hull[i] for i in hull]

        if len(hull) == len(old_hull):
            break
        elif len(hull) <= 3:
            break

        old_hull = hull

    return hull


def convex_hull(points, adjacency):
    perimeter = traverse_perimeter(points, adjacency)
    #import matplotlib.pyplot as plt
    #plt.plot(*points[perimeter].T)
    #plt.show()

    hull = _perimeter2hull(points[perimeter])
    #print(perimeter, hull)
    return [perimeter[i] for i in hull]
