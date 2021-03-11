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

        for j, other_face in enumerate(faces[(i + 1):]):

            if len(set(face).intersection(other_face)) > 0:
                adjacency[i].add(j + 1)
                adjacency[j + 1].add(i)

    return adjacency


def connected_faces(faces):

    adjacency = face_adjacency(faces)

    connected = connected_components(adjacency2matrix(adjacency))[1]

    return labels2groups(connected, sort_by_counts=True)


def find_outer_face(points, adjacency, faces):
    """Return the index of the outer face from a list of faces."""

    leftmost = np.argmin(points[:, 0])

    slopes = (points[list(adjacency[leftmost])][:, 0] - points[leftmost, 0]) / \
             (points[list(adjacency[leftmost])][:, 1] - points[leftmost, 1])
    pair = str(list(adjacency[leftmost])[np.argmin(slopes)]) + '|' + str(leftmost)

    for i, face in enumerate(faces):
        if pair in '|'.join(map(str, face+[face[0]])):
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
        del faces[find_outer_face(points, adjacency, faces)]

    return faces
