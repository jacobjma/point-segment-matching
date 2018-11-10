from collections import deque

import numpy as np
from scipy.optimize import least_squares

from psm.cluster import Cluster
from psm.graph.faces import connected_faces
from psm.match import FaceMatcher, RMSD
from psm.segments import Segments, select_segments


def assign_miller(faces, vectors, seed=0):
    edge2face = {}
    for i, face in enumerate(faces):
        for j in range(len(face)):
            edge2face[(face[j - 1], face[j])] = i

    next_vector = {}
    for i in range(len(vectors)):
        next_vector[vectors[i - 1]] = vectors[i]
        next_vector[(-vectors[i - 1][0], -vectors[i - 1][1])] = (-vectors[i][0], -vectors[i][1])

    miller = {faces[seed][0]: (0, 0)}
    queue = deque([(faces[seed][1], faces[seed][0])])
    marked = set((seed,))
    for i, j in enumerate(faces[seed][1:]):
        miller[j] = (miller[faces[seed][i]][0] + vectors[i][0],
                     miller[faces[seed][i]][1] + vectors[i][1])

        queue.append((faces[seed][(i + 2) % len(faces[seed])], faces[seed][i + 1]))

    while queue:
        edge = queue.pop()

        try:
            face_idx = edge2face[edge]

        except KeyError:
            pass

        else:
            if face_idx not in marked:
                face = faces[face_idx]
                marked.add(face_idx)
                first = face.index(edge[1])
                for i in range(len(face)):
                    i = (i + first) % len(face)
                    queue.append((face[i], face[i - 1]))

                    if face[i] not in miller.keys():
                        vector = (miller[face[i - 1]][0] - miller[face[i - 2]][0],
                                  miller[face[i - 1]][1] - miller[face[i - 2]][1])

                        vector = next_vector[vector]

                        miller[face[i]] = (miller[face[i - 1]][0] + vector[0],
                                           miller[face[i - 1]][1] + vector[1])

    return miller


def _build_residual(miller):
    def residual(p):
        ax, ay, bx, by, x0, y0 = p
        r = 0
        for (h, k), (x, y) in miller.items():
            r += (h * ax + k * bx + x0 - x) ** 2 + (h * ay + k * by + y0 - y) ** 2
        return r

    return residual


def fit_lattice_parameters(miller):
    ax = miller[(0, 0)][0] - miller[(1, 0)][0]
    ay = miller[(0, 0)][1] - miller[(1, 0)][1]
    bx = ay
    by = ax
    x0 = miller[(0, 0)][0]
    y0 = miller[(0, 0)][1]

    p = (ax, ay, bx, by, x0, y0)

    residual = _build_residual(miller)

    result = least_squares(residual, p)

    a = result.x[:2]
    b = result.x[2:4]
    x0 = result.x[4:]

    return a, b, x0


class BravaisDetector(object):

    def __init__(self, rmsd_max, sample=1):

        self.rmsd_max = rmsd_max
        self.sample = sample

        self._segments = None
        self._pieces = None
        self._rmsd = None

    def detect(self, points):

        self._segments = Segments(points)
        self._segments.build_graph()
        self._segments.faces()

        if self.sample < 1:
            sample = self._segments.sample(self.sample)
        else:
            sample = self._segments

        rmsd_calc = RMSD(transform='rigid', pivot='cop', matcher=FaceMatcher())

        clusterer = Cluster(eps=self.rmsd_max, min_samples=1, rmsd_calc=rmsd_calc)
        clusterer.fit(sample, progress_bar=False)
        principal_structure = clusterer.principal_structures(1, self.rmsd_max)

        rmsd = rmsd_calc.register(principal_structure, self._segments)[0]

        mask = ~np.isnan(rmsd)
        mask[mask] &= rmsd[mask] < self.rmsd_max
        rmsd[~mask] = np.nan

        indices = np.where(mask)[0]

        crystal_faces = [self._segments.indices[i] for i in indices]

        self._pieces = [[indices[i] for i in piece] for piece in connected_faces(crystal_faces)]
        self._rmsd = rmsd

    @property
    def n_faces(self):
        return sum([len(piece) for piece in self._pieces])

    @property
    def rmsd(self):
        return self._rmsd

    def get_crystal_piece(self, i):
        piece = self._pieces[i]

        points, faces = select_segments(piece, self._segments._points, self._segments._indices)

        adjacency = [set() for i in range(len(points))]
        for face in faces:
            for i in range(len(face)):
                adjacency[face[i - 1]].update(set((face[i - 2], face[i])))

        return Segments(points, faces, adjacency)

    def get_miller_indices(self):

        segments = self.get_crystal_piece(0)

        vectors = [(0, 1), (1, -1), (-1, 0)]

        return assign_miller(segments.indices, vectors)

    def get_lattice_parameters(self):

        segments = self.get_crystal_piece(0)

        miller = self.get_miller_indices()

        miller = {(h, k): segments.points[i] for i, (h, k) in miller.items()}

        a, b, x0 = fit_lattice_parameters(miller)

        return a, b, x0
