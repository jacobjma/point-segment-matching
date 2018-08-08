import heapq
import itertools

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import DBSCAN

from psm.graph.graphutils import find_clockwise
from psm.graph.isomorphism import check_isomorphism, subgraph_isomorphisms
from psm.geometry.rmsd import safe_rmsd
from psm.structures import Structures
from psm.utils import noobar


def _affine_transform(src, dst):
    coeffs = range(6)

    xs = src[:, 0]
    ys = src[:, 1]
    xd = dst[:, 0]
    yd = dst[:, 1]
    rows = src.shape[0]

    # params: a0, a1, a2, b0, b1, b2, c0, c1
    A = np.zeros((rows * 2, 9))
    A[:rows, 0] = xs
    A[:rows, 1] = ys
    A[:rows, 2] = 1
    A[:rows, 6] = - xd * xs
    A[:rows, 7] = - xd * ys
    A[rows:, 3] = xs
    A[rows:, 4] = ys
    A[rows:, 5] = 1
    A[rows:, 6] = - yd * xs
    A[rows:, 7] = - yd * ys
    A[:rows, 8] = xd
    A[rows:, 8] = yd

    # Select relevant columns, depending on params
    A = A[:, list(coeffs) + [8]]

    _, _, V = np.linalg.svd(A)

    H = np.zeros((3, 3))
    # solution is right singular vector that corresponds to smallest
    # singular value
    H.flat[list(coeffs) + [8]] = - V[-1, :-1] / V[-1, -1]
    H[2, 2] = 1

    return H


def _relabel(labels):
    outliers = labels == -1
    unique, counts = np.unique(labels[outliers == False], return_counts=True)

    relabeled = -np.ones_like(labels)
    for i, label in enumerate(unique[np.argsort(-counts)]):
        relabeled[labels == label] = i

    return relabeled


class RMSD(object):
    # TODO: Docstring

    def __init__(self, transform='rigid', rotation_invariant=True, scale_invariant=True, pivot='cop'):

        if transform not in ['rigid', 'similarity']:
            raise NotImplementedError

        if pivot not in ['cop', 'front']:
            raise NotImplementedError

        self.transform = transform
        self.scale_invariant = scale_invariant
        self.rotation_invariant = rotation_invariant
        self.pivot = pivot

        self._A = None
        self._B = None

        self._rmsd = None
        self._permutations = None

        self.precalcs = [('pivot', self._calc_pivot),
                         ('scale', self._calc_scale)]

    @property
    def rmsd(self):
        return self._rmsd

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    @property
    def permutations(self):
        return self._permutations

    def _get_other(self, structures):
        if structures == self._A:
            return self._B
        elif structures == self._B:
            return self._A
        else:
            raise RuntimeError

    def _get_axis(self, structures):

        if structures == self._A:
            return 0
        elif structures == self._B:
            return 1
        else:
            raise RuntimeError()

    def _calc_pivot(self, points, precalced):

        if self.pivot == 'cop':
            pivot = np.mean(points, axis=0)
        else:
            pivot = points[0]

        return pivot

    def _calc_norms(self, points, precalced):

        try:
            pivot = precalced['pivot']
        except:
            pivot = self._calc_pivot(points, precalced)

        return np.linalg.norm(points - pivot, axis=1)

    def _calc_scale(self, points, precalced, permutation=None):

        try:
            norms = precalced['norms']
        except:
            norms = self._calc_norms(points, precalced)

        return np.sqrt(np.sum(norms ** 2))

    def _do_precalc(self, A, B):

        precalced = {}
        for structures in [A, B]:

            for i, structure in enumerate(structures):
                precalced[structure] = {}

                for key, func in self.precalcs:
                    precalced[structure][key] = func(structure.points, precalced[structure])

        return precalced

    def _get_points(self, a, b, precalced):

        p = a.points - precalced[a]['pivot']
        q = b.points - precalced[b]['pivot']

        if (self.transform == 'rigid') & self.scale_invariant:
            scale = np.sqrt(precalced[a]['scale'] ** 2 + precalced[b]['scale'] ** 2)
            p = p / scale
            q = q / scale

        elif (self.transform == 'similarity') & self.scale_invariant:
            p = p / precalced[a]['scale']
            q = q / precalced[b]['scale']

        elif (self.transform == 'similarity') & (not self.scale_invariant):
            q = q * precalced[a]['scale'] / precalced[b]['scale']

        return p, q

    def connected_components(self):

        adjacency = np.isnan(self._rmsd) == False

        labels = connected_components(adjacency)[1]

        return labels  # _relabel(labels)

    def is_connected(self, a, b):
        return len(a) == len(b), None

    def get_rmsd(self, a, b, precalced, permutations=None):

        p, q = self._get_points(a, b, precalced)

        rmsd = safe_rmsd(p, q)

        return rmsd, None

    def _cluster_connected(self, rmsd, eps, min_samples=1):

        if not self.A == self.B:
            raise RuntimeError()

        if np.isnan(rmsd).any():
            raise RuntimeError()

        clusterer = DBSCAN(eps=eps, min_samples=min_samples)

        labels = clusterer.fit_predict(rmsd)

        return labels

    def cluster(self, eps, min_samples=1):

        labels = self.connected_components()

        unique, counts = np.unique(labels, return_counts=True)

        clusters = -np.ones_like(labels)
        for label in unique[counts > min_samples]:
            cols = np.where(labels == label)[0]

            connected_rmsd = self.rmsd[cols[:, None], cols]

            new_clusters = self._cluster_connected(connected_rmsd, eps, min_samples)
            inliers = new_clusters > -1

            if inliers.any():
                new_clusters[inliers] = new_clusters[inliers] + np.max(clusters) + 1
                clusters[cols] = new_clusters

        return _relabel(clusters)

    def register(self, A, B, progress_bar=True):

        N, M = len(A), len(B)

        self._A, self._B = A, B
        self._rmsd = np.empty((N, M))
        self._rmsd[:] = np.nan

        if A == B:
            indices = itertools.combinations(range(N), 2)
            np.fill_diagonal(self._rmsd, 0)
            total = N * (N - 1) // 2
        else:
            indices = itertools.product(range(N), range(M))
            total = N * M

        precalced = self._do_precalc(A, B)

        for i, j in noobar(indices, num_iter=total, units='registrations', disable=not progress_bar):

            is_connected, permutations = self.is_connected(A[i], B[j])

            if is_connected:
                rmsd, permutation = self.get_rmsd(A[i], B[j], precalced, permutations)
                self._rmsd[i, j] = rmsd

                if self.permutations is not None:
                    self._permutations[(A[i], B[j])] = permutation

            if A == B:
                self._rmsd[j, i] = self._rmsd[i, j]

        return self._rmsd

    def best_matches(self, structures):

        if structures == self._A:
            rmsd = self.rmsd.copy()
        else:
            rmsd = self.rmsd.copy().T

        best_matches = -np.ones(len(structures), dtype=int)

        valid = np.all(np.isnan(rmsd), axis=1) == 0

        best_matches[valid] = np.nanargmin(rmsd[valid, :], axis=1)

        return best_matches, rmsd[range(len(best_matches)), best_matches]

    def calc_strain(self, structures, match='best', rmsd_max=np.inf):

        if match == 'best':
            matches, rmsd = self.best_matches(structures)
        elif match == 'widest':
            matches, rmsd = self.widest_matches(structures, rmsd_max)
        else:
            raise ValueError()

        strain = np.empty((len(structures), 2, 2))
        strain[:] = np.nan

        rotation = np.empty(len(structures))
        rotation[:] = np.nan

        for i, structure in enumerate(structures):

            other = self._get_other(structures)[matches[i]]

            if rmsd[i] <= rmsd_max:

                p = other.points
                q = structure.points

                if self.permutations is not None:
                    permutation = self.permutations[(other, structure)]
                    q = q[permutation]

                A = _affine_transform(p, q)

                # pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
                # X = pad(other.points)
                # Y = pad(structure.points[permutation])
                # A, res, rank, s = np.linalg.lstsq(X, Y)

                U, P = scipy.linalg.polar(A[:-1, :-1], side='left')

                rotation[i] = np.arctan2(U[0, 1], U[0, 0])

                strain[i] = P - np.identity(2)

        return strain, rotation

    def _rigid_rotation(self, src, dst):

        A = np.dot(dst.T, src)

        V, S, W = np.linalg.svd(A)
        U = np.dot(V, W)

        return np.dot(src, U.T)

    def _average(self, dst, structures, tol=1e-12):

        while 1:
            rotated = np.zeros_like(dst)

            for structure in structures:
                src = structure.points - structure.center

                rotated += self._rigid_rotation(src, dst)

            rotated = rotated / len(structures)

            if np.sum((rotated - dst) ** 2) / len(rotated) < tol:
                break

            dst = rotated

        return rotated


class MatchGraph(RMSD):
    # TODO: Docstring
    def __init__(self, transform='rigid', scale_invariant=True, pivot='cop', isomorphism=False,
                 subgraph_isomorphism=False):

        super(MatchGraph, self).__init__(transform=transform, scale_invariant=scale_invariant, pivot=pivot)

        self.isomorphism = isomorphism
        self.subgraph_isomorphism = subgraph_isomorphism

        self._permutations = {}

        self.precalcs = [('pivot', self._calc_pivot),
                         ('norms', self._calc_norms),
                         ('scale', self._calc_scale)]

    def is_connected(self, a, b):

        if a.adjacency == b.adjacency:
            return True, None
        elif self.isomorphism:
            clockwise = find_clockwise(a.points, a.adjacency)

            return check_isomorphism(a.adjacency, clockwise, b.adjacency), None
        elif self.subgraph_isomorphism:

            clockwise = find_clockwise(b.points, b.adjacency)

            permutations = subgraph_isomorphisms(b.adjacency, clockwise, a.adjacency)

            if len(permutations) > 0:
                return True, permutations

        return False, None

    def _get_partial_points(self, a, b, precalced, permutation):

        p = a.points - precalced[a]['pivot']

        q = b.points[permutation]
        q = q - self._calc_pivot(q, precalced[b])

        if (self.transform == 'rigid') & self.scale_invariant:
            scale = np.sqrt(precalced[a]['scale'] ** 2 + self._calc_scale(q, precalced[b]) ** 2)
            p = p / scale
            q = q / scale

        elif (self.transform == 'similarity') & self.scale_invariant:
            p = p / precalced[a]['scale']
            q = q / np.sqrt(np.sum(precalced[b]['norms'][permutation] ** 2))

        elif (self.transform == 'similarity') & (not self.scale_invariant):
            q = q * precalced[a]['scale'] / self._calc_scale(q, precalced[b])

        return p, q

    def get_rmsd(self, a, b, precalced, permutations=None):

        if permutations is None:
            p, q = self._get_points(a, b, precalced)

            rmsd = safe_rmsd(p, q)

            return rmsd, None
        else:
            rmsd = np.zeros(len(permutations))

            for i, permutation in enumerate(permutations[:]):
                p, q = self._get_partial_points(a, b, precalced, permutation)

                rmsd[i] = safe_rmsd(p, q)

                # rmsd[i] = rmsd_kabsch(p, q)
                # print(rmsd[i])

            return np.min(rmsd), permutations[np.argmin(rmsd)]

    def widest_matches(self, structures, rmsd_max):

        if structures == self._A:
            rmsd = self.rmsd.copy()
        else:
            rmsd = self.rmsd.copy().T

        rmsd[rmsd > rmsd_max] = np.nan

        is_nan = (np.isnan(rmsd) == 0).astype(int)

        widest_matches = -np.ones(len(structures), dtype=int)

        valid = np.all(np.isnan(rmsd), axis=1) == 0

        sizes = np.array([len(structure) for structure in self._get_other(structures)])

        widest_matches[valid] = np.argmax(is_nan * sizes[None, :], axis=1)[valid]

        best_rmsd = np.empty_like(widest_matches, dtype=float)
        best_rmsd[:] = np.nan

        best_rmsd[valid] = rmsd[range(len(widest_matches)), widest_matches][valid]

        return widest_matches, best_rmsd

    def principal_structures(self, n, eps, min_samples=1):

        labels = self.cluster(eps, min_samples)

        unique, counts = np.unique(labels[labels != -1], return_counts=True)

        structures = Structures()
        for label in unique[np.argsort(-counts)][:n]:
            cluster = self.A[np.where(label == labels)[0]]

            dst = cluster[0].points - cluster[0].center

            points = self._average(dst, cluster)

            structures.extend(Structures(points, [range(0, len(dst))], cluster[0].adjacency))

        return structures


class BNB(RMSD):
    # TODO: Docstring
    def __init__(self, tol=1e-2, transform='rigid', scale_invariant=True, pivot='cop'):

        super(BNB, self).__init__(transform=transform, scale_invariant=scale_invariant, pivot=pivot)

        self._max_level = max(int(np.ceil(np.log2(2 * np.pi / (tol)))), 1)

        self.precalcs = [('pivot', self._calc_pivot),
                         ('norms', self._calc_norms),
                         ('scale', self._calc_scale)]

    def get_rmsd(self, a, b, precalced):

        p, q = self._get_points(a, b, precalced)

        diagonal = precalced[a]['norms'] ** 2 + np.expand_dims(precalced[b]['norms'] ** 2, 1)
        off_diagonal = precalced[a]['norms'] * np.expand_dims(precalced[b]['norms'], 1)

        # positive clockwise angles
        outer = np.dot(q, p.T)
        cross = np.cross(np.expand_dims(q, axis=1), np.expand_dims(p, axis=0))
        angles = (np.arctan2(cross, outer) + 2 * np.pi) % (2 * np.pi)

        node0 = Node(0, (0, 2 * np.pi))
        node0.eval_bound(angles, diagonal, off_diagonal)

        heap = [node0]
        while len(heap) > 0:

            node = heapq.heappop(heap)

            if node.level == self._max_level:
                break
            elif node.level < self._max_level:
                children = node.generate_children(2)
            else:
                continue

            for child in children:
                child.eval_bound(angles, diagonal, off_diagonal)

                heapq.heappush(heap, child)

        lower_bound, permutation = node.eval_bound(angles, diagonal,
                                                   off_diagonal, assignment='hungarian')

        rmsd = safe_rmsd(q, p[permutation])

        return rmsd  # , permutation


class Node(object):
    # TODO: Docstring
    def __init__(self, level, limits):
        self._level = level
        self._limits = limits
        self._lower_bound = None

    @property
    def limits(self):
        return self._limits

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def level(self):
        return self._level

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._lower_bound == other._lower_bound
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self._lower_bound < other._lower_bound
        return NotImplemented

    def eval_bound(self, angles, diagonal, off_diagonal,
                   assignment='nearest'):

        inside = np.logical_and(angles > self.limits[0],
                                angles <= self.limits[1])

        max_cos = np.ones_like(angles)
        max_cos[inside == False] = np.cos(angles[inside == False] - self.limits[1])

        distance = diagonal - 2 * off_diagonal * max_cos

        if assignment.lower() == 'nearest':
            col = np.argmin(distance, axis=0)
            self._lower_bound = distance[range(len(col)), col].sum()

        elif assignment.lower() == 'hungarian':
            row, col = linear_sum_assignment(distance)
            self._lower_bound = distance[row, col].sum()

        else:
            raise NotImplementedError

        return self._lower_bound, col

    def generate_children(self, n):

        sweep = (self.limits[1] - self.limits[0]) / n

        children = []
        for i in range(n):
            child = Node(self._level + 1,
                         (self.limits[0] + i * sweep,
                          self.limits[0] + (i + 1) * sweep))

            children += [child]

        return children
