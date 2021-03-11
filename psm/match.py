from collections import deque, defaultdict

import numpy as np
import scipy

from psm.geometry.rmsd import safe_rmsd
from psm.graph.graphutils import find_clockwise
from psm.utils import ProgressBar


def clockwise_traversal(edge, allowed, clockwise, degree):
    queue = deque([edge])
    traversal = [edge[0]]

    while queue:
        edge = queue.popleft()

        for k in range(degree[edge[0]]):
            edge = clockwise[edge]
            if edge[1] in allowed:
                queue.append((edge[1], edge[0]))
                traversal.append(edge[1])
                allowed.remove(edge[1])

    return traversal


def order_segments(segments, in_place=True):
    if not in_place:
        segments = segments.copy()

    clockwise = find_clockwise(segments.points, segments.adjacency)
    degree = [len(adjacent) for adjacent in segments.adjacency]

    for segment in segments:
        segment.indices = clockwise_traversal((segment.indices[0], next(iter(segments.adjacency[segment.indices[0]]))),
                                              set(segment.indices[1:]),
                                              clockwise, degree)

    return segments


def permutation_generator(segment, clockwise, degree):
    edge = (segment.indices[0], segment.indices[1])
    allowed = set(segment.indices[1:])
    for _ in range(degree[edge[0]]):
        edge = clockwise[edge]
        if edge[1] in allowed:
            traversal = clockwise_traversal(edge, allowed.copy(), clockwise, degree)
            yield [segment.indices.index(i) for i in traversal]


def subgraph(adjacency, indices):
    return tuple(frozenset(indices.index(j) for j in adjacency[i] if j in indices) for i in indices)


def find_alternate_orders(segments):
    clockwise = find_clockwise(segments.points, segments.adjacency)
    degree = [len(adjacent) for adjacent in segments.adjacency]

    alternate_orders = defaultdict(list)
    graph_hashes = np.empty(len(segments), dtype=np.int64)
    hash_labels = {}

    k = 0
    for i, segment in enumerate(segments):
        graph_hashes[i] = hash(tuple(segment.adjacency))

        if graph_hashes[i] not in hash_labels.keys():

            g = permutation_generator(segment, clockwise, degree)

            alternate_hashes = []
            permutations = []
            for permutation in g:
                alternate_hashes.append(hash(subgraph(segment.adjacency, permutation)))
                permutations.append(permutation)

            for j, alternate_hash in enumerate(alternate_hashes):
                if alternate_hash not in alternate_orders.keys():
                    reverse_permutation = np.argsort(permutations[j])
                    for permutation in permutations:
                        alternate_orders[alternate_hash].append(reverse_permutation[permutation])

            hash_labels.update(dict.fromkeys(alternate_hashes, k))
            k += 1

    return alternate_orders, graph_hashes, hash_labels


class Matcher(object):

    def __init__(self, segments=None):

        if segments is None:
            self._segments = segments

        else:
            self.segments = segments

    @property
    def segments(self):
        return self._segments

    @segments.setter
    def segments(self, segments):
        self._segments = segments

    def _self_match_generator(self, progress_bar=False):
        raise NotImplementedError

    def _match_generator(self, segments, progress_bar=False):
        raise NotImplementedError

    def match(self, segments, progress_bar=False):

        if self.segments is None:
            raise RuntimeError()

        if segments == self.segments:
            return self._self_match_generator(progress_bar=progress_bar)

        else:
            return self._match_generator(segments, progress_bar=progress_bar)


class FaceMatcher(Matcher):

    def __init__(self, segments=None):

        super(FaceMatcher, self).__init__(segments)

    @property
    def segments(self):
        return self._segments

    @segments.setter
    def segments(self, segments):
        self._segments = segments

    def _self_match_generator(self, progress_bar=False):
        A, B = self.segments, self.segments

        N = len(A)

        bar = ProgressBar(N * (N - 1) // 2, units='registrations', disable=not progress_bar)
        k = 0
        for i, a in enumerate(A):
            linspace = np.linspace(0, len(a) - 1, len(a), dtype=int)
            permutations = [np.roll(linspace, i) for i in range(len(a))]

            for j, b in enumerate(B[:i]):
                bar.print(k)
                k += 1

                if len(a) != len(b):
                    yield i, j, None

                else:
                    yield i, j, permutations

    def _match_generator(self, segments, progress_bar=False):
        A, B = self.segments, segments
        N, M = len(A), len(B)

        # TODO : Check symmetries

        bar = ProgressBar(N * M, units='registrations', disable=not progress_bar)
        for i, a in enumerate(A):
            linspace = np.linspace(0, len(a) - 1, len(a), dtype=int)
            permutations = np.array([np.roll(linspace, i) for i in range(len(a))])

            for j, b in enumerate(B):
                bar.print(i * M + j)

                if len(a) != len(b):
                    yield i, j, None

                else:

                    if a.labels is not None:
                        if b.labels is None:
                            raise RuntimeError()

                        labels = a.labels[:, None][permutations[None, :]][0, ..., 0]

                        matching_labels = np.all(labels == b.labels[None], axis=1)

                        if not np.any(matching_labels):
                            yield i, j, None

                        else:
                            yield i, j, permutations[matching_labels]
                    else:

                        yield i, j, permutations


# def SubgraphMatcher(Matcher):
#     def __init__(self, segments=None):
#         super(RootedMatcher, self).__init__(segments)
#
#     @property
#     def segments(self):
#         return self._segments
#
#     @segments.setter
#     def segments(self, segments):
#         self._segments = order_segments(segments)
#
#     def _match_generator(self, segments, progress_bar):
#         A, B = self.segments, segments
#         N, M = len(A), len(B)


class RootedMatcher(Matcher):

    def __init__(self, segments=None):

        super(RootedMatcher, self).__init__(segments)

    @property
    def segments(self):
        return self._segments

    @segments.setter
    def segments(self, segments):
        self._segments = order_segments(segments)
        self._permutations, self._graph_hashes, self._hash_labels = find_alternate_orders(segments)

    def _self_match_generator(self, progress_bar=False):

        A, B = self.segments, self.segments

        N = len(A)

        bar = ProgressBar(N * (N - 1) // 2, units='registrations', disable=not progress_bar)
        k = 0
        for i, a in enumerate(A):

            permutations = self._permutations[self._graph_hashes[i]]

            a_label = self._hash_labels[self._graph_hashes[i]]

            for j, b in enumerate(B[:i]):
                bar.print(k)
                k += 1

                b_label = self._hash_labels[self._graph_hashes[j]]

                if a_label == b_label:
                    yield i, j, permutations

                else:
                    yield i, j, None

    def _match_generator(self, segments, progress_bar=False):
        A, B = self.segments, segments
        N, M = len(A), len(B)
        B = order_segments(B)

        bar = ProgressBar(N * M, units='registrations', disable=not progress_bar)
        for i, a in enumerate(A):

            permutations = np.array(self._permutations[self._graph_hashes[i]])

            for j, b in enumerate(B):
                bar.print(i * M + j)

                if len(a) != len(b):
                    yield i, j, None

                else:
                    if a.labels is not None:
                        if b.labels is None:
                            raise RuntimeError()

                        labels = a.labels[:, None][permutations[None, :]][0, ..., 0]

                        matching_labels = np.all(labels == b.labels[None], axis=1)

                        if not np.any(matching_labels):
                            yield i, j, None

                        else:
                            yield i, j, permutations[matching_labels]

                    else:
                        yield i, j, permutations


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


class RMSD(object):
    # TODO: Docstring

    def __init__(self, transform='rigid', rotation_invariant=True, scale_invariant=True, pivot='cop', matcher=None):

        if transform not in ['rigid', 'similarity']:
            raise ValueError()

        if pivot not in ['cop', 'front']:
            raise ValueError()

        self.transform = transform
        self.scale_invariant = scale_invariant
        self.rotation_invariant = rotation_invariant
        self.pivot = pivot

        if matcher is None:
            self.matcher = RootedMatcher()

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

    def _calc_scale(self, points, precalced):

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

    def _get_points(self, a, b, precalced, permutation):

        p = a.points[permutation] - precalced[a]['pivot']
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

    def get_rmsd(self, a, b, precalced, permutation):
        p, q = self._get_points(a, b, precalced, permutation)
        rmsd = safe_rmsd(p, q)
        return rmsd

    def register(self, A, B, progress_bar=False):

        N, M = len(A), len(B)

        self._A, self._B = A, B
        self._rmsd = np.empty((N, M))
        self._rmsd[:] = np.nan
        self._permutations = {}

        precalced = self._do_precalc(A, B)

        # if A.is_faces():
        matcher = FaceMatcher(A)
        # else:
        #
        #
        #matcher = RootedMatcher(A)

        for i, j, permutations in matcher.match(B, progress_bar=progress_bar):

            if permutations is not None:
                # assert len(A[i]) == len(B[j])

                rmsds = np.zeros(len(permutations))

                for k, permutation in enumerate(permutations):
                    rmsds[k] = self.get_rmsd(A[i], B[j], precalced, permutation)

                best_match = np.argmin(rmsds)

                self._permutations[frozenset((i, j))] = permutations[best_match]

                self._rmsd[i, j] = rmsds[best_match]

                if A == B:
                    self._rmsd[j, i] = self._rmsd[i, j]

        if A == B:
            np.fill_diagonal(self._rmsd, 0)

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

    def calc_strain(self, structures, match='best', rmsd_max=np.inf, return_affine=False):

        if match == 'best':
            matches, rmsd = self.best_matches(structures)
        elif match == 'widest':
            matches, rmsd = self.widest_matches(structures, rmsd_max)
        else:
            raise ValueError()

        if return_affine:
            affine = np.empty((len(structures), 3, 3))
            affine[:] = np.nan
        else:
            strain = np.empty((len(structures), 2, 2))
            strain[:] = np.nan

            rotation = np.empty(len(structures))
            rotation[:] = np.nan

        axis = self._get_axis(structures)
        other = self._get_other(structures)

        for i, structure in enumerate(structures):

            if rmsd[i] <= rmsd_max:
                try:
                    permutation = self._permutations[frozenset((matches[i], i))]

                    q = structure.points
                    p = other[matches[i]].points

                    if axis:
                        p = p[permutation]
                    else:
                        q = q[permutation]

                    A = _affine_transform(p, q)

                    if return_affine:
                        affine[i] = A
                    else:
                        U, P = scipy.linalg.polar(A[:-1, :-1], side='left')

                        rotation[i] = np.arctan2(U[0, 1], U[0, 0])

                        strain[i] = P - np.identity(2)
                except:
                    pass

        if return_affine:
            return affine
        else:
            return strain, rotation
