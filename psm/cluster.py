import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import DBSCAN

import psm.segments
from psm.match import RMSD
from psm.graph.graphutils import subgraph


def _relabel(labels):
    outliers = labels == -1
    unique, counts = np.unique(labels[outliers == False], return_counts=True)

    relabeled = -np.ones_like(labels)
    for i, label in enumerate(unique[np.argsort(-counts)]):
        relabeled[labels == label] = i

    return relabeled


def _rigid_rotation(src, dst):
    A = np.dot(dst.T, src)

    V, S, W = np.linalg.svd(A)
    U = np.dot(V, W)

    return np.dot(src, U.T)


class Cluster(object):

    # TODO : Write test

    def __init__(self, rmsd_calc=None, clusterer=None, **kwargs):

        if rmsd_calc is None:
            self.rmsd_calc = RMSD(transform='rigid')
        else:
            self.rmsd_calc = rmsd_calc

        if clusterer is None:
            self.clusterer = DBSCAN(**kwargs)
        else:
            self.clusterer = clusterer

        self._structures = None

    def _cluster(self, rmsd):

        if np.isnan(rmsd).any():
            raise RuntimeError()

        return self.clusterer.fit_predict(rmsd)

    def fit(self, structures, progress_bar=True):

        self._structures = structures

        rmsd = self.rmsd_calc.register(structures, structures, progress_bar=progress_bar)

        pieces = connected_components(np.isnan(rmsd) == False)[1]

        self._labels = -np.ones_like(pieces)
        for piece in np.unique(pieces):
            cols = np.where(pieces == piece)[0]

            connected_rmsd = rmsd[cols[:, None], cols]

            new_labels = self._cluster(connected_rmsd)
            inliers = new_labels > -1

            if inliers.any():
                new_labels[inliers] = new_labels[inliers] + np.max(self._labels) + 1
                self._labels[cols] = new_labels

        self._labels = _relabel(self._labels)

        return self._labels

    def principal_structures(self, n, tol=1e-12):

        unique = np.unique(self._labels[self._labels != -1])

        segments = psm.segments.Segments()
        for label in unique[:n]:

            cluster = np.where(label == self._labels)[0]

            permutation = self.rmsd_calc._permutations[frozenset((cluster[0], cluster[1]))]

            dst = self._structures[cluster[0]].points[permutation] - self._structures[cluster[0]].center

            while 1:
                rotated = dst.copy()
                for i in cluster[1:]:
                    permutation = self.rmsd_calc._permutations[frozenset((cluster[0], i))]
                    src = self._structures[i].points[permutation] - self._structures[i].center
                    rotated += _rigid_rotation(src, dst)

                rotated = rotated / len(cluster)
                dst = rotated

                if np.sum((rotated - dst) ** 2) / len(rotated) < tol:
                    break

            adjacency = subgraph(self._structures[cluster[-1]].adjacency, list(permutation))
            segments.extend(psm.segments.Segments(rotated, [range(len(permutation))], adjacency))

        return segments
