import numbers
from collections import defaultdict
from copy import deepcopy

import numpy as np
import scipy.spatial

import psm.cluster
from psm import plotutils
from psm.geometry import transform
from psm.graph import geometric
from psm.graph.faces import find_faces, find_outer_face, traverse_perimeter, convex_hull, face_adjacency
from psm.graph.graphutils import subgraph, adjacency2edges, edges2adjacency
from psm.match import RMSD
from psm.utils import in_groups, relabel_groups

try:
    from psm.graph.traversal import clockwise_traversal, clockwise_traversal_with_depth
except:
    from psm.graph.traversal_slow import clockwise_traversal, clockwise_traversal_with_depth

from psm.graph.traversal_slow import breadth_first_traversal_with_depth


class Segments(object):
    """Segments object.

    The Segments object represents segments of a larger group of
    points and its assigned geometric graph. 
    
    Parameters:
    ----------
    points : ndarray of floats
        ndarray of shape (n,2) defining the xy-positions: [(x1,y1), (x2,y2), ...].
    indices : list of lists of ints
        Each segment is defined by the ints in a sublist.
    adjacency : list of n sets of ints
        The adjacency of each point is defined by a set of ints.
    """

    def __init__(self, points=None, adjacency=None, indices=None, labels=None):

        if points is None:
            self._points = np.zeros((0, 2))
        else:
            self._points = points

        if indices is None:
            self._indices = []
        else:
            self._indices = indices

        if adjacency is None:
            self._adjacency = [set() for _ in range(len(self._points))]
        else:
            self._adjacency = adjacency

        self.labels = labels

        self._rmsd_calc = None
        self._strain = None

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, values):
        if values is None:
            self._labels = None
        elif len(values) == len(self.points):
            self._labels = np.array(values, dtype=np.int)
        else:
            raise RuntimeError

    @property
    def points(self):
        return self._points

    @property
    def indices(self):
        return self._indices

    @property
    def adjacency(self):
        return self._adjacency

    @property
    def edges(self):
        return adjacency2edges(self._adjacency)

    @property
    def fronts(self):
        """Return the first point of each segment."""

        return np.array([segment.front for segment in self])

    @property
    def centers(self):
        """Return the center of positions of each segment."""

        return np.array([segment.center for segment in self])

    @property
    def rmsd_calc(self):
        return self._rmsd_calc

    def is_faces(self):
        return all([segment.is_face() for segment in self])

    def set_point(self, i, new_position):
        self._points[i] = new_position

    def copy(self):
        """Return a copy."""

        indices = deepcopy(self.indices)
        adjacency = deepcopy(self.adjacency)
        return self.__class__(self.points, indices, adjacency)

    def extend(self, other):

        n = len(self.points)

        self._points = np.vstack((self.points, other.points))

        self._indices += [[n + i for i in segment] for segment in other.indices]

        self._adjacency += [set(n + i for i in adjacent) for adjacent in other.adjacency]

    def add_segments(self, other):
        """Extend structures object by appending indices from *other*."""

        if not np.all(self._points == other._points):
            raise RuntimeError()

        if not self._adjacency == other._adjacency:
            raise RuntimeError()

        self._indices += other._indices

        return self

    def __add__(self, other):
        structures = self.copy()
        structures += other
        return structures

    __iadd__ = add_segments

    def __len__(self):
        return len(self.indices)

    def __delitem__(self, i):

        if isinstance(i, numbers.Integral):
            del self._indices[i]
        else:
            for j in sorted(i, reverse=True):
                del self._indices[j]

    def remove_nodes(self, nodes):
        self._adjacency = [adjacency - nodes if (i not in nodes) else set() for i, adjacency in
                           enumerate(self.adjacency)]

    def __getitem__(self, i):

        """Return a subset of the structures.

        i -- scalar integer, list of integers, or slice object
        describing which atoms to return.

        If i is a scalar, return a Structure object. If i is a list or a
        slice, return a Structures object with the same associated info 
        as the original Structures object.
        """

        if isinstance(i, numbers.Integral):
            if i < -len(self) or i >= len(self):
                raise IndexError('Index out of range.')

            return Segment(segments=self, index=i)

        if isinstance(i, slice):
            indices = self.indices[i]
        else:
            indices = [self.indices[j] for j in i]

        return self.__class__(self.points, indices, self.adjacency, self.labels)

    def dual(self, label_size=False):

        if not self.is_faces():
            raise RuntimeError()

        faces = self.faces(set_indices=False)

        points = np.zeros((len(faces), 2))
        for i, face in enumerate(faces):
            points[i] = np.mean(self.points[face], axis=0)

        adjacency = face_adjacency(faces)

        if label_size:
            labels = [len(face) for face in faces]
            return self.__class__(points, adjacency=adjacency, labels=labels)

        else:
            return self.__class__(points, adjacency=adjacency)

    def sample(self, n):
        """ Return a random subsample.

        Parameters:
        ----------
        fraction: float
            The fraction of the structures to return.
        """

        if not isinstance(n, numbers.Integral):
            n = int(len(self) * n)

        if n == 1:
            return self.copy()
        else:
            indices = np.random.choice(len(self), n, replace=False)
            return self[indices]

    def build_graph(self, min_alpha=0, n_neighbors=None, max_aspect=None, hull=False):

        self._threshold_graph(min_alpha=min_alpha, n_neighbors=n_neighbors, max_aspect=max_aspect, hull=hull)

    def _threshold_graph(self, edges=None, min_alpha=0, n_neighbors=None, max_aspect=np.inf, hull=False):

        delaunay = scipy.spatial.Delaunay(self.points)

        simplices = delaunay.simplices

        if max_aspect is not None:
            ar = np.array([_aspect_ratio(*self.points[s]) for s in simplices])
            simplices = simplices[ar < max_aspect]

        alphas = geometric.delaunay_edge_stability(self.points, simplices)

        if n_neighbors is not None:
            adjacency = edges2adjacency(alphas.keys(), len(self.points))

            faces = find_faces(self.points, adjacency, remove_outer_face=False)

            outer_face = faces[find_outer_face(self.points, adjacency, faces)]

            min_alpha = geometric.estimate_min_alpha(alphas, n_neighbors, exclude=outer_face)

        if edges is None:
            edges = alphas.keys()

        edges = [edge for edge in edges if alphas[frozenset(edge)] >= min_alpha]

        if hull:
            edges += [list(edge) for edge in delaunay.convex_hull]

        self._adjacency = edges2adjacency(edges, len(self.points))

        return alphas

    def threshold_graph(self, min_alpha=0, n_neighbors=None):

        edges = adjacency2edges(self.adjacency)

        self._threshold_graph(edges=edges, min_alpha=min_alpha, n_neighbors=n_neighbors)

    def traversals(self, max_depth, set_indices=True, origins=None):

        if origins is None:
            origins = range(len(self.points))

        if max_depth == 1:
            traversals = [[origin] for origin in origins]
        else:
            traversals = []
            for origin in origins:
                traversals.append(breadth_first_traversal_with_depth(origin, self.adjacency, max_depth))

        if set_indices:
            self._indices = traversals

        return traversals

    def faces(self, set_indices=True, remove_outer_face=True, remove_hull=False):

        faces = find_faces(self.points, self.adjacency, remove_outer_face, remove_hull)

        if set_indices:
            self._indices = faces

        return faces

    def outer_face(self):

        faces = self.faces(set_indices=False, remove_outer_face=False)

        return faces[find_outer_face(self.points, self.adjacency, faces)]

    def trim(self, min_degree=None):

        new_indices = []

        if min_degree is not None:
            for segment in self:
                degrees = segment.degrees
                new_indices.append([j for i, j in enumerate(segment.indices) if degrees[i] >= min_degree])

        self._indices = new_indices

    def complete_faces(self):

        faces = self.faces(set_indices=False)

        face_references = defaultdict(list)
        for i, face in enumerate(faces):
            for j in face:
                face_references[j].append(i)

        new_indices = []

        for segment in self:
            segment_faces = set(
                l for k in set(j for i in segment.indices for j in face_references[i]) for l in faces[k])
            segment_faces = list(segment_faces - set(segment.indices))
            new_indices.append(segment.indices + segment_faces)

        self._indices = new_indices

    def prinpical_structure(self, **kwargs):
        clusterer = psm.cluster.Cluster(**kwargs)
        clusterer.fit(self)
        return clusterer.principal_structures(1)

    def show_segments(self, n=1, axes=None, nrows=1, show_order=False, sample=False, **kwargs):

        if sample:
            segments = self.sample(n)
        else:
            segments = self

        return plotutils.show_segments(segments, n=n, axes=axes, nrows=nrows, show_order=show_order, **kwargs)

    def show_graph(self, ax=None, c='k', color_mode='edges', **kwargs):
        plotutils.edge_plot(self, ax=ax, c=c, color_mode=color_mode, **kwargs)

    def show_nodes(self, ax=None):

        if self.labels is not None:
            c = plotutils._get_colors_array(self.labels, cmap='tab10', vmin=0, vmax=10)

        else:
            c = 'k'

        plotutils.scatter_plot(self.points, ax=ax, c=c)

    def register(self, other, rmsd_calc=None, progress_bar=True, **kwargs):
        if rmsd_calc is None:
            self._rmsd_calc = RMSD(**kwargs)
        else:
            self._rmsd_calc = rmsd_calc

        return self._rmsd_calc.register(other, self, progress_bar=progress_bar)

    def best_matches(self):
        return self.rmsd_calc.best_matches(self)

    def calc_strain(self, adjust_zero=None, match='best', rmsd_max=np.inf, return_affine=False):
        if return_affine:
            return self.rmsd_calc.calc_strain(self, match=match, rmsd_max=rmsd_max, return_affine=True)
        else:
            strain, rotation = self.rmsd_calc.calc_strain(self, match=match, rmsd_max=rmsd_max, return_affine=False)

            if adjust_zero is 'median':
                strain = transform.zero_median(strain)
            elif adjust_zero is 'mean':
                raise NotImplementedError()
            elif adjust_zero is not None:
                raise ValueError()

            return strain, rotation


class Segment(object):

    def __init__(self, segments, index):
        self._index = index
        self._segments = segments

    def __len__(self):
        return len(self.indices)

    @property
    def index(self):
        return self._index

    @property
    def segments(self):
        return self._segments

    @property
    def points(self):
        return self.segments.points[self.indices]

    @property
    def labels(self):
        if self.segments.labels is None:
            return None
        else:
            return self.segments.labels[self.indices]

    @property
    def indices(self):
        return self.segments.indices[self.index]

    @indices.setter
    def indices(self, new_indices):
        self.segments.indices[self.index] = new_indices

    @property
    def adjacency(self):
        return subgraph(self.segments.adjacency, self.indices)

    @property
    def degrees(self):
        return [len(adjacent) for adjacent in self.adjacency]

    @property
    def edges(self):
        return adjacency2edges(self.adjacency)

    @property
    def front(self):
        return self.points[0]

    @property
    def center(self):
        return np.mean(self.points, axis=0)

    def is_face(self):
        return all([len(adjacent) == 2 for adjacent in self.adjacency])

    def perimeter(self):
        return traverse_perimeter(self.points, self.adjacency)

    def hull(self):
        return convex_hull(self.points, self.adjacency)

    def show(self, ax=None, c='k', show_order=False):
        plotutils.show_segment(self, ax=ax, c=c, show_order=show_order)

    def set_point(self, i, new_position):
        i = self.indices[i]
        self.segments.points[i] = new_position

    def add_points(self, new_points):
        self.segments._indices[self.index] = self.segments.indices[self.index] + [len(self.segments.points) + i for i
                                                                                  in range(len(new_points))]
        self.segments._points = np.vstack((self.segments.points, new_points))

    def detach(self, segment_indices=None):
        if segment_indices is None:
            indices = [list(range(len(self)))]
            return Segments(self.points, indices, self.adjacency)
        else:
            points = self.points[segment_indices]
            indices = [list(range(len(segment_indices)))]
            adjacency = subgraph(self.adjacency, segment_indices)
            return Segments(points, indices, adjacency)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return (self.segments == other.segments) & (self.index == other.index)

    def __hash__(self):
        return hash((self.segments, self.index))
