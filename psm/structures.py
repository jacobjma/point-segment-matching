import numpy as np
import numbers
from ssm import graphs
from copy import deepcopy

def segment_centers(points, segments):
    return np.array([np.mean(points[segment],axis=0) for segment in segments])
    
class Structures(object):
    
    """Structures object.

    The Structures object represents segments of a larger group of 
    points and its assigned geometric graph. 
    
    Parameters:
    ----------
    points: ndarray of floats
        ndarray of shape (n,2) defining the xy-positions: [(x1,y1), (x2,y2), ...].
    segments: list of lists of ints
        Each segment is defined by the ints in a sublist.
    adjacency: list of n sets of ints
        The adjacency of each point is defined by a set of ints.
    """
    
    def __init__(self, points=None, segments=None, adjacency=None):
        
        if points is None:
            self._points = np.zeros((0,2))
        else:
            self._points = points
        
        if segments is None:
            self._segments = []
        else:
            self._segments = segments
        
        if adjacency is None:
            self._adjacency = []
        else:
            self._adjacency = adjacency
    
    @property
    def points(self):
        return self._points
    
    @property
    def segments(self):
        return self._segments
    
    @property
    def adjacency(self):
        return self._adjacency

    @property
    def fronts(self):
        """Return the first point of each segment."""
        
        return np.array([self.points[segment[0]] for segment in self.segments])

    @property
    def centers(self):
        """Return the center of positions of each segment."""
    
        return segment_centers(self.points, self.segments)
            
    def copy(self):
        """Return a copy."""
        
        segments = deepcopy(self.segments)
        adjacency = deepcopy(self.adjacency)
        return self.__class__(self.points, segments, adjacency)
    
    def extend(self, other):
        
        n = len(self.points)
        
        self._points = np.vstack((self.points, other.points))
        
        self._segments += [[n+i for i in segment] for segment in other.segments]
        
        self._adjacency += [set(n+i for i in adjacent) for adjacent in other.adjacency]
    
    def add_segments(self, other):
        """Extend structures object by appending segments from *other*."""
        
        if not np.all(self._points == other._points):
            raise RuntimeError()
        
        if not self._adjacency == other._adjacency:
            raise RuntimeError()
        
        self._segments += other._segments
        
        return self
    
    def __add__(self, other):
        structures = self.copy()
        structures += other
        return structures
    
    __iadd__ = add_segments
    
    def __len__(self):
        return len(self.segments)
    
    def __delitem__(self, i):
        
        if isinstance(i, numbers.Integral):
            del self._segments[i]
        else:
            for j in sorted(i, reverse=True):
                del self._segments[j]
    
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
        
            return Structure(structures=self, index=i)
            
        if isinstance(i, slice):
            segments = self.segments[i]
        else:
            segments = [self.segments[j] for j in i]
        
        return self.__class__(self.points, segments, self.adjacency)
    
    def sample(self, fraction):
        """ Return a random subsample.
        
        Parameters:
        ----------
        fraction: float
            The fraction of the structures to return.
        """
        
        n = int(len(self) * fraction)
        indices = np.random.choice(len(self), n, replace=False)
        return self[indices]

class Structure(object):
    
    def __init__(self, structures, index):
        
        self.__dict__['_index'] = index
        self.__dict__['_structures'] = structures
    
    def __len__(self):
        return len(self.segment)
    
    @property
    def index(self):
        return self._index
    
    @property
    def structures(self):
        return self._structures
    
    @property
    def points(self):
        return self.structures.points[self.segment]
    
    @property
    def segment(self):
        return self.structures.segments[self.index]
    
    @property
    def adjacency(self):
        return graphs.subgraph(self.structures.adjacency, self.segment)
        
    @property
    def front(self):
        return self.points[0]
    
    @property
    def center(self):
        return np.mean(self.points, axis=0)
    
    def detach(self):
        
        segments = [list(range(len(self)))]
        
        return Structures(self.points, segments, self.adjacency)
    
    def __eq__(self, other):
        
        if not isinstance(other, self.__class__):
            return False
        
        return (self.structures == other.structures)&(self.index == other.index)
    
    def __hash__(self):
        return hash((self.structures, self.index))