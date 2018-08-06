import numpy as np
from scipy.spatial import Delaunay, Voronoi
from collections import defaultdict
import itertools
from tqdm import tqdm_notebook as tqdm
from ssm.simplex_merging import simplex_merging
from scipy.sparse.csgraph import connected_components

def _flatten(lists):
    return [item for sublist in lists for item in sublist]

def _in_segments(segments):
    return sorted(list(set(_flatten(segments))))
    
def _relabel_segments(segments):
    mapping = {j:i for (i,j) in enumerate(_in_segments(segments))}
    return [[mapping[i] for i in segment] for segment in segments]

def select_segments(indices, points, segments, adjacency=None):
    
    indices_in_segments = _in_segments(segments)
    
    segments = [segments[i] for i in indices]
    points = points[_in_segments(segments)]
    segments = _relabel_segments(segments)
    
    if adjacency is None:
        return points, segments
    else:
        adjacency = subgraph(adjacency, indices_in_segments)
        return points, segments, adjacency
    
def adjacent_faces(points, faces):

    adjacency = face_adjacency(faces)
    connected = connected_components(adjacency2matrix(adjacency))[1]
    unique, counts = np.unique(connected, return_counts=True)
    
    indices = []
    for label in unique[np.argsort(counts)]:
        indices.append(np.where(connected == label)[0])
    
    return indices
    
def select_adjacent_faces(points, faces):
    
    adjacency = face_adjacency(faces)
    
    connected = connected_components(adjacency2matrix(adjacency))[1]
    
    unique, counts = np.unique(connected, return_counts=True)

    selections = []
    for label in unique[np.argsort(counts)]:
        indices = np.where(connected == label)[0]
        points, faces = select_segments(indices, points, faces)
        selections.append((points, faces))
    
    return selections
    
def adjacency2edges(adjacency):
    edges = [frozenset(itertools.product([i], vertices)) for i,vertices in enumerate(adjacency)]
    edges = set(_flatten(edges))
    return edges

def edges2adjacency(edges, num_nodes):
    adjacency = [set() for i in range(num_nodes)]
    for edge in edges:
        adjacency[edge[0]].add(edge[1])
        adjacency[edge[1]].add(edge[0])
    return adjacency            

def adjacency2matrix(adjacency):
    m = np.zeros((len(adjacency),)*2,dtype=bool)
    for i,adjacent in enumerate(adjacency):
        for j in adjacent:
            m[i,j] = True
            m[j,i] = True
    return m
    
def subgraph(adjacency, indices):
    """
    
    
    todo:
    - improve speed
    
    """
    indices = list(indices)
    return [set(indices.index(j) for j in adjacency[i] if j in indices) for i in indices]
    
def _ccw(a, b, c):
    return np.cross(b-a, c-a) <= 0
    #return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
    
def gabriel(points):
    """Return the Gabriel Graph as an adjacency list."""
    
    def intersect(a, b):
        return ((_ccw(a[0],b[0],b[1]) != _ccw(a[1],b[0],b[1]))&
                (_ccw(a[0],a[1],b[0]) != _ccw(a[0],a[1],b[1])))

    simplices = Delaunay(points).simplices
    
    edges = _simplex_edges(simplices)
    vor = Voronoi(points)
    
    to_keep = np.zeros(len(edges), dtype=bool)
    for i,edge in enumerate(edges):
        a = vor.regions[vor.point_region[edge[0]]]
        b = vor.regions[vor.point_region[edge[1]]]

        common = np.intersect1d(a, b, assume_unique=True)

        if intersect(vor.vertices[common], points[edge]):
            to_keep[i]=True
    
    return edges2adjacency(edges[to_keep], len(points))

def _simplex_edges(simplices):
    edges = set([(simplex[i-1],simplex[i]) for i in range(3)] for simplex in simplices)
    #edges = np.array(list(itertools.chain.from_iterable(edges)))
    
    return #np.unique(np.sort(edges,axis=1),axis=0)    
    
def urquhart(points, progress_bar=False):
    """Return the Urquhart Graph as an adjacency list."""

    simplices = Delaunay(points).simplices
    edges = [[frozenset((simplex[i-1],simplex[i])) for i in range(3)] for simplex in simplices]
    edges = set(itertools.chain.from_iterable(edges))
    lengths = {edge : np.linalg.norm(np.diff([points[i] for i in edge],axis=0),axis=1)[0] for edge in edges}
    
    to_remove=set()
    for simplex in tqdm(simplices, unit = ' simplices', disable = not progress_bar):
        simplex_edges = [frozenset(tuple(np.take(simplex, [k,k+1], mode='wrap'))) for k in range(3)]
        longest = simplex_edges[np.argmax([lengths[edge] for edge in simplex_edges])]
        
        to_remove.add(longest)
        
    edges = edges-to_remove
    
    edges = np.array([list(edge) for edge in edges])
    
    return edges2adjacency(edges, len(points))
    
def find_clockwise(points, adjacency):
    """Create a dict mapping a directed edge to the next edge 
    adjacent its tail going in the clockwise direction.
    
    todo:
     - Faster method without calculating angles?
    
    """
    
    clockwise = {}
    for i,point in enumerate(points):
        adjacent = list(adjacency[i])
        
        adjacent_points = points[adjacent] - point
        angle = np.arctan2(adjacent_points[:,0], adjacent_points[:,1])
        adjacent = [adjacent[j] for j in np.argsort(angle)]
        
        for j,k in enumerate(adjacent):
            clockwise[(i,adjacent[j-1])] = (i,k)
    
    return clockwise

def face_adjacency(faces):
    adjacency = [set() for i in range(len(faces))]
    for i,face in enumerate(faces):
        for j,other_face in enumerate(faces):
            if len(set(face).intersection(other_face))==2:
                adjacency[i].add(j)
    return adjacency    
    
def _remove_outer_face(points, adjacency, faces):
    """Remove the outer face from a list of the faces of a plane graph."""
    
    leftmost = np.argmin(points[:,0])
    
    slopes = (points[list(adjacency[leftmost])][:,0] - points[leftmost,0])/\
             (points[list(adjacency[leftmost])][:,1] - points[leftmost,1])
    pair = str(list(adjacency[leftmost])[np.argmin(slopes)]) + '|' + str(leftmost)
    
    return [face for face in faces if not pair in '|'.join(map(str,face))]

def find_faces(points, adjacency):
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
    
    return _remove_outer_face(points, adjacency, faces)
    
def clockwise_traversal(edge, adjacency, clockwise):
    
    degree = [len(adjacent) for adjacent in adjacency]
    queue = [edge]
    traversal = [edge[0]]
    
    while queue:
        edge = queue.pop(0)
        
        for i in range(degree[edge[0]]):
            edge = clockwise[edge]
            
            if not edge[1] in traversal:
                queue.append((edge[1], edge[0]))
                traversal.append(edge[1])
    
    return traversal

def clockwise_traversal_w_depth(edge, adjacency, clockwise, max_depth=None):
    
    degree = [len(adjacent) for adjacent in adjacency]
    
    if max_depth is None:
        max_depth = len(degree)
    
    queue = [edge]
    traversal = [edge[0]]
    depth = [0]
    
    while queue:
        edge = queue.pop(0)
        
        for i in range(degree[edge[0]]):
            edge = clockwise[edge]
            
            if not edge[1] in traversal:
                current_depth = depth[traversal.index(edge[0])] + 1
                
                if current_depth < max_depth:
                    depth.append(current_depth)
                else:
                    break
                
                queue.append((edge[1], edge[0]))
                traversal.append(edge[1])
    
    return traversal, depth

def all_traversals(points, adjacency, second_point='random', max_points=None, 
                    permutation=None, max_depth=None, progress_bar=False):
    
    if not second_point in ['random', 'all']:
        raise ValueError()
    
    if (permutation is None)&(max_points is None):
        permutation = range(0, len(adjacency))
    elif permutation is None:
        permutation = range(0, max_points)
    
    clockwise = find_clockwise(points, adjacency)
    import time
    traversals = []
    for i,point in enumerate(tqdm(points, unit=' traversals', miniters=1, mininterval=.01, disable = not progress_bar)):
        for j in adjacency[i]:
            
            edge = (i, j)
            traversal, _ = clockwise_traversal_w_depth(edge, adjacency, clockwise, max_depth=max_depth)
            traversals.append(traversal)
            #[traversal[k] for k in permutation if k < len(traversal)])
            
            if second_point == 'random':
                break
        
    return traversals
    
def check_isomorphism(adjacency, clockwise, other_adjacency, other_clockwise=None):
    
    if not other_clockwise is None:
        edges = (0, next(iter(other_adjacency[0])))
        traversal = clockwise_traversal(edges, other_adjacency, other_clockwise)[0]
        other_adjacency = subgraph(other_adjacency, traversal)
    
    for i in adjacency[0]:
        edge = (0,i)
        traversal = clockwise_traversal(edge, adjacency, clockwise)
        permuted = subgraph(adjacency, traversal)
        
        if permuted == other_adjacency:
            return True
    
    return False