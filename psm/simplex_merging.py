from collections import defaultdict

import numpy as np
import scipy.spatial
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def _calc_circumcentre(p1, p2, p3):

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    a = np.linalg.det([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]])
    bx = -np.linalg.det([[x1**2 + y1**2, y1, 1],
                         [x2**2 + y2**2, y2, 1],
                         [x3**2 + y3**2, y3, 1]])

    by = np.linalg.det([[x1**2 + y1**2, x1, 1],
                        [x2**2 + y2**2, x2, 1],
                        [x3**2 + y3**2, x3, 1]])
    x0 = -bx / (2 * a)
    y0 = -by / (2 * a)
    return np.array([x0, y0])

def _simplex_edges(simplices):

    edges = []
    for s in simplices:
        for i in range(len(s)):
            a = s[i]
            b = s[(i + 1) % 3]
            edges += [(a, b)]
    return np.array(edges)

def _order_exterior_vertices(simplices):

    edges = _simplex_edges(simplices).tolist()
    
    tally = defaultdict(list)
    for i,item in enumerate(edges):
        tally[tuple(sorted(item))].append(i)
        
    edges = {edges[locs[0]][0]:edges[locs[0]][1] for locs 
                        in tally.values() if len(locs)==1}
    
    order = [list(edges.keys())[0]]
    while len(order) < len(edges):
        order += [edges[order[-1]]]

    return order
    
def simplex_merging(points, k=.5, min_samples=2, return_segments=False):
    
    simplices = scipy.spatial.Delaunay(points).simplices
    vertices = np.array([_calc_circumcentre(*points[s]) for s in simplices])
    
    nbrs = NearestNeighbors(n_neighbors = 2).fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    eps = k * np.median(distances[:,1])
    estimator = DBSCAN(eps=eps, min_samples=min_samples)

    labels = estimator.fit_predict(vertices)
    
    segments = []
    for i in range(labels.max()+1):
        v = np.where(labels == i)[0]
        order = _order_exterior_vertices(simplices[v])
        segments += [order]    
    
    adjacency = [set() for i in range(len(points))]
    for segment in segments:
        for i,j in enumerate(segment):
            adjacency[j].add(segment[i-1])
    
    if return_segments:
        return adjacency, segments
    else:
        return adjacency