import numpy as np
from ssm.graphs import select_segments
from scipy.optimize import least_squares
from ssm import graphs, peaks
from ssm.structures import segment_centers
from ssm import Structures
from scipy.ndimage import gaussian_laplace
from ssm.fitting import Polynomial2D
from copy import copy
from tqdm import tqdm_notebook as tqdm

'''
todo: 
- check for bravais
- cleanup
- docstrings
'''

def select_border(root_face, tail_face):
    
    border = set(root_face).intersection(set(tail_face))
    
    indices = sorted([root_face.index(i) for i in border])
    
    if len(border.intersection(set((root_face[0], root_face[-1])))) == 2:
        tail = root_face[indices[1]]
        head = root_face[indices[0]]
    else:
        tail = root_face[indices[0]]
        head = root_face[indices[1]]
        
    return (tail, head)

def miller_index_traversal(edge, faces, adjacency, clockwise):

    hk = {edge[0]:[0,0], edge[1]:[1,0]}
    queue = [edge]
    
    while queue:
        edge = queue.pop(0)
        old_edge = edge
        
        root_face = faces[old_edge[0]]
        tail_face = faces[old_edge[1]]
        border = select_border(root_face, tail_face)
        
        for i in range(len(adjacency[edge[0]])):
            edge = clockwise[edge]

            if not edge[1] in hk.keys():

                if hk[old_edge[0]][0] == hk[old_edge[1]][0]:
                    index_of_change = 1
                else:
                    index_of_change = 0

                sign_of_change = -(hk[old_edge[0]][index_of_change] - hk[old_edge[1]][index_of_change])

                hk[edge[1]] = hk[old_edge[0]].copy()

                if border[0] in faces[edge[1]]:
                    hk[edge[1]][1-index_of_change] += sign_of_change * (2*(index_of_change)-1)

                elif border[1] in faces[edge[1]]:
                    hk[edge[1]][1-index_of_change] -= sign_of_change * (2*(index_of_change)-1)

                else:
                    hk[edge[1]][index_of_change] -= sign_of_change

                queue.append((edge[1], edge[0]))
                
    return hk

def assign_miller_to_faces(points, faces):

    adjacency = graphs.face_adjacency(faces)
    
    centers = segment_centers(points, faces)
    
    clockwise = graphs.find_clockwise(centers, adjacency)
    
    root = 0
    edge = (root, next(iter(adjacency[root])))
    
    miller = miller_index_traversal(edge, faces, adjacency, clockwise)

    return miller
    
def assign_miller_to_corners(points, faces):
    
    adjacency = graphs.face_adjacency(faces)
    
    face_id = 0
    face = faces[face_id]
    miller = {i:hk for i,hk in zip(face, [[0,0],[1,0],[1,1],[0,1]])}
    marked = set((face_id,))
    queue = [(i, face_id) for i in adjacency[face_id]]

    while queue:
        face_id, old_face_id = queue.pop()
        
        if face_id in marked:
            continue
        
        marked.add(face_id)
        queue += [(i, face_id) for i in adjacency[face_id]]
        
        old_face = faces[old_face_id]
        face = faces[face_id]
        border = select_border(old_face, face)
        
        i = (faces[face_id].index(border[-1]) - 1)
        for l in range(len(face)-2):
            i = i - l
            j = (i + 1) % len(face)
            k = (i + 2) % len(face)

            if miller[face[j]][0] == miller[face[k]][0]:
                index_of_change = 0
            else:
                index_of_change = 1

            sign_of_change = miller[face[k]][1-index_of_change]-miller[face[j]][1-index_of_change]

            miller[face[i]] = miller[face[j]].copy()
            miller[face[i]][index_of_change] += sign_of_change * (2*(index_of_change)-1)
    
    return miller

    
def _build_residual(miller):
    def residual(p):
        ax,ay,bx,by,x0,y0 = p
        r = 0
        for (h,k),(x,y) in miller.items():
            r += (h*ax+k*bx+x0-x)**2 + (h*ay+k*by+y0-y)**2
        return r
    return residual    

def fit_lattice_parameters(miller):
    
    ax = miller[(0,0)][0] - miller[(1,0)][0]
    ay = miller[(0,0)][1] - miller[(1,0)][1]
    bx = ay
    by = ax
    x0 = miller[(0,0)][0]
    y0 = miller[(0,0)][1]
    
    p =(ax, ay, bx, by, x0, y0)
    
    residual = _build_residual(miller)
    
    result = least_squares(residual, p)
    
    a = result.x[:2]
    b = result.x[2:4]
    x0 = result.x[4:]
    
    return a, b, x0
    
class CrystalDetector(object):
    
    def __init__(self, rmsd_calc, rmsd_max, graph_func=None):
        
        self.rmsd_calc = rmsd_calc
        
        self.rmsd_max = rmsd_max
        
        if graph_func is None:
            self.graph_func = graphs.urquhart
        else:
            self.graph_func = graph_func
        
        self._points = None
        self._faces = None
        self._adjacency = None
        
        self._indices = []
        self._crystal_rmsd = None
        self._principal_structure = None
        self._pieces = []
    
    def detect(self, points):
        
        self._points = points
        
        self._adjacency = self.graph_func(points)
        
        self._faces = graphs.find_faces(points, self._adjacency)
        
        structures = Structures(self._points, self._faces, self._adjacency)
        
        rmsd = self.rmsd_calc.register(structures, structures, progress_bar=False)
        
        if len(rmsd) > 0:
            principal_structure = self.rmsd_calc.principal_structures(1, self.rmsd_max)
            rmsd = self.rmsd_calc.register(principal_structure, structures, progress_bar=False)
        else:
            return None
        
        if len(rmsd) > 0:
            self._rmsd = rmsd[0]
        
            mask = ~np.isnan(self._rmsd)
            
            mask[mask] &= self._rmsd[mask] < self.rmsd_max
            self._rmsd[mask] = np.nan
            
            self._indices = np.where(mask)[0]
            
            points, faces = graphs.select_segments(self._indices, self._points, self._faces)
            
            self._pieces = graphs.adjacent_faces(points, faces)

            return self._indices
        else:
            return None
    
    def crystal_points(self):
        return self._points[np.unique([self._faces[i] for i in self._indices])]
    
    def average_crystal_rmsd(self):
        return np.sum(self._rmsd[self._indices]) / len(self._indices)
        
    def num_crystal_faces(self):
        return len(self._indices)
    
    def get_crystal(self):
        points, faces, adjacency = graphs.select_segments(self._indices, self._points, self._faces, self._adjacency)
        return Structures(points, faces, adjacency)
    
    def get_crystal_piece(self, i):
        points, faces, adjacency = graphs.select_segments(self._pieces[i], self._points, self._faces, self._adjacency)
        return Structures(points, faces, adjacency)
    
    def get_miller_indices(self, sites='faces'):
        
        structures = self.get_crystal_piece(0)
        
        if sites == 'faces':
            return assign_miller_to_faces(structures.points, structures.segments)
        elif sites == 'corners':
            return assign_miller_to_corners(structures.points, structures.segments)
        else:
            raise ValueError()
    
    def get_lattice_parameters(self, sites='faces'):
        
        structures = self.get_crystal_piece(0)
        
        miller = self.get_miller_indices()
        
        miller = {(h,k) : structures.centers[i] for i,(h,k) in miller.items()}
        
        a,b,x0 = fit_lattice_parameters(miller)
        
        return a,b,x0

def single_crystal_sweep(image, detector, num_peaks, min_sigma=1, step_size=1, max_sigma=None, progress_bar=False):
    
    if max_sigma is None:
        max_sigma = np.max(image.shape)//2
    
    sigmas = np.arange(min_sigma,max_sigma,step_size)
    
    detectors = [copy(detector) for _ in sigmas]
    for sigma,detector in tqdm(zip(sigmas, detectors), total=len(detectors), disable=not progress_bar):
        
        gl = -gaussian_laplace(image, sigma)
        points = peaks.find_local_peaks(gl, min_distance=10, exclude_border=1)#.astype(float)
        
        gl_val = gl[points[:,0], points[:,1]]
        points = points[np.argsort(-gl_val)][:num_peaks]
        points = peaks.refine_peaks(gl, points, model=Polynomial2D(), extent=3)
        
        indices = detector.detect(points)
    
    return detectors