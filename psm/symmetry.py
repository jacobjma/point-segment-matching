import numpy as np

def _group_duplicates(array, tol=1e-12):
    edges = np.diff(array) > tol
    labels = np.hstack((0, np.cumsum(edges)))
    return [np.where(labels==i)[0] for i in range(labels.max()+1)]

def _group_identical(array, tol=1e-12):
    order = np.argsort(array)    
    edges = np.diff(array[order]) > tol
    labels = np.hstack((0,np.cumsum(edges)))
    labels = labels[np.argsort(order)]
    return [np.where(labels==i)[0] for i in range(labels.max()+1)]
    
def _angular_symmetries(angles, tol=1e-12):
    
    angles = np.sort(angles)
    angles = np.hstack((angles, angles[0]+2*np.pi))

    diffs = np.abs(np.diff(angles))
    diffs = np.hstack((diffs, diffs))
    
    symmetries = []
    for i in range(1,len(angles)-1):
        if np.all(np.abs(diffs[:len(angles)] - 
                 np.roll(diffs,i)[:len(angles)]) < tol):
            symmetries += [np.sum(diffs[:i])]
    
    return np.array(symmetries)
    
def point_symmetries(points, tol=1e-12):
    
    norms = np.linalg.norm(points, axis=1)
    points = points[norms>tol]
    norms = norms[norms>tol]

    angles = np.arctan2(points[:,0], points[:,1])

    order = np.argsort(angles)
    angles = (angles[order] + 2*np.pi) % (2*np.pi)
    
    shells = _group_identical(norms[order], tol=tol)
    
    symmetries = []
    for shell in shells:
        symmetries.append(_angular_symmetries(angles[shell]))

    for i in range(0,len(symmetries)):
        symmetries[0] = symmetries[0][(np.abs(symmetries[i][:,None] 
                                            - symmetries[0]) < tol).any(0)]
    
    return len(symmetries[0]) + 1
    