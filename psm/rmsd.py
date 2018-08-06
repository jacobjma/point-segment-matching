import numpy as np
#from ssm.qcp import rmsd_qcp

def rmsd_kabsch(src, dst):
    
    A = np.dot(dst.T, src)
    
    V, S, W = np.linalg.svd(A)
    
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    
    U = np.dot(V, W)
    
    rotated = np.dot(src, U.T)
    
    return np.sqrt(np.sum((rotated - dst)**2)/len(src))

def rmsd_qcp(src, dst):
    
    M = np.dot(dst.T, src)
    
    xx, xy = M[0, :]
    yx, yy = M[1, :]
    
    xx_yy = xx + yy
    xy_yx = xy - yx
    xy_yx_2 = xy_yx**2
    
    u = np.sqrt(xy_yx_2 + xx_yy**2)
    xx_yy_u = xx_yy + u
    xx_yy_u_2 = xx_yy_u**2
    
    denom = xx_yy_u_2 + xy_yx_2
    
    if denom == 0.:
        return np.nan
    
    Uxx = (xx_yy_u_2 - xy_yx_2) / denom
    Uxy = 2 * xy_yx * xx_yy_u / denom
    
    U = np.array([[Uxx, -Uxy], [Uxy, Uxx]])
    
    rmsd = np.sqrt(np.sum((np.dot(src, U) - dst)**2) / len(src))
    
    return rmsd

def safe_rmsd(src, dst):
    
    rmsd = rmsd_qcp(src, dst)
    
    if np.isnan(rmsd):
        return rmsd_kabsch(src, dst)
    else:
        return rmsd
    
    
def _rmsd_qcp(src, dst):
    
    M = np.dot(dst.T, src)
    
    xx, xy = M[0, :]
    yx, yy = M[1, :]
    
    xx_yy = xx + yy
    xy_yx = xy - yx
    xy_yx_2 = xy_yx**2
    
    u = np.sqrt(xy_yx_2 + xx_yy**2)
    xx_yy_u = xx_yy + u
    xx_yy_u_2 = xx_yy_u**2
    
    denom = xx_yy_u_2 + xy_yx_2
    
    Uxx = (xx_yy_u_2 - xy_yx_2) / denom
    Uxy = 2 * xy_yx * xx_yy_u / denom
    
    U = np.array([[Uxx, -Uxy], [Uxy, Uxx]])
    
    rmsd = np.sqrt(np.sum((np.dot(src, U) - dst)**2) / len(src))
    
    return rmsd