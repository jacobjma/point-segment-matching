import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def rmsd_qcp(np.ndarray[DTYPE_t, ndim=2] A, np.ndarray[DTYPE_t, ndim=2] B):
    
    cdef int n_points = A.shape[0]
    cdef float G_A = 0
    cdef float G_B = 0
    cdef float Sxx = 0
    cdef float Syy = 0
    cdef float Syx = 0
    cdef float Sxy = 0
    cdef float Sxx2, Syy2, Sxy2, Syx2, C0, C2, E0
    cdef float SxypSyx, SxymSyx, SxxpSyy, SxxmSyy
    cdef float Syy2mSxx2, Sxy2mSyx2, max_eigenvalue, rmsd
    
    for i in xrange(0,n_points):
        G_A += A[i,0]**2 + A[i,1]**2
        G_B += B[i,0]**2 + B[i,1]**2
        Sxx += A[i,0]*B[i,0]
        Syy += A[i,1]*B[i,1]
        Sxy += A[i,1]*B[i,0]
        Syx += A[i,0]*B[i,1]

    Sxx2 = Sxx * Sxx
    Syy2 = Syy * Syy
    Sxy2 = Sxy * Sxy
    Syx2 = Syx * Syx

    C2 = - 2.0 * (Sxx2 + Syy2 + Sxy2 + Syx2)
    
    SxypSyx = Sxy + Syx
    SxymSyx = Sxy - Syx
    SxxpSyy = Sxx + Syy
    SxxmSyy = Sxx - Syy
    Syy2mSxx2 = Syy2 - Sxx2
    Sxy2mSyx2 = Sxy2 - Syx2
    
    C0 = Sxy2mSyx2 * Sxy2mSyx2 + Syy2mSxx2 * Syy2mSxx2 \
        + (SxymSyx * SxxmSyy * SxymSyx * SxxmSyy) \
        + (SxypSyx * SxxpSyy * SxypSyx * SxxpSyy)
    
    E0 = (G_A + G_B) / 2.0
    
    C3 = max(C2**2-4*C0, 0.0)
    
    max_eigenvalue = np.sqrt(np.sqrt(C3)-C2)/np.sqrt(2)
    rmsd = np.sqrt(np.abs(2.0 * (E0 - max_eigenvalue) / n_points))
    
    return rmsd