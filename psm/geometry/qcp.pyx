import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs

DTYPE = np.float
ctypedef np.float_t DTYPE_t


def rmsd_qcp(np.ndarray[DTYPE_t, ndim=2] src, np.ndarray[DTYPE_t, ndim=2] dst):

    cdef int n_points = src.shape[0]
    cdef float xx = 0
    cdef float yy = 0
    cdef float yx = 0
    cdef float xy = 0
    cdef float xx_yy, xy_yx, xy_yx_2, u, xx_yy_u, xx_yy_u_2, denom, Uxx, Uxy, rmsd

    for i in xrange(0,n_points):
        xx += src[i,0]*dst[i,0]
        yy += src[i,1]*dst[i,1]
        xy += src[i,1]*dst[i,0]
        yx += src[i,0]*dst[i,1]

    xx_yy = xx + yy
    xy_yx = xy - yx
    xy_yx_2 = xy_yx**2

    u = sqrt(xy_yx_2 + xx_yy**2)
    xx_yy_u = xx_yy + u
    xx_yy_u_2 = xx_yy_u**2

    denom = xx_yy_u_2 + xy_yx_2

    Uxx = (xx_yy_u_2 - xy_yx_2) / denom
    Uxy = 2 * xy_yx * xx_yy_u / denom

    if denom == 0.:
        return np.nan

    rmsd = 0
    for i in xrange(0,n_points):
        rmsd += (Uxx * src[i,0] + Uxy * src[i,1] - dst[i,0])**2 + (Uxx * src[i,1] - Uxy * src[i,0] - dst[i,1])**2

        #rotated[i,0] = Uxx * src[i,0] + Uxy * src[i,1]
        #rotated[i,1] = Uxx * src[i,1] - Uxy * src[i,0]

    return sqrt(rmsd / n_points)
