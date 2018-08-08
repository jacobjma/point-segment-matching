import numbers

import numpy as np


def rotate_points(points, angle, center='cop'):
    """Rotate point positions.

    Parameters:
    angle = None:
        Angle that the points are rotated
    center = (0, 0, 0):
        The center is kept fixed under the rotation. Use 
        'COP' to fix the center of positions."""

    if isinstance(center, str):
        if center.lower() == 'cop':
            center = points.mean(axis=0)
    else:
        center = np.array(center)

    cosa = np.cos(angle)
    sina = np.sin(angle)

    R = np.array([[cosa, sina], [-sina, cosa]])

    return np.dot(points - center, R.T) + center


def rotate_strain(strains, angles):
    R = np.array([[np.cos(angles), np.sin(angles)],
                  [-np.sin(angles), np.cos(angles)]])

    if isinstance(angles, numbers.Number):
        strains = np.array([np.dot(R, np.dot(strain, R.T)) for strain in strains])
    elif len(angles) == len(strains):
        strains = np.array([np.dot(R[..., i], np.dot(strain, R[..., i].T)) for i, strain in enumerate(strains)])
    else:
        raise ValueError()

    return strains


def scale_strain(strain, factor):
    strain[:, 0, 0] = (1 + strain[:, 0, 0]) / (1 + factor) - 1
    strain[:, 1, 1] = (1 + strain[:, 1, 1]) / (1 + factor) - 1
    strain[:, 0, 1] = strain[:, 0, 1] / (1 + factor)
    strain[:, 1, 0] = strain[:, 1, 0] / (1 + factor)
    return strain


def planar_strain(strain):
    return np.trace(strain, axis1=1, axis2=2) / 2


def zero_median(strain):
    planar = planar_strain(strain)
    factor = np.nanmedian(planar)
    return scale_strain(strain, factor)
