import numbers

import numpy as np


def rotate_points(points, angle, center='cop'):
    """Rotate point positions.

    Parameters:
    angle: float
        Angle that the points are rotated
    center: str or 2-tuple
        The center is kept fixed under the rotation. Use 
        'cop' to fix the center of positions.
    """

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
    """Rotate the xy-directions of an array of strain tensors.

    Parameters:
    strains: Nx4x4 array
        Array of N strain tensors
    angle: float
        Angle that the strain is rotated
    """
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
    # TODO: Docstring
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


def minimal_rotation(rotation, symmetry_order):
    angle = 2 * np.pi / symmetry_order

    valid = np.isnan(rotation) == 0

    new_rotation = np.empty_like(rotation)
    new_rotation[:] = np.nan

    rotation = rotation[valid] % angle

    new_rotation[valid] = np.where(rotation <= np.abs(rotation - angle), rotation, rotation - angle)

    return new_rotation
