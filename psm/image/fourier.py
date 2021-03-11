import numpy as np
from scipy import ndimage
from scipy.stats import vonmises


def unroll(image, inner, outer, nbins_radial=None, nbins_angular=None):
    if nbins_angular is None:
        nbins_angular = int(np.pi * inner)

    if nbins_radial is None:
        nbins_radial = outer - inner

    shape = image.shape

    sx, sy = shape
    X, Y = np.ogrid[0:sx, 0:sy]
    r = np.hypot(X - sx / 2, Y - sy / 2)
    radial_bins = -np.ones(shape, dtype=int)
    valid = (r > inner) & (r < outer)
    radial_bins[valid] = nbins_radial * (r[valid] - inner) / (outer - inner)

    angles = np.arctan2(X - sx / 2, Y - sy / 2) + np.pi + np.pi / nbins_angular
    angular_bins = (nbins_angular * angles / (2 * np.pi) - 1e-7).astype(np.int)

    unrolled = np.zeros((nbins_radial, nbins_angular))
    for i in range(nbins_radial):
        j = radial_bins == i
        unrolled[i] = ndimage.mean(image[j], angular_bins[j], range(0, nbins_angular))

    return unrolled


def build_template(positions, length, amplitudes=None, width=5e2):
    rv = vonmises(width)
    x = np.linspace(0, 2 * np.pi, length, endpoint=False)
    y = np.sum(rv.pdf(x[:, None] - positions), axis=1)
    return y


class Mask(object):

    def __init__(self, width, ratio=None):
        self._width = width
        self._ratio = ratio
        self._shape = None

    def set_shape(self, shape):
        self._shape = shape
        if self._ratio is None:
            self._ratio = shape[0] / shape[1]

    def indices(self, center):
        center_round = np.round(center).astype(int)

        width_x, width_y = self.mask_array_widths()

        min_x = np.max((center_round[0] - width_x, 0))
        max_x = np.min((center_round[0] + width_x + 1, self._shape[0]))
        min_y = np.max((center_round[1] - width_y, 0))
        max_y = np.min((center_round[1] + width_y + 1, self._shape[1]))

        return np.mgrid[min_x:max_x, min_y:max_y]

    def mask_array_widths(self):
        return int(5 * self._width * self._ratio), int(5 * self._width)

    def apply(self, spectrum, centers):
        shape = spectrum.shape

        mask_array = np.zeros(shape)

        self.set_shape(shape)

        for center in centers:
            x, y, mask_values = self.get_mask(center)

            mask_array[x, y] += mask_values

        mask_array[mask_array > 1] = 1

        return mask_array

    def get_mask(self, center):
        raise NotImplementedError()


class GaussianMask(Mask):

    def __init__(self, width, ratio=None):
        super().__init__(width, ratio)

    def get_mask(self, center):
        x0, y0 = center
        x, y = self.indices(center)

        return x, y, np.exp(
            -(x0 - x) ** 2 / (2 * (self._width * self._ratio) ** 2) - (y0 - y) ** 2 / (2 * self._width ** 2))


class ButterworthMask(Mask):

    def __init__(self, width, ratio=None, order=4):
        super().__init__(width, ratio)

        self._order = order

    def get_mask(self, center):
        x0, y0 = center
        x, y = self.indices(center)

        r = lambda x0, y0, x, y: np.sqrt(((x0 - x) / self._ratio) ** 2 + (y0 - y) ** 2) / self._width

        return x, y, np.sqrt(1 / (1 + r(x0, y0, x, y) ** (2. * self._order)))


class CosineMask(Mask):

    def __init__(self, width, ratio=None):
        super().__init__(width, ratio)

    def get_mask(self, center):
        x0, y0 = center
        x, y = self.indices(center)

        r = lambda x0, y0, x, y: np.sqrt(((x0 - x) / self._ratio) ** 2 + (y0 - y) ** 2) / (self._width / 2)

        return x, y, (1. / 2 + np.cos(r(x0, y0, x, y)) / 2.) * (r(x0, y0, x, y) < np.pi)
