import matplotlib.pyplot as plt
import numpy as np

from psm.bravais import BravaisDetector
from psm.image.peaks import single_crystal_sweep
from psm.gui_old import PointsEditor
from psm.image.preprocess import normalize_range


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


class GaussianMask(Mask):

    def __init__(self, width, ratio=None):
        super(GaussianMask, self).__init__(width, ratio)

    def apply(self, center):
        x0, y0 = center
        x, y = self.indices(center)

        return x, y, np.exp(
            -(x0 - x) ** 2 / (2 * (self._width * self._ratio) ** 2) - (y0 - y) ** 2 / (2 * self._width ** 2))


class ButterworthMask(Mask):

    def __init__(self, width, ratio=None, order=4):
        super(ButterworthMask, self).__init__(width, ratio)

        self._order = order

    def apply(self, center):
        x0, y0 = center
        x, y = self.indices(center)

        r = lambda x0, y0, x, y: np.sqrt(((x0 - x) / self._ratio) ** 2 + (y0 - y) ** 2) / self._width

        return x, y, np.sqrt(1 / (1 + r(x0, y0, x, y) ** (2. * self._order)))


class CosineMask(Mask):

    def __init__(self, width, ratio):
        super(CosineMask, self).__init__(width, ratio)

    def apply(self, center):
        x0, y0 = center
        x, y = self.indices(center)

        r = lambda x0, y0, x, y: np.sqrt(((x0 - x) / self._ratio) ** 2 + (y0 - y) ** 2) / (self._width / 2)

        return x, y, (1. / 2 + np.cos(r(x0, y0, x, y)) / 2.) * (r(x0, y0, x, y) < np.pi)


def _fft_extent(n, d=1.0):
    if n % 2 == 0:
        return [-n / 2 / (d * n), (n / 2 - 1) / (d * n)]
    else:
        return [-(n - 1) / 2 / (d * n), (n - 1) / 2 / (d * n)]


def _fft_extent_2d(shape, d=1.0):
    return _fft_extent(shape[0], d) + _fft_extent(shape[1], d)


class BraggFilter(object):
    """BraggFilter object.

    The BraggFilter object is used filter an image based masking spatial frequencies 
    away from the main spots in a periodic image.
    
    Parameters
    ----------
    image : ndarray
        Input image.
    centers : Points object
        The reciprocal lattice centers corresponding to the
    
    """

    def __init__(self, image, centers=None):

        self.set_image(image)

        self.centers = centers

        self._reciprocal_lattice = None

    def register_diffraction_peaks(self, rmsd_max, num_peaks=9, **kwargs):

        """Finds the reciprocal lattice of a single crystal. 
        
        The peak finding is performed using stm.find_local_peaks on the normalized 
        transformed power spectrum, P'', given by
            P'' = (P' - min(P')) / (max(P') - min(P'))
        where
            P' = log(1 + s * P)
        and s controls the enhancement of low intensity pixels.
        
        Parameters
        ----------
        min_distance : int
            Minimum number of pixels separating peaks in a region of `2 * min_distance + 1` 
            (i.e. peaks are separated by at least `min_distance`).
            To find the maximum number of peaks, use `min_distance=1`.
        threshold : float, optional
            Minimum relative intensity of peaks. By default, the threshold 
            is zero.
        scale : float
            Constant controlling the enhancement of low intensity pixels. 
        exclusion radius : float
            Radius from zero-frequency component excluded.
        **kwargs :
            Additional keyword arguments for stm.find_local_peaks
        """

        power_spectrum = np.log(1 + 1 * np.abs(self._fft_image))

        detector = BravaisDetector(rmsd_max)

        detectors = single_crystal_sweep(power_spectrum, detector, num_peaks, **kwargs)

        n_faces = [detector.n_faces for detector in detectors]

        best = [detector for detector in detectors if detector.n_faces == max(n_faces)]

        rmsd = [np.mean(detector.rmsd) for detector in best]

        best = best[rmsd.index(min(rmsd))]

        g1, g2, _ = best.get_lattice_parameters()

        self._reciprocal_lattice = np.vstack((g1, g2))

    @property
    def reciprocal_lattice(self):
        return self._pixels2nyquist(self._reciprocal_lattice)

    def _nyquist2pixels(self, vectors):
        vectors = vectors.copy()
        extent = _fft_extent_2d(self._fft_image.shape)
        vectors[:, 0] = vectors[:, 0] * self._fft_image.shape[0] / (extent[1] - extent[0])
        vectors[:, 1] = vectors[:, 1] * self._fft_image.shape[1] / (extent[2] - extent[3])
        return vectors

    def _pixels2nyquist(self, vectors):
        vectors = vectors.copy()
        extent = _fft_extent_2d(self._fft_image.shape)
        vectors[:, 0] = vectors[:, 0] * (extent[0] - extent[1]) / self._fft_image.shape[0]
        vectors[:, 1] = vectors[:, 1] * (extent[3] - extent[2]) / self._fft_image.shape[1]
        return vectors

    def autoset_mask_centers(self, max_radius):

        g1, g2 = self._pixels2nyquist(self._reciprocal_lattice)

        min_norm = np.min(np.linalg.norm([g1, g2], axis=0))

        centers = []
        index_max = np.ceil(max_radius / min_norm).astype(int)
        for h in range(-index_max, index_max + 1):
            for k in range(-index_max, index_max + 1):
                center = h * g1 + k * g2
                if np.linalg.norm(center) <= max_radius:
                    centers.append(center)

        self.centers = np.array(centers)

    def show_mask_centers(self, ax=None, scale=1, facecolors='none', edgecolors='r', **kwargs):

        if ax is None:
            ax = plt.subplot()

        pow_spec = np.log(1 + scale * np.abs(self._fft_image)).T

        ax.imshow(pow_spec, cmap='gray', interpolation='nearest', extent=_fft_extent_2d(self._fft_image.shape))

        if self.centers is not None:
            ax.scatter(self.centers[:, 0], self.centers[:, 1], facecolors=facecolors, edgecolors=edgecolors, **kwargs)

        return ax

    def show_mask(self, mask, ax=None, **kwargs):

        if ax is None:
            ax = plt.subplot()

        shape = self._fft_image.shape

        mask_array = self._get_mask(shape, mask)

        ax.imshow(mask_array.T, extent=_fft_extent(shape[0]) + _fft_extent(shape[1]), **kwargs)

        return ax

    def edit_centers(self, scale=10, **kwargs):
        """Edit the mask centers.
        
        This method uses the Points editor to manually set the mask centers.
        
        Parameters
        ----------
        scale : float
            Constant controlling the enhancement of low intensity pixels. This only modifies
            the visual appearance. 
        """

        ax = plt.subplot()

        pow_spec = np.log(1 + scale * np.abs(self._fft_image)).T

        ax.imshow(pow_spec, cmap='gray', interpolation='nearest',
                  extent=_fft_extent(pow_spec.shape[0]) + _fft_extent(pow_spec.shape[1]))

        if self.centers is None:
            self.centers = np.zeros((0, 2))

        self.pe = PointsEditor(ax, self.centers)

        def _set_centers(centers):
            self.centers = centers

        self.pe.edit(close_callback=_set_centers, **kwargs)

    def set_image(self, image):
        """Set the image to be filtered.
        
        Parameters
        ----------
        image : ndarray
            Input image.
        """

        self._fft_image = np.fft.fftshift(np.fft.fft2(image))

    def _get_mask_array(self, center, x, y, mask):

        mask_array = mask.apply(center[0], center[1], x, y)

        return mask_array

    def _get_mask(self, shape, mask):
        # TODO: Speed up mask calculation
        mask_array = np.zeros(shape)

        extent = _fft_extent_2d(self._fft_image.shape)

        centers = self._nyquist2pixels(self.centers - np.array([extent[0], extent[-1]]))

        mask.set_shape(shape)

        for center in centers:
            x, y, mask_values = mask.apply(center)

            mask_array[x, y] += mask_values

        mask_array[mask_array > 1] = 1

        return mask_array

    def apply_filter(self, mask, image=None, return_mask=False):

        """Apply Bragg filter to image.
        
        Parameters
        ----------
        mask : callable
            The function defining the Bragg filter mask(s). 
            Must be in the form mask(x0, y0, x, y). The mask center 
            image indices, (x0,y0), are floats, and the image indices 
            x and y are ndarrays.
        image : ndarray
            Input image. Defaults to the previously set image.
        return_mask : bool
            If True, also return the mask array.
            
        Returns
        ----------
        filter image : ndarray
            The Bragg filtered image.
        filter image : ndarray, optional
            The mask array multiplied with the fourier transformed image.
        """

        if image is None:
            fft_image = self._fft_image.copy()
        else:
            fft_image = np.fft.fftshift(np.fft.fft2(image))

        mask_array = self._get_mask(fft_image.shape, mask)

        filtered_image = np.fft.fft2(np.fft.fftshift(fft_image * mask_array)).real

        filtered_image = normalize_range(filtered_image)

        if return_mask:
            return filtered_image, mask_array
        else:
            return filtered_image
