import numpy as np
import matplotlib.pyplot as plt
from ssm.gui import PointsEditor
from ssm.preprocess import normalize_range
from ssm.register import MatchGraph
from ssm.bravais import single_crystal_sweep, CrystalDetector
from tqdm import tqdm_notebook as tqdm

def gaussian_mask(width, ratio=1):
    return lambda x0,y0,x,y: np.exp(-(x0 - x)**2 / (2*(width * ratio)**2) - (y0 - y)**2 / (2*width**2))

def butterworth_mask(width, ratio=1, order=3):
    r = lambda x0,y0,x,y: np.sqrt(((x0 - x) / ratio)**2 + (y0 - y)**2) / width
    return lambda x0,y0,x,y: np.sqrt(1 / (1 + r(x0,y0,x,y)**(2. * order)))

def cosine_mask(width, ratio=1):
    r = lambda x0,y0,x,y: np.sqrt(((x0 - x) / ratio)**2 + (y0 - y)**2) / (width/2)
    return lambda x0,y0,x,y: (1./2+np.cos(r(x0,y0,x,y))/2.)*(r(x0,y0,x,y)<np.pi)

def _fft_extent(n, d=1.0):
    if n%2 == 0:
        return [-n/2/(d*n),(n/2-1)/(d*n)]
    else:
        return [-(n-1)/2/(d*n),(n-1)/2/(d*n)]
    
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
    
    def register_diffraction_peaks(self, rmsd_max=.1, alpha=1, num_peaks=9, **kwargs):
        
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
        
        power_spectrum = np.log(1 + alpha*np.abs(self._fft_image))
        
        rmsd_calc = MatchGraph(transform='rigid', scale_invariant=True, pivot='cop')

        detector = CrystalDetector(rmsd_calc, rmsd_max)

        detectors = single_crystal_sweep(power_spectrum, detector, num_peaks, **kwargs)
        
        num_crystal_faces = [detector.num_crystal_faces() for detector in detectors]

        best = [detector for detector in detectors if detector.num_crystal_faces() == max(num_crystal_faces)]

        average_crystal_rmsd = [detector.average_crystal_rmsd() for detector in best]

        best = best[average_crystal_rmsd.index(min(average_crystal_rmsd))]
        
        self._reciprocal_lattice = best.get_lattice_parameters('corners')
    
    @property
    def reciprocal_lattice(self):
        
        extent = _fft_extent(self._fft_image.shape[0])
        g1 = self._reciprocal_lattice[1]*(extent[1] - extent[0])/self._fft_image.shape[0]
        extent = _fft_extent(self._fft_image.shape[1])
        g2 = self._reciprocal_lattice[0]*(extent[0] - extent[1])/self._fft_image.shape[1]

        return np.vstack((g1,g2))
    
    def autoset_mask_centers(self, max_norm):
        
        g1,g2,_ = self._reciprocal_lattice
        
        center = np.array(self._fft_image.shape)/2
        centers = []
        h_max = np.floor(max_norm).astype(int)
        for h in range(-h_max,h_max+1):
            k_max = np.sqrt(max_norm**2 - h**2).astype(int)
            for k in range(-k_max, k_max+1):
                if np.sqrt(h**2 + k**2) <= max_norm:
                    centers.append([h*g1[0] + k*g2[0] + center[0],
                                     h*g1[1] + k*g2[1] + center[1]])
        
        #for h in range(-max_order,max_order+1):
        #    k_max = np.sqrt(max_order**2 - h**2).astype(int)
        #    for k in range(-k_max, k_max+1):
        #        if not (h==0)&(k==0):
        
        #centers = best.crystal_points()
        
        self.centers = self.transform_centers(np.array(centers))
        
    def transform_centers(self, centers):
        new_centers = np.zeros_like(centers)
        
        extent = _fft_extent(self._fft_image.shape[0])
        new_centers[:,0] = centers[:,0]*(extent[1] - extent[0])/self._fft_image.shape[0] + extent[0]
        extent = _fft_extent(self._fft_image.shape[1])
        new_centers[:,1] = centers[:,1]*(extent[0] - extent[1])/self._fft_image.shape[1] + extent[1]
        
        return new_centers
    
    def untransform_centers(self, centers):
        new_centers = np.zeros_like(centers)
    
        extent = _fft_extent(self._fft_image.shape[0])
        new_centers[:,0] = (centers[:,0]-extent[0])*self._fft_image.shape[0]/(extent[1] - extent[0])
        extent = _fft_extent(self._fft_image.shape[1])
        new_centers[:,1] = (centers[:,1]-extent[1])*self._fft_image.shape[1]/(extent[0] - extent[1])
        return new_centers
        
    def show_mask_centers(self, ax=None, scale=1, facecolors='none', edgecolors='r', **kwargs):
        
        if ax is None:
            ax=plt.subplot()
        
        pow_spec = np.log(1 + scale*np.abs(self._fft_image)).T
        
        ax.imshow(pow_spec, cmap='gray', interpolation='nearest', 
                    extent=_fft_extent(self._fft_image.shape[0])+_fft_extent(self._fft_image.shape[1]))
        
        if self.centers is not None:
            ax.scatter(self.centers[:,0], self.centers[:,1], facecolors=facecolors, 
                        edgecolors=edgecolors, **kwargs)
        
        return ax
    
    def show_mask(self, ax, mask, **kwargs):
        
        if ax is None:
            ax=plt.subplot()
        
        shape = self._fft_image.shape
        
        mask_array = self._get_mask(shape, mask)
        
        ax.imshow(mask_array.T, extent=_fft_extent(shape[0]) + _fft_extent(shape[1]), **kwargs)
        
        return ax
    
    def set_centers(self, centers):
        self.centers = centers
    
    def edit_centers(self, scale=10, **kwargs):
        """Edit the mask centers.
        
        This method uses the Points editor to manually set the mask centers.
        
        Parameters
        ----------
        scale : float
            Constant controlling the enhancement of low intensity pixels. This only modifies
            the visual appearance. 
        """
        
        ax=plt.subplot()
        
        pow_spec = np.log(1 + scale*np.abs(self._fft_image)).T
        
        ax.imshow(pow_spec, cmap='gray', interpolation='nearest', 
                extent=_fft_extent(pow_spec.shape[0])+_fft_extent(pow_spec.shape[1]))
        
        if self.centers is None:
            self.centers = np.zeros((0,2))

        self.pe = PointsEditor(ax, self.centers)
        
        self.pe.edit(close_callback = self.set_centers, **kwargs)
    
    def set_image(self, image):
        """Set the image to be filtered.
        
        Parameters
        ----------
        image : ndarray
            Input image.
        """
        
        self._fft_image = np.fft.fftshift(np.fft.fft2(image))
    
    def _get_mask_array(self, point, shape, mask):
        
        x, y = np.mgrid[0:shape[0], 0:shape[1]]
        
        mask_array = np.zeros(shape)
        
        mask_array = mask_array + mask(point[0], point[1], x, y)
        
        return mask_array
    
    def _get_mask(self, shape, mask):
        mask_array = np.zeros(shape)
        
        points = self.untransform_centers(self.centers)
        
        for point in tqdm(points):
            mask_array += self._get_mask_array(point, shape, mask)
        
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
