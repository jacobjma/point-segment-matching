import numpy as np
from scipy.ndimage import median_filter
from scipy.ndimage.filters import gaussian_filter


def normalize_range(image, vmin=0, vmax=1):
    # TODO: Docstring
    if vmax < vmin:
        raise ValueError()
    
    image = (image-image.min())/(image.max()-image.min())
    
    return vmin + (vmax-vmin)*image

def local_normalization(image, sigma1, sigma2):
    # TODO: Docstring
    image = normalize_range(image)
    
    image = image - gaussian_filter(image,sigma1)
    
    image = image / np.sqrt(gaussian_filter(image**2, sigma2))
        
    return image

def remove_hot_pixels(image, threshold=.5, filter_size=3):
    # TODO: Docstring
    filtered = median_filter(image, size=filter_size)
    
    hot = normalize_range(image - filtered) > threshold

    image[hot] = filtered[hot]
    
    return image
    
def remove_dead_pixels(image, threshold=.5, filter_size=3):
    # TODO: Docstring
    filtered = median_filter(image, size=filter_size)
    
    dead = normalize_range(filtered - image) > threshold
    
    image[dead] = filtered[dead]
    
    return image