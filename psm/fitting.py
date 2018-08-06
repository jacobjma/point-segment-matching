import numpy as np
from scipy import optimize


class PeakModel(object):
    
    def __init__(self):
        self._params = None
    
    @property
    def params(self):
        return self._params

class Polynomial2D(PeakModel):
    
    def __init__(self):
        super(Polynomial2D, self).__init__()
    
    def fit(self, x, y, z):
        
        x,y,z = x.ravel(),y.ravel(),z.ravel()
        
        X = np.array([np.ones(len(x)), x, y, x**2, x * y, y**2])
        
        params, residues, rank, singval = np.linalg.lstsq(X.T, z, rcond=None)
        
        self._params = params
    
    def peak_position(self):
        
        M = [[2 * self.params[3], self.params[4]], 
            [self.params[4], 2 * self.params[5]]]
        
        return np.linalg.solve(M, -self.params[1:3])

class Gaussian2D(PeakModel):
    
    def __init__(self, elliptical=False):
        
        super(Gaussian2D, self).__init__()
        
        self.elliptical = elliptical
        
        if elliptical:
            self._n_params = 7
        else:
            self._n_params = 5

    @property
    def params(self):
        return self._params
            
    def fit(self, x, y, z):
        
        x,y,z=x.ravel(),y.ravel(),z.ravel()
        
        initial = self.get_initial(x, y, z)
        
        bounds = self.get_bounds(x, y, z)
        
        if self.elliptical:
            def fit_func(p):
                x0, y0, z0, A, a, b, c = p
                return A * np.exp(-(a*(x-x0)**2 - 2*b*(x-x0)*(y-y0) + c*(y-y0)**2)) - z
        else:            
            def fit_func(p): 
                x0, y0, z0, A, a = p
                return z0 + A * np.exp(-a*((x-x0)**2 + (y-y0)**2)) - z
        
        ls = optimize.least_squares(fit_func, initial, bounds=bounds)
        
        self._params = ls.x
        
    def get_bounds(self, x, y, z):
    
        extent = np.sqrt((np.max(x)-np.min(x))*(np.max(y)-np.min(y)))
        
        if self.elliptical:
            bounds = [(-extent, -extent, 0, 0, 0, -extent, 0),
                        (extent, extent, z.max(), np.inf, extent, extent, extent)]
        else:
            bounds = [(-extent, -extent, 0, 0, 0),
                        (extent, extent, z.max(), np.inf, extent)]
        
        return bounds
        
    def get_initial(self, x, y, z):

        initial = np.zeros(self._n_params)
        initial[0] = (x * z).sum() / z.sum()
        initial[1] = (y * z).sum() / z.sum()
        initial[2] = z.min()
        initial[3] = z.max() - z.min()
        
        polynomial = Polynomial2D()
        polynomial.fit(x, y, z)
        
        if self.elliptical:
            initial[4] = np.abs(polynomial.params[3])
            initial[5] = 0
            initial[6] = np.abs(polynomial.params[5])
        else:
            initial[4] = (np.abs(polynomial.params[3]) + np.abs(polynomial.params[5]))/2
        
        return initial
        
    def peak_position(self):
        return self.params[0], self.params[1]

class Gaussian2DBatch(Gaussian2D):
    
    def __init__(self, elliptical=False, n_samples=100):
        
        super(Gaussian2DBatch, self).__init__(elliptical=elliptical)
        
        self._n_samples = n_samples
        
        self._saved_params = np.zeros((n_samples, self._n_params))
        
        self._initial = None
        
        self._n_saved = 0
        
    def get_initial(self, x, y, z):
        
        if self._initial is None:
            return super(Gaussian2DBatch, self).get_initial(x, y, z)
        else:
            return self._initial
        
    def fit(self, x, y, z):
        
        super(Gaussian2DBatch, self).fit(x, y, z)
        
        if self._n_saved < self._n_samples:
            self._saved_params[self._n_saved] = self._params
            self._n_saved += 1
        elif self._n_saved == self._n_samples:
            self._initial = np.median(self._saved_params, axis=0)