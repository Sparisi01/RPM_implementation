# FunctionsGenerator.py>

import numpy as np

class FunctionsGenerator:
    
    def __init__(self, t_array: np.ndarray, l: float = 1.0, sigma: float = 1.0):
        
        self.t_array = t_array
        self.l = l
        self.sigma = sigma
        self.nt = t_array.size
        
        if self.nt <= 1:
            raise ValueError("nt must be greater than 1")
        
        self.dt = t_array[1] - t_array[0]
        self.freqs = np.fft.rfftfreq(self.nt, self.dt)
        self.omegas = 2 * np.pi * self.freqs
        
        self.K = None
    
    def compute_covariance_matrix(self, numerical_stability_factor: float = 1e-8) -> np.ndarray:
        return self._gaussian_kernel(self.t_array, self.t_array) + numerical_stability_factor * np.eye(self.nt)
    
    def _gaussian_kernel(self, x1, x2) -> np.ndarray:
        sqdist = (x1[:, None] - x2[None, :])**2
        return (self.sigma**2)*np.exp(-sqdist / (2 * self.l**2))
    
    def generate_from_white_noise(self, seed = None) -> np.ndarray:
        
        if seed is not None:
            np.random.seed(seed=seed)
        
        # Sample white noise and compute its FFT 
        f = np.random.normal(0, 1, self.nt)
        f_hat = np.fft.rfft(f)

        # Define the power spectrum
        S = self.sigma**2 * np.sqrt(2 * np.pi) * self.l * np.exp(-0.5 * self.l**2 * self.omegas**2)

        # Perform convolution in Fourier space (Note: the factor 1/dt comes from the discretization of the numpy FFT,
        # it is required for keeping the right f variance)
        f_hat_filtered = f_hat * np.sqrt(S / self.dt)
        
        f_filtered = np.fft.irfft(f_hat_filtered, n=self.nt)
        return f_filtered

    def generate_from_Gaussian_Process(self, seed = None) -> np.ndarray:
        
        if seed is not None:
            np.random.seed(seed=seed)
            
        if self.K is None: 
            self.compute_covariance_matrix()
        
        u = np.random.multivariate_normal(mean=np.zeros(self.nt), cov=self.K)
        return u