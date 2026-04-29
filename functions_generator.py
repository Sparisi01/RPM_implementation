# generate_f_from_white_noise.py>

import numpy as np
import matplotlib.pyplot as plt

def generate_random_f_from_white_noise(t_array, l=1.0, sigma=1.0):
    
    nt = t_array.size
    if nt <= 1:
        raise ValueError("nt must be greater than 1")
    dt = t_array[1] - t_array[0]

    f = np.random.normal(0, 1, nt)
    f_hat = np.fft.rfft(f)
    freqs = np.fft.rfftfreq(nt, dt)
    omega = 2 * np.pi * freqs

    S = sigma**2 * np.sqrt(2 * np.pi) * l * np.exp(-0.5 * l**2 * omega**2)

    # dividere per dt per correggere fattore di normalizzazione della fft discreta di numpy
    f_hat_filtered = f_hat * np.sqrt(S / dt)

    f_filtered = np.fft.irfft(f_hat_filtered, n=nt)
    return f_filtered

def generate_random_f_from_Gaussian_Process(t_array : np.ndarray, l : float = 0.1, sigma : float = 1) -> np.ndarray:
    
    n = np.size(t_array)

    # Gaussian kernel
    def kernel(x1, x2, l: np.array=0.1, sigma: np.ndarray = 1):
        sqdist = (x1[:, None] - x2[None, :])**2    
        return (sigma*sigma)*np.exp(-sqdist / (2 * l**2))

    # Covariance Matrix
    K = kernel(t_array, t_array, l=l, sigma=sigma)
 
    K += 1e-8 * np.eye(n)

    # Sample
    u = np.random.multivariate_normal(mean=np.zeros(n), cov=K)
    
    return u