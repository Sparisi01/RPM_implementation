# generate_f_from_white_noise.py>

import numpy as np
import matplotlib.pyplot as plt

def generate_random_f_from_white_noise(t_array : np.ndarray, l : float = 1, sigma : float = 1, verbose : bool=False) -> np.ndarray:
    
    nt = t_array.size
    if nt <= 1:
        raise ValueError('nt must be greater than 1')

    dt = t_array[1]-t_array[0]
      
    f = np.random.normal(0,1,nt)

    f_hat = np.fft.rfft(f)
    freqs = np.fft.rfftfreq(nt, dt)

    omega = 2*np.pi*freqs

    #H = sigma * np.sqrt(l) * np.exp(-0.25 * l**2 * omega**2)    
    #f_hat_filtered = f_hat * H
    S = sigma**2 * np.sqrt(2*np.pi) * l * np.exp(-0.5 * l**2 * omega**2)

    f_hat_filtered = f_hat * np.sqrt(S)

    f_filtered = np.fft.irfft(f_hat_filtered, n=nt)
    
    # Parto da segnale con spettro in potenza piatto e rimuovo frequenze quindi
    # "perdo energia", devo rinormalizzare se voglio mantenere lo stesso integrale dello spettro energetico
    f_filtered = f_filtered / np.std(f_filtered) * sigma

    # Plot del white noise e della funzione generata
    if verbose:
        
        # Plot Noise e funzione generata
        plt.figure(figsize=(3.4, 2.6))    
        plt.plot(t_array, f, label="White Noise", alpha=0.5)
        plt.plot(t_array, f_filtered, label="Generated f",lw=2)
        plt.legend()
        plt.minorticks_on()
        plt.xlabel("Time")
        plt.ylabel("Temperature")
        plt.title("Random function generater from White Noise")
        plt.tight_layout()
        plt.savefig("assets/random_f_from_white_noise.pdf")
        plt.show()

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