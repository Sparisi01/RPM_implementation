# generate_f_from_white_noise.py>

import numpy as np
import matplotlib.pyplot as plt

def generate_random_f_from_white_noise(t_array: np.ndarray, filter, sigma_white_noise: float = 1, T_0: float = 0, oscillations_power: float = 1, verbose:bool=False) -> np.ndarray:
    
    nt = t_array.size
    if nt <= 1:
        raise ValueError('nt must be greater than 1')

    dt = t_array[1]-t_array[0]
      
    # Genera white noise nel tempo
    # Proprietà da rispettare:
    # - Media nulla
    # - Matrice di correlazione proporzionale all'identità
    f = np.random.normal(0,sigma_white_noise,nt)

    f_hat = np.fft.rfft(f)
    freqs = np.fft.rfftfreq(nt, dt)

    # Applica filtro gaussiano nello spazio delle frequenze
    # NOTE: Il filtro gaussiano è una scelta di comodo, bisogna capire in base al problema
    # che tipo di filtro nello spazio delle frequenza applicare. Questo mi aspetto influenzi fortemente la fase 
    # di training e la risultante capacità di generalizzazione del modello.
    f_hat_filtered = f_hat * filter(freqs)

    f_filtered = np.fft.irfft(f_hat_filtered, n=nt)

    # Normalizza f_filtered e rendilo fisicamente coerente con una possibile oscillazione attorno a un valore noto.
    # Di default oscilla attorno a 0 con amplificazione unitaria.
    f_filtered = (f_filtered - f_filtered.mean()) / f_filtered.std()
    f_filtered = T_0 + oscillations_power*f_filtered
    
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


def generate_f_as_Gaussian_Process(n : int ,N_samples : int = 1,low : float = 0, high : float = 1, l : float = 0.1, sigma : float = 1) -> np.ndarray:
    
    # Space array
    x_array = np.linspace(low, high, n)

    # Gaussian kernel
    def kernel(x1, x2, l: np.array=0.1, sigma: np.ndarray = 1):
        sqdist = (x1[:, None] - x2[None, :])**2
        return (sigma*sigma)*np.exp(-sqdist / (2 * l**2))

    # Covariance Matrix
    K = kernel(x_array, x_array, l=l, sigma=sigma)
 
    #K += 1e-8 * np.eye(n)

    # Sample
    u = np.random.multivariate_normal(mean=np.zeros(n), cov=K, size=N_samples).T
    
    return u