"""
Noise generation module for thesis - RPM regularization testing
Provides multiple realistic noise scenarios for dataset augmentation

"""

import numpy as np
from typing import Literal, Tuple, Optional
from scipy.ndimage import gaussian_filter1d
from scipy import signal
from src.print_utils import print_kv

NoiseType = Literal[
    "white", 
    "correlated", 
    "lowfreq_systematic", 
    "multiplicative"
]

class NoiseGenerator:
    """
    Generate various realistic noise scenarios for testing RPM robustness.
    All noise follows E[n] = 0 (zero mean) assumption.
    """
    
    def __init__(self, seed: int = 42, dt: float = 0.005):
        self.rng = np.random.default_rng(seed)
        self.dt = dt  # Time step
        self.fs = 1.0 / dt  # Sampling frequency
    
    def white_noise(self, g_all: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Independent Gaussian noise
        
        n(t) ~ N(0, var)
        var = noise_level * std(g)
        
        Args:
            g_all: Observed signal (N_samples, N_time)
            noise_level: Fraction of signal std, e.g., 0.05 = 5%
        
        Returns:
            Noise array same shape as g_all
        """
        
        sample_std = np.std(g_all, axis=1, keepdims=True)
        noise = self.rng.normal(
            0.0,
            noise_level * sample_std,
            size=g_all.shape
        ).astype(np.float32)
        
        return noise
    
    def correlated_noise(
        self,
        g_all: np.ndarray,
        noise_level: float,
        white_noise_factor: float = 0.25,
        tau: int = 20,
    ) -> np.ndarray:
        """
        Noise correlated with observed signal, optionally with a temporal delay.

        Model:
            n(t) = noise_level * g(t - tau) + white(t)

        where:
            - g_all: signal array, shape (n_signals, n_samples)
            - noise_level: strength of the correlated component
            - white_noise_factor: strength of the independent white noise
            - tau: temporal delay in samples.

        Returns:
            Noise array with the same shape as g_all.
        """

        # Correlated component
        if tau == 0:
            g_shifted = g_all
        elif tau > 0:
            # g(t - tau): delay by tau samples
            g_shifted = np.zeros_like(g_all)
            g_shifted[:, tau:] = g_all[:, :-tau]
        else:
            # tau < 0: advance by |tau| samples
            shift = abs(tau)
            g_shifted = np.zeros_like(g_all)
            g_shifted[:, :-shift] = g_all[:, shift:]

        systematic = noise_level * g_shifted

        # Independent white noise
        sample_std = np.std(g_all, axis=1, keepdims=True)

        white_component = self.rng.normal(
            0.0,
            white_noise_factor * sample_std,
            size=g_all.shape
        ).astype(np.float32)

        noise = (systematic + white_component).astype(np.float32)

        return noise

    
    def lowfreq_systematic_noise(self, g_all: np.ndarray, 
                                noise_level: float, 
                                sigma_smooth: int = 15) -> np.ndarray:
        """
        Low-frequency systematic errors (D concentrated at low omegas).
        Simulates sensor drift, baseline shift, slow calibration errors.
        
        Args:
            g_all: Observed signal
            noise_level: Amplitude factor
            sigma_smooth: Gaussian blur std (higher = slower drift)
        
        Returns:
            Low-frequency noise array
        """
        raw_noise = self.rng.normal(0, 1, size=g_all.shape).astype(np.float32)
        lowfreq = gaussian_filter1d(raw_noise, sigma=sigma_smooth, axis=1)
        lowfreq = lowfreq / np.max(np.abs(lowfreq)) * noise_level
        
        return lowfreq.astype(np.float32)
    
    def multiplicative_noise(self, g_all: np.ndarray, 
                            noise_level: float) -> np.ndarray:
        """
        Multiplicative noise (amplitude-dependent error).
        
        Args:
            g_all: Observed signal
            noise_level: Std of multiplicative factor
        
        Returns:
            Multiplicative noise array
        """
        gains = self.rng.normal(
            loc=1.0, 
            scale=noise_level, 
            size=(g_all.shape[0], 1)
        )
        
        noise = (gains - 1.0) * g_all
        return noise.astype(np.float32)
    
    def generate_all_scenarios(self, 
                              g_all: np.ndarray,
                              noise_level: float) -> dict:
        """
        Generate all noise scenarios for comparison testing.
        
        Args:
            g_all: Observed signal (N_samples, N_time)
            noise_level: Noise magnitude (0.05 = 5% typical)
        
        Returns:
            Dictionary with keys for each scenario
        """
        scenarios = {
            'white': self.white_noise(g_all, noise_level),
            'correlated': self.correlated_noise(g_all, noise_level, correlation_factor=0.3),
            'lowfreq_systematic': self.lowfreq_systematic_noise(g_all, noise_level, sigma_smooth=15),
            'multiplicative': self.multiplicative_noise(g_all, noise_level),
        }
        
        return scenarios
    
    def apply_noise(self, g_all: np.ndarray, 
                   noise_level: float, 
                   noise_type: NoiseType = 'white') -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a single noise type to signal.
        
        Args:
            g_all: Clean signal
            noise_level: Noise magnitude
            noise_type: Which scenario
        
        Returns:
            Tuple of (g_noisy, noise_array)
        """
        if noise_type == 'white':
            noise = self.white_noise(g_all, noise_level)
        elif noise_type == 'correlated':
            noise = self.correlated_noise(g_all, noise_level)
        elif noise_type == 'lowfreq_systematic':
            noise = self.lowfreq_systematic_noise(g_all, noise_level)
        elif noise_type == 'multiplicative':
            noise = self.multiplicative_noise(g_all, noise_level)
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")
        
        g_noisy = g_all + noise
        
        return g_noisy, noise

    def print_noise_stats(noise: np.ndarray, scenario_name: str):
        """Print statistics about the generated noise."""
        print(f"\n{scenario_name.upper()}")
        print_kv("Mean", f"{np.mean(noise):12.4e}")
        print_kv("Std", f"{np.std(noise):12.4e}")
        print_kv("Max", f"{np.max(np.abs(noise)):12.4e}")
        print_kv("Energy", f"{np.sum(noise**2):12.4e}")