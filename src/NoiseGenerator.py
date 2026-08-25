"""
Noise generation module for thesis - RPM regularization testing
Provides multiple realistic noise scenarios for dataset augmentation

EXTENDED VERSION: Colored noise, thermal noise, and sensor bandwidth filtering
"""

import numpy as np
from typing import Literal, Tuple, Optional
from scipy.ndimage import gaussian_filter1d
from scipy import signal
from scipy.fft import rfft, irfft
from src.print_utils import print_kv, print_section

NoiseType = Literal[
    "white", 
    "correlated", 
    "lowfreq_systematic", 
    "multiplicative",
    "pink",           # 1/f noise (flicker noise)
    "brown",          # 1/f² noise (Brownian motion)
    "thermal",        # Johnson-Nyquist thermal noise
]


class NoiseGenerator:
    """
    Generate various realistic noise scenarios for testing RPM robustness.
    All noise follows E[n] = 0 (zero mean) assumption.
    
    Includes colored noise (pink/brown) and thermal noise for physical realism.
    """
    
    def __init__(self, seed: int = 42, dt: float = 0.005):
        """
        Args:
            seed: Random seed for reproducibility
            dt: Time step for signal (used for thermal noise normalization)
        """
        self.rng = np.random.default_rng(seed)
        self.dt = dt  # Time step
        self.fs = 1.0 / dt  # Sampling frequency
    
    def white_noise(self, g_all: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Independent Gaussian noise (D ≈ 0 baseline case).
        
        n(t) ~ N(0, σ²)
        σ = noise_level * std(g)
        
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
    
    def pink_noise(self, g_all: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Pink noise (1/f) - flicker noise.
        
        Power spectral density: S(f) ∝ 1/f
        
        Physical origin:
        - Electronic flicker noise in sensors
        - Fluctuations in material properties
        - Defect trapping/detrapping in semiconductors
        
        Generation method:
        1. Generate white noise in frequency domain
        2. Scale by 1/sqrt(f) to get 1/f spectrum
        3. Inverse FFT back to time domain
        
        Args:
            g_all: Observed signal
            noise_level: Amplitude scaling
        
        Returns:
            Pink noise array
        """
        n_samples, n_time = g_all.shape
        pink_noise_array = np.zeros_like(g_all, dtype=np.float32)
        
        for i in range(n_samples):
            # Generate white noise
            white = self.rng.normal(0, 1, n_time)
            
            # FFT
            white_fft = rfft(white)
            
            # Frequency array (exclude DC at f=0 to avoid division by zero)
            freqs = np.fft.rfftfreq(n_time, self.dt)
            
            # Scale by 1/sqrt(f) for 1/f spectrum (pink noise)
            # Set f[0] = f[1] to avoid singularity at DC
            scaling = np.ones_like(freqs)
            scaling[1:] = 1.0 / np.sqrt(freqs[1:])
            scaling[0] = scaling[1]  # Handle DC component
            
            # Apply scaling in frequency domain
            pink_fft = white_fft * scaling
            
            # Inverse FFT
            pink = irfft(pink_fft, n=n_time)
            
            # Normalize and scale
            pink = pink / np.std(pink)
            pink_noise_array[i] = pink
        
        # Scale by signal and noise level
        sample_std = np.std(g_all, axis=1, keepdims=True)
        pink_noise_array = (noise_level * sample_std * pink_noise_array).astype(np.float32)
        
        return pink_noise_array
    
    def brown_noise(self, g_all: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Brown noise (1/f²) - Brownian motion / random walk.
        
        Power spectral density: S(f) ∝ 1/f²
        
        Physical origin:
        - Sensor baseline drift
        - Random walk of bias voltage
        - Mechanical thermal noise (Brownian motion)
        
        Generation: Same as pink but with 1/f scaling (twice the pink noise suppression).
        
        Args:
            g_all: Observed signal
            noise_level: Amplitude scaling
        
        Returns:
            Brown noise array
        """
        n_samples, n_time = g_all.shape
        brown_noise_array = np.zeros_like(g_all, dtype=np.float32)
        
        for i in range(n_samples):
            white = self.rng.normal(0, 1, n_time)
            white_fft = rfft(white)
            
            # Frequency array
            freqs = np.fft.rfftfreq(n_time, self.dt)
            
            # Scale by 1/f for 1/f² spectrum (brown noise)
            scaling = np.ones_like(freqs)
            scaling[1:] = 1.0 / freqs[1:]
            scaling[0] = scaling[1]
            
            brown_fft = white_fft * scaling
            brown = irfft(brown_fft, n=n_time)
            
            # Normalize
            brown = brown / np.std(brown)
            brown_noise_array[i] = brown
        
        # Scale by signal and noise level
        sample_std = np.std(g_all, axis=1, keepdims=True)
        brown_noise_array = (noise_level * sample_std * brown_noise_array).astype(np.float32)
        
        return brown_noise_array
    
    def thermal_noise(self, g_all: np.ndarray, 
                     temperature: float = 298.0,
                     resistance: float = 1e4,
                     bandwidth: Optional[float] = None) -> np.ndarray:
        """
        Johnson-Nyquist thermal noise (electronic noise from thermal motion).
        
        Power spectral density: S(f) = 4 k_B T R (flat, white spectrum)
        RMS voltage: V_rms = sqrt(4 k_B T R Δf)
        
        Physical parameters:
        - k_B = 1.38e-23 J/K (Boltzmann constant)
        - T: absolute temperature (K), typically 298 K (25°C)
        - R: resistance (Ω), typical 1k-100k for sensors
        - Δf: measurement bandwidth (Hz)
        
        Physically this is WHITE noise, but scaled by realistic parameters.
        
        Args:
            g_all: Observed signal
            temperature: Absolute temperature in Kelvin (default 298 K = 25°C)
            resistance: Sensor/amplifier resistance in Ohms
            bandwidth: Measurement bandwidth (if None, use Nyquist frequency)
        
        Returns:
            Thermal noise array (scaled by physical parameters)
        
        Example:
            For a 10kΩ sensor at room temperature (298K):
            V_rms ~ sqrt(4 * 1.38e-23 * 298 * 10000 * 200) ~ 9.2 μV
        """
        # Physical constants
        k_B = 1.38e-23  # Boltzmann constant [J/K]
        
        # If bandwidth not specified, use Nyquist
        if bandwidth is None:
            bandwidth = self.fs / 2.0
        
        # Thermal noise RMS voltage
        V_rms = np.sqrt(4.0 * k_B * temperature * resistance * bandwidth)
        
        # Scale relative to signal
        signal_std = np.std(g_all)
        
        # Generate white Gaussian noise
        n_samples, n_time = g_all.shape
        thermal = self.rng.normal(0, V_rms / signal_std, size=g_all.shape)
        
        return thermal.astype(np.float32)
    
    def correlated_noise(self, g_all: np.ndarray, 
                        noise_level: float, 
                        correlation_factor: float = 0.25) -> np.ndarray:
        """
        Noise correlated with observed signal (D ≠ 0).
        Simulates systematic sensor errors proportional to measurement.
        
        n(t) = α·g(t) + ε(t)
        where α = correlation_factor, ε ~ N(0, σ_white²)
        
        Example physical cause: 
        - Sensor gain varies with temperature/measurement
        - Signal-dependent amplifier noise
        
        Args:
            g_all: Observed signal
            noise_level: Base noise level
            correlation_factor: Strength of correlation with g (0-1 typical)
        
        Returns:
            Correlated noise array
        """
        systematic = correlation_factor * g_all
        
        sample_std = np.std(g_all, axis=1, keepdims=True)
        white_component = self.rng.normal(
            0.0,
            noise_level * sample_std * (1-correlation_factor),
            size=g_all.shape
        ).astype(np.float32)
        
        noise = (systematic + white_component).astype(np.float32)
        return noise
    
    def lowfreq_systematic_noise(self, g_all: np.ndarray, 
                                noise_level: float, 
                                sigma_smooth: int = 15) -> np.ndarray:
        """
        Low-frequency systematic errors (D concentrated at low ω).
        Simulates sensor drift, baseline shift, slow calibration errors.
        
        Args:
            g_all: Observed signal
            noise_level: Amplitude factor
            sigma_smooth: Gaussian blur σ (higher = slower drift)
        
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
    
    # =========================================================================
    # SENSOR BANDWIDTH FILTERING
    # =========================================================================
    
    def apply_sensor_bandpass(self, g_all: np.ndarray,
                             f_low: float = 0.5,
                             f_high: float = 100.0,
                             order: int = 4) -> np.ndarray:
        """
        Apply bandpass filter to simulate sensor frequency response.
        
        Real sensors have limited bandwidth - they cannot measure below f_low
        or above f_high. This filter simulates that limitation.
        
        Butterworth bandpass filter: maximally flat response in passband.
        
        Physical examples:
        - Thermocouples: typically 0.1 - 50 Hz
        - Pressure sensors: 0.1 - 500 Hz
        - Displacement sensors: 0.01 - 1000 Hz
        
        Args:
            g_all: Input signal (N_samples, N_time)
            f_low: Low cutoff frequency (Hz)
            f_high: High cutoff frequency (Hz)
            order: Filter order (higher = steeper rolloff)
        
        Returns:
            Filtered signal with same shape
        """
        # Design Butterworth bandpass filter
        nyquist = self.fs / 2.0
        
        # Normalize frequencies to [0, 1] range
        if f_low < 1e-3:
            # If no lower cutoff, use highpass instead
            sos = signal.butter(order, f_high / nyquist, btype='low', output='sos')
        elif f_high > nyquist * 0.99:
            # If no upper cutoff, use lowpass instead
            sos = signal.butter(order, f_low / nyquist, btype='high', output='sos')
        else:
            sos = signal.butter(order, [f_low / nyquist, f_high / nyquist], 
                               btype='band', output='sos')
        
        # Apply filter to each sample
        g_filtered = np.zeros_like(g_all, dtype=np.float32)
        for i in range(g_all.shape[0]):
            g_filtered[i] = signal.sosfilt(sos, g_all[i]).astype(np.float32)
        
        return g_filtered
    
    def apply_sensor_highpass(self, g_all: np.ndarray,
                             f_cutoff: float = 0.1,
                             order: int = 4) -> np.ndarray:
        """
        Apply highpass filter (removes low-frequency baseline drift).
        
        Useful for sensors with AC coupling or baseline removal.
        
        Args:
            g_all: Input signal
            f_cutoff: Cutoff frequency (Hz)
            order: Filter order
        
        Returns:
            Highpass filtered signal
        """
        nyquist = self.fs / 2.0
        sos = signal.butter(order, f_cutoff / nyquist, btype='high', output='sos')
        
        g_filtered = np.zeros_like(g_all, dtype=np.float32)
        for i in range(g_all.shape[0]):
            g_filtered[i] = signal.sosfilt(sos, g_all[i]).astype(np.float32)
        
        return g_filtered
    
    def apply_sensor_lowpass(self, g_all: np.ndarray,
                            f_cutoff: float = 100.0,
                            order: int = 4) -> np.ndarray:
        """
        Apply lowpass filter (anti-aliasing, smoothing).
        
        Simulates sensor inertia or anti-aliasing filter.
        
        Args:
            g_all: Input signal
            f_cutoff: Cutoff frequency (Hz)
            order: Filter order
        
        Returns:
            Lowpass filtered signal
        """
        nyquist = self.fs / 2.0
        sos = signal.butter(order, f_cutoff / nyquist, btype='low', output='sos')
        
        g_filtered = np.zeros_like(g_all, dtype=np.float32)
        for i in range(g_all.shape[0]):
            g_filtered[i] = signal.sosfilt(sos, g_all[i]).astype(np.float32)
        
        return g_filtered
    
    # =========================================================================
    # COMBINED SCENARIOS
    # =========================================================================
    
    def realistic_sensor_measurement(self, g_all: np.ndarray,
                                     f_low: float = 0.1,
                                     f_high: float = 50.0,
                                     noise_level: float = 0.05,
                                     temperature: float = 298.0,
                                     resistance: float = 1e4) -> np.ndarray:
        """
        Simulate realistic sensor measurement with all imperfections:
        1. Bandpass filtering (sensor bandwidth)
        2. Thermal noise (Johnson-Nyquist)
        3. Pink noise (flicker noise in electronics)
        
        This represents a realistic experimental setup.
        
        Args:
            g_all: Clean signal
            f_low: Sensor lower frequency cutoff (Hz)
            f_high: Sensor upper frequency cutoff (Hz)
            noise_level: Thermal noise level (fraction of signal std)
            temperature: Sensor temperature (K)
            resistance: Sensor/amplifier resistance (Ω)
        
        Returns:
            Signal with realistic measurement artifacts
        """
        # Step 1: Apply sensor bandpass
        g_filtered = self.apply_sensor_bandpass(g_all, f_low, f_high, order=2)
        
        # Step 2: Add thermal noise
        noise_thermal = self.thermal_noise(g_filtered, temperature, resistance, 
                                          bandwidth=f_high-f_low)
        
        # Step 3: Add pink noise (flicker noise)
        noise_pink = self.pink_noise(g_filtered, noise_level * 0.5)
        
        # Combine
        g_measured = g_filtered + noise_thermal + noise_pink
        
        return g_measured.astype(np.float32)
    
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
            'pink': self.pink_noise(g_all, noise_level),
            'brown': self.brown_noise(g_all, noise_level),
            'thermal': self.thermal_noise(g_all),
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
        elif noise_type == 'pink':
            noise = self.pink_noise(g_all, noise_level)
        elif noise_type == 'brown':
            noise = self.brown_noise(g_all, noise_level)
        elif noise_type == 'thermal':
            noise = self.thermal_noise(g_all)
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


def apply_noise_to_dataset(g_all: np.ndarray,
                          f_all: np.ndarray,
                          noise_level: float = 0.0,
                          noise_type: NoiseType = 'white',
                          sensor_f_low: Optional[float] = None,
                          sensor_f_high: Optional[float] = None,
                          seed: int = 42,
                          verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main function to apply noise to dataset (use in notebook).
    
    Parameters:
    -----------
    g_all : np.ndarray
        Observed signals, shape (n_samples, n_time)
    f_all : np.ndarray
        Target signals, shape (n_samples, n_time)
    noise_level : float
        Noise magnitude (0.05 = 5% of signal std)
    noise_type : str
        One of: 'white', 'pink', 'brown', 'thermal', 'correlated', 
                'lowfreq_systematic', 'multiplicative'
    sensor_f_low : float, optional
        Lower sensor frequency cutoff (Hz). If None, no filtering applied.
    sensor_f_high : float, optional
        Upper sensor frequency cutoff (Hz). If None, no filtering applied.
    seed : int
        Random seed for reproducibility
    verbose : bool
        Print statistics
    
    Returns:
    --------
    g_noisy : np.ndarray
        Signal with added noise (and optional filtering)
    f_all : np.ndarray
        Unchanged (for convenience)
    """
    dt = 0.005  # Adjust based on your actual dt
    generator = NoiseGenerator(seed=seed, dt=dt)
    
    # Apply sensor filtering if specified
    if sensor_f_low is not None or sensor_f_high is not None:
        f_low = sensor_f_low if sensor_f_low is not None else 0.001
        f_high = sensor_f_high if sensor_f_high is not None else 1000.0
        g_all = generator.apply_sensor_bandpass(g_all, f_low, f_high)
        if verbose:
            print(f"✓ Sensor bandpass filter applied: [{f_low} Hz, {f_high} Hz]")
    
    # Apply noise
    g_noisy, noise = generator.apply_noise(g_all, noise_level, noise_type)
    
    if verbose and noise_level > 0:
        print(f"\n✓ Noise applied: {noise_type}")
        print_noise_stats(noise, scenario_name=noise_type)
        
        # Estimate correlation indicator
        D_empirical = np.mean(noise * f_all, axis=1)
        C_empirical = np.mean(g_noisy * f_all, axis=1)
        ratio = np.abs(D_empirical) / (np.abs(C_empirical) + 1e-10)
        print_kv("D/C ratio (correlation indicator)", f"{np.mean(ratio):12.4f}")
    
    return g_noisy, f_all