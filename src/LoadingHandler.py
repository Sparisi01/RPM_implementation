
import numpy as np
from typing import Tuple
import h5py

def load_simulation_info(dataset_code: str) -> Tuple[float, float, int, float, float, float, float, float]:
    """Load simulation parameters from CSV."""
    L, Tf, nt, D, h_param, l, _, r, _ = np.loadtxt(
        f"dataset/{dataset_code}simulation_info.csv", delimiter=";", usecols=1
    )
    return L, Tf, int(nt), D, h_param, l, r, Tf / int(nt)

def load_impulse_response(dataset_code: str, n: int, dt: float) -> Tuple[np.ndarray, int, np.ndarray]:
    """Load and FFT impulse response."""
    h_vals = np.loadtxt(f"dataset/{dataset_code}impulse_response.csv", delimiter=";")
    H_np = np.fft.rfft(h_vals, n).astype(np.complex64)
    nk = len(H_np)
    omegas = np.fft.rfftfreq(n, d=dt) * 2 * np.pi
    return H_np, nk, omegas

def load_dataset(dataset_code: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load f and g from HDF5"""
    H5_PATH = f"dataset/{dataset_code}dataset.h5"
    with h5py.File(H5_PATH, "r") as hf:
        data = hf["dataset"][:]  # (N, 2, nt)
    
    f_all = data[:, 0, :].astype(np.float32)
    g_all = data[:, 1, :].astype(np.float32)
    return f_all, g_all