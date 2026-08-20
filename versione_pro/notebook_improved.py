"""
IMPROVED THESIS NOTEBOOK - RPM Learned Tikhonov Regularization
Complete training pipeline for inverse problem regularization

Structure:
    1. IMPORTS & SETUP
    2. UTILITY FUNCTIONS (training, plotting, noise)
    3. PARAMETERS & CONFIGURATION
    4. DATA LOADING
    5. NOISE APPLICATION (4 scenarios)
    6. MODEL INITIALIZATION
    7. TRAINING
    8. DIAGNOSTICS & VALIDATION
"""

# =============================================================================
# 1. IMPORTS & SETUP
# =============================================================================

from src.LearnedTikhonov import LearnedTikhonov
from src.print_utils import print_kv, print_section
from src.Heat1DCrankNicolson import Heat1DCrankNicolson
from src.TikhonovHandler import TikhonovHandler

import torch
import torch.nn as nn
import h5py
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal, get_args, Tuple

plt.style.use("styles/plots.mplstyle")


# =============================================================================
# 2. UTILITY FUNCTIONS
# =============================================================================

# -------- NOISE GENERATION --------

NoiseType = Literal["white", "correlated", "lowfreq_systematic", "multiplicative"]


class NoiseGenerator:
    """Generate realistic noise scenarios for testing RPM robustness."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    
    def white_noise(self, g_all: np.ndarray, noise_level: float) -> np.ndarray:
        """Independent Gaussian noise (D ≈ 0, baseline case)."""
        sample_std = np.std(g_all, axis=1, keepdims=True)
        noise = self.rng.normal(0.0, noise_level * sample_std, 
                               size=g_all.shape).astype(np.float32)
        return noise
    
    def correlated_noise(self, g_all: np.ndarray, 
                        noise_level: float, 
                        correlation_factor: float = 0.3) -> np.ndarray:
        """Noise correlated with signal: n(t) = α·g(t) + ε(t)"""
        systematic = correlation_factor * g_all
        sample_std = np.std(g_all, axis=1, keepdims=True)
        white_component = self.rng.normal(0.0, noise_level * sample_std * 0.1, 
                                         size=g_all.shape).astype(np.float32)
        noise = (systematic + white_component).astype(np.float32)
        return noise
    
    def lowfreq_systematic_noise(self, g_all: np.ndarray, 
                                noise_level: float, 
                                sigma_smooth: int = 15) -> np.ndarray:
        """Low-frequency systematic errors (sensor drift)."""
        from scipy.ndimage import gaussian_filter1d
        raw_noise = self.rng.normal(0, 1, size=g_all.shape).astype(np.float32)
        lowfreq = gaussian_filter1d(raw_noise, sigma=sigma_smooth, axis=1)
        lowfreq = lowfreq / np.max(np.abs(lowfreq)) * noise_level
        return lowfreq.astype(np.float32)
    
    def multiplicative_noise(self, g_all: np.ndarray, 
                            noise_level: float) -> np.ndarray:
        """Multiplicative noise (gain variations): n(t) = ε(t)·g(t)"""
        gains = self.rng.normal(loc=1.0, scale=noise_level, 
                               size=(g_all.shape[0], 1))
        noise = (gains - 1.0) * g_all
        return noise.astype(np.float32)
    
    def apply_noise(self, g_all: np.ndarray, 
                   noise_level: float, 
                   noise_type: NoiseType = 'white') -> Tuple[np.ndarray, np.ndarray]:
        """Apply single noise type to signal."""
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


def print_noise_stats(noise: np.ndarray, scenario_name: str, g_all: np.ndarray = None):
    """Print statistics about generated noise and estimate D."""
    print(f"\n  {scenario_name:25s}:")
    print(f"    Mean:      {np.mean(noise):12.4e}")
    print(f"    Std:       {np.std(noise):12.4e}")
    print(f"    Max:       {np.max(np.abs(noise)):12.4e}")
    print(f"    Energy:    {np.sum(noise**2):12.4e}")
    
    if g_all is not None:
        snr = np.mean(np.std(g_all, axis=1)**2) / np.mean(np.std(noise, axis=1)**2)
        print(f"    SNR:       {10*np.log10(snr):12.2f} dB")


# -------- TRAINING --------

def smoothness_regularization(model: LearnedTikhonov) -> torch.Tensor:
    """Penalize sharp variations in R(ω) and Φ(ω)."""
    dR   = model._get_R()[1:]   - model._get_R()[:-1]
    dPhi = model._get_phi()[1:] - model._get_phi()[:-1]
    return (dR ** 2).mean() + (dPhi ** 2).mean()


_TYPES = Literal["ReduceLR", "Cosine"]


def train(
    model: LearnedTikhonov,
    G_train: torch.Tensor,
    F_train: torch.Tensor,
    G_val:   torch.Tensor,
    F_val:   torch.Tensor,
    n_epochs: int        = 2000,
    lr: float            = 1e-3,
    weight_decay: float  = 0.0,
    checkpoint_path: str = "assets/tikhonov_learned/best_tikhonov.pt",
    scheduler_patience: int   = 50,
    scheduler_factor:   float = 0.5,
    log_every: int       = 50,
    lambda_smooth: float = 0,
    scheduler_type: _TYPES = "Cosine"
) -> dict:
    """Train LearnedTikhonov model with various options."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    options = get_args(_TYPES)
    assert scheduler_type in options, f"'{scheduler_type}' is not in {options}"
    
    if scheduler_type == "Cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=2000, T_mult=2, eta_min=1e-6
        )
    elif scheduler_type == "ReduceLR":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=scheduler_patience, factor=scheduler_factor
        )

    loss_fn = nn.MSELoss()
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val = float("inf")

    for epoch in range(1, n_epochs + 1):
        # Training Step
        model.train()
        optimizer.zero_grad()
        
        pred_train = model(G_train)
        mse_train = loss_fn(pred_train, F_train)
        loss_train = mse_train
        
        if lambda_smooth != 0:
            reg_smooth = smoothness_regularization(model)
            loss_train += lambda_smooth * reg_smooth
        
        loss_train.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validation Step
        model.eval()
        with torch.no_grad():
            pred_val = model(G_val)
            mse_val = loss_fn(pred_val, F_val)
            loss_val = mse_val
            
            if lambda_smooth != 0:
                reg_smooth = smoothness_regularization(model)
                loss_val += lambda_smooth * reg_smooth

        train_l = loss_train.item()
        val_l = loss_val.item()
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_l)
        history["val_loss"].append(val_l)
        history["lr"].append(current_lr)

        if val_l < best_val:
            best_val = val_l
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": best_val,
            }, checkpoint_path)

        if epoch % log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:5d}/{n_epochs} | train MSE: {train_l:.4e} | "
                  f"val MSE: {val_l:.4e} | lr: {current_lr:.2e}")

    print_kv(f"Best val MSE", f"{best_val:.4e}, saved to {checkpoint_path}")
    return history


# -------- PLOTTING --------

def plot_diagnostics(model: LearnedTikhonov, omegas: np.ndarray, tikh_w: np.ndarray = None,
                    save_path: str = "diagnostics.pdf"):
    """Plot learned R(ω), Φ(ω), and comparison with Tikhonov."""
    fig, axes = plt.subplots(3, 1, figsize=(3.2, 3.6), sharex=True)
    ax1, ax2, ax3 = axes
    
    ax1.plot(omegas, model.get_phi(), color="steelblue", lw=1.5)
    ax1.axhline(0, color="k", lw=0.2, ls="--")
    ax1.set_ylabel(r"$\phi$")
    ax1.set_title(r"Learned model $W_\theta$")
    ax1.set_ylim([-3.3, 3.3])
    ax1.grid(True, alpha=0.3)
    
    ax2.semilogy(omegas, model.get_R(), color="darkorange", lw=1)
    ax2.set_ylabel(r"$R$")
    ax2.grid(True, alpha=0.3)

    ax3.semilogy(omegas, model.get_W_magnitude(), color="r", lw=1, label=r"Learned $|\mathcal{H}|$")
    if tikh_w is not None:
        ax3.semilogy(omegas, np.abs(tikh_w), color="b", lw=1, alpha=1, label=r"Tikhonov $|\mathcal{H}|$")
    
    ax3.set_ylabel(r"$|\mathcal{H}|$")
    ax3.set_xlabel(r"$\omega$")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print_kv(f"Diagnostic saved", f"{save_path}")


def plot_history(history: dict, save_path: str = "loss_history.pdf"):
    """Plot training history."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots()
    
    ax.semilogy(epochs, history["train_loss"], label="Train MSE")
    ax.semilogy(epochs, history["val_loss"], label="Val MSE", ls="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Loss curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print_kv(f"Loss history saved", f"{save_path}")


def plot_prediction(model: LearnedTikhonov,
                   G_val: torch.Tensor,
                   F_val: torch.Tensor,
                   dt: float,
                   n_samples: int = 3,
                   save_path: str = "predictions.pdf"):
    """Plot predictions vs ground truth."""
    model.eval()
    with torch.no_grad():
        f_pred = model(G_val).cpu().numpy()
    f_true = F_val.cpu().numpy()
    nt = f_true.shape[1]
    t = np.arange(nt)

    n_show = min(n_samples, len(f_true))
    fig, axes = plt.subplots(n_show, 1, figsize=(3.2, 1.2 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    G_array = G_val.cpu().numpy()
    mse_array = np.mean((f_pred - f_true) ** 2, axis=1)
    
    for i, ax in enumerate(axes):
        ax.plot(t, G_array[i], label=r"$g_{obs}$", c="k")
        ax.plot(t, f_true[i], label=r"$f_{true}$", lw=3, c="b", alpha=0.5)
        ax.plot(t, f_pred[i], label=r"$f_{rec}$", c="r")
        
        mse_i = mse_array[i]
        ax.text(0.06, 0.90, f"MSE = {mse_i:.3e}",
                transform=ax.transAxes, verticalalignment='top')
        
        ax.set_ylabel(r"$\Theta$")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-4, 6])
        ax.minorticks_on()

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Sample")
    axes[0].set_title(f"R-phi model - MSE: {np.mean(mse_array):.3e}")
    
    print_kv("Val MSE (RPM):", f"{np.mean(mse_array):.3e}")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print_kv(f"Predictions saved", f"{save_path}")


# =============================================================================
# 3. CONFIGURATION & PARAMETERS
# =============================================================================

# Dataset configuration
DATASET_CODE = "lru5_20_D0_1/"  # Change this for different datasets
VAL_FRACTION = 0.1
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Noise configuration - CHOOSE YOUR SCENARIO HERE
NOISE_CONFIG = {
    "noise_level": 0.05,           # 0-1, e.g., 0.05 = 5%
    "noise_type": "white",         # Choose: "white", "correlated", "lowfreq_systematic", "multiplicative"
    "apply_delay": False,          # Add time delay? (set rolling > 0 if True)
    "rolling": 0,                  # Number of samples to roll (delay)
}

# Training configuration
TRAINING_CONFIG = {
    "perform_training": False,
    "n_epochs": 50000,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "scheduler_type": "Cosine",
    "log_every": 100,
    "lambda_smooth": 0,  # Smoothness regularization weight
}


# =============================================================================
# 4. DATA LOADING
# =============================================================================

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
    """Load f and g from HDF5."""
    H5_PATH = f"dataset/{dataset_code}dataset.h5"
    with h5py.File(H5_PATH, "r") as hf:
        data = hf["dataset"][:]  # (N, 2, nt)
    
    f_all = data[:, 0, :].astype(np.float32)
    g_all = data[:, 1, :].astype(np.float32)
    return f_all, g_all


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    print_section("LOADING DATA")
    
    # Load simulation parameters
    L, Tf, nt, D, h_param, l, r, dt = load_simulation_info(DATASET_CODE)
    n = int(2 * nt - 1)
    
    print_kv("Simulation parameters", f"L={L}, Tf={Tf}, nt={nt}, dt={dt:.4e}")
    
    # Load impulse response
    H_np, nk, omegas = load_impulse_response(DATASET_CODE, n, dt)
    print_kv("Impulse response", f"nk={nk}, n={n}")
    
    # Load dataset
    f_all, g_all = load_dataset(DATASET_CODE)
    N = len(f_all)
    print_kv("Dataset", f"N={N}, nt={nt}")
    
    # ========================================================================
    # NOISE APPLICATION - THIS IS THE KEY SECTION
    # ========================================================================
    
    print_section("NOISE CONFIGURATION")
    
    noise_level = NOISE_CONFIG["noise_level"]
    noise_type = NOISE_CONFIG["noise_type"]
    
    print_kv("Noise type", noise_type)
    print_kv("Noise level", f"{noise_level*100:.1f}%")
    
    if noise_level > 0:
        generator = NoiseGenerator(seed=SEED)
        g_all, noise = generator.apply_noise(g_all, noise_level, noise_type)
        print_noise_stats(noise, noise_type, g_all)
    else:
        print_kv("Status", "No noise applied (clean data)")
    
    # Apply delay if needed
    if NOISE_CONFIG["apply_delay"] and NOISE_CONFIG["rolling"] != 0:
        rolling = NOISE_CONFIG["rolling"]
        g_rolled = np.roll(g_all, rolling, axis=1)
        g_rolled[:, :rolling] = 0
        g_all = g_rolled
        print_kv("Time delay applied", f"{rolling} samples = {rolling*dt:.4e} s")
    
    # ========================================================================
    # DATASET SPLIT
    # ========================================================================
    
    print_section("DATASET SPLIT")
    
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(N)
    n_val = max(1, int(VAL_FRACTION * N))
    idx_val = idx[:n_val]
    idx_train = idx[n_val:]
    
    print_kv("Dataset split", f"total={N}, train={len(idx_train)}, val={n_val}")
    print_kv("Device", DEVICE)
    
    G_train = torch.from_numpy(g_all[idx_train]).to(DEVICE)
    F_train = torch.from_numpy(f_all[idx_train]).to(DEVICE)
    G_val = torch.from_numpy(g_all[idx_val]).to(DEVICE)
    F_val = torch.from_numpy(f_all[idx_val]).to(DEVICE)
    
    # ========================================================================
    # MODEL INITIALIZATION
    # ========================================================================
    
    print_section("MODEL INITIALIZATION")
    
    model = LearnedTikhonov(H=H_np, n=n, nt=nt, dt=dt).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print_kv("Parameters", f"phi: {nk}, log_R: {nk} (total: {n_params})")
    
    # ========================================================================
    # TRAINING
    # ========================================================================
    
    if TRAINING_CONFIG["perform_training"]:
        print_section("TRAINING")
        
        history = train(
            model=model,
            G_train=G_train,
            F_train=F_train,
            G_val=G_val,
            F_val=F_val,
            n_epochs=TRAINING_CONFIG["n_epochs"],
            lr=TRAINING_CONFIG["lr"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
            checkpoint_path=f"assets/tikhonov_learned/{DATASET_CODE}best_tikhonov.pt",
            scheduler_type=TRAINING_CONFIG["scheduler_type"],
            log_every=TRAINING_CONFIG["log_every"],
            lambda_smooth=TRAINING_CONFIG["lambda_smooth"],
        )
    else:
        print_kv("Training", "Skipped (perform_training=False)")
        history = None
    
    # ========================================================================
    # DIAGNOSTICS & VALIDATION
    # ========================================================================
    
    print_section("DIAGNOSTICS")
    
    # Load trained model
    ckpt = torch.load(f"assets/tikhonov_learned/{DATASET_CODE}best_tikhonov.pt", 
                     map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to("cpu")
    
    # Initialize solver for comparison
    solver = Heat1DCrankNicolson(L=L, Tf=Tf, nt=nt, D=D, h=h_param, wanted_r=r)
    th = TikhonovHandler(solver)
    
    # Plot diagnostics
    plot_diagnostics(model, omegas, th._build_W(),
                    save_path=f"assets/tikhonov_learned/{DATASET_CODE}diagnostics.pdf")
    
    if history is not None:
        plot_history(history, 
                    save_path=f"assets/tikhonov_learned/{DATASET_CODE}loss_history.pdf")
    
    # Plot predictions
    plot_prediction(model, G_val.cpu(), F_val.cpu(), dt,
                   n_samples=7,
                   save_path=f"assets/tikhonov_learned/{DATASET_CODE}predictions.pdf")
    
    print_section("COMPLETE")
    print_kv("Status", "✓ Analysis complete!")
    print_kv("Outputs", f"assets/tikhonov_learned/{DATASET_CODE}/")
