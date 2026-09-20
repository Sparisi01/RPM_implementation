# LearnedTikhonov.py>

import numpy as np
from sympy import Float
import torch
import torch.nn as nn
from typing import Literal, get_args, Tuple
from src.print_utils import print_kv, print_section

class LearnedTikhonov(nn.Module):
    """
    Buffers (fissi):
      H_abs2  : (nk,) float32
      H_conj  : (nk,) complex64

    Parametri appresi:
      phi_raw : (nk) float32
      log_R   : (nk) float32 
    """

    def __init__(self, H: np.ndarray, n: int, nt: int, dt: float, lambda_tik: float = 1e-2):
        super().__init__()
        self.n  = n
        self.nt = nt
        self.dt = dt
        nk = len(H)
        self.nk = nk
        self.n_even = (n % 2 == 0)
        
        H_t = torch.from_numpy(H.astype(np.complex64))
        self.register_buffer("H_abs2", H_t.abs().pow(2))   # float32
        self.register_buffer("H_conj", H_t.conj())          # complex64

        # --------------------------------------------------
        # Inizializzazione Tikhonov standard:
        # R(omega) = lambda * omega^2
        # --------------------------------------------------
        eps_R = 1e-12

        self.freq = torch.fft.rfftfreq(n, d=dt)
        self.omegas = 2.0 * torch.pi * self.freq

        R0 = lambda_tik * self.omegas.pow(2)
        R0 = torch.clamp(R0, min=eps_R) + self.H_abs2
    
        self.phi_raw = nn.Parameter(torch.zeros(nk))
        #self.log_R   = nn.Parameter(torch.full((nk,), np.log(1e-2)))
        self.log_R = nn.Parameter(torch.log(R0))
        #self.log_R = nn.Parameter(torch.arcsinh(R0))
        
        # --------------------------------------------------
        # Mask degli elementi allenabili
        # --------------------------------------------------
        self.phi_mask = torch.zeros(nk, dtype=torch.bool)
        self.R_mask   = torch.zeros(nk, dtype=torch.bool)
    
        # Gradiente nullo sugli elementi congelati
        self.phi_raw.register_hook(
            lambda grad: grad * self.phi_mask.to(grad.device)
        )

        self.log_R.register_hook(
            lambda grad: grad * self.R_mask.to(grad.device)
        )


    def _get_phi(self) -> torch.Tensor:
        """Forza phi[0]=0 (DC). Se n pari forza anche phi[-1]=0 (Nyquist)."""
        
        phi = self.phi_raw.clone() 
        phi[0] = 0.0
        if self.n_even:
            phi[-1] = 0.0
            
        return torch.pi * torch.tanh(self.phi_raw)
        #phi = 2 * torch.pi * torch.sigmoid(self.phi_raw)
        #return phi                                       # (nk,) float32

    def _get_R(self) -> torch.Tensor:
        return torch.exp(self.log_R) - self.H_abs2                           # (nk,) float32, sempre > 0
        #return torch.sinh(self.log_R) 

    def _build_W(self) -> torch.Tensor:
        """Costruisce il filtro W (nk,) complex64."""
        phi = self._get_phi()
        R   = self._get_R()
        # torch.polar: device-safe, niente letterali Python complessi
        exp_i_phi = torch.polar(torch.ones_like(phi), phi)      # (nk,) complex64
        return self.H_conj * exp_i_phi / (self.H_abs2 + R + 1e-12)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        """
        g : (N, nt) float32  — segnale al bordo sinistro
        Ritorna f_pred : (N, nt) float32
        """
        # Integrazione trapezoidale
        g_trap = g.clone()
        g_trap[:, :-1] = (g[:, :-1] + g[:, 1:]) / 2.0
        g_trap[:,  -1] = g[:, -1] / 2.0

        G      = torch.fft.rfft(g_trap, n=self.n, dim=1)        # (N, nk)
        W      = self._build_W()                                # (nk,)
        F_pred = G * W.unsqueeze(0)                             # (N, nk)

        f_pred = torch.fft.irfft(F_pred, n=self.n, dim=1)[:, :self.nt]
        return f_pred / self.dt

    # ------------------------------------------------------------------
    # Utilità diagnostiche
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_phi(self) -> np.ndarray:
        return self._get_phi().cpu().numpy()

    @torch.no_grad()
    def get_R(self) -> np.ndarray:
        return self._get_R().cpu().numpy()

    @torch.no_grad()
    def get_W_magnitude(self) -> np.ndarray:
        return self._build_W().abs().cpu().numpy()

    def freeze_phi(self):
            self.phi_raw.requires_grad_(False)
    
    def unfreeze_phi(self):
        self.phi_raw.requires_grad_(True)

    def freeze_R(self):
        self.log_R.requires_grad_(False)

    def unfreeze_R(self):
        self.log_R.requires_grad_(True)
    
    def set_trainable_range(self, start, end, phi=True, R=True):
        if phi:
            self.phi_mask[:] = False
            self.phi_mask[start:end] = True

        if R:
            self.R_mask[:] = False
            self.R_mask[start:end] = True
    
    def set_tainable_range_by_omegas(self, min_omega, max_omega):
        self.phi_mask = (self.omegas >= min_omega) & (self.omegas <= max_omega)
        self.R_mask   = (self.omegas >= min_omega) & (self.omegas <= max_omega)

def smoothness_regularization(model: LearnedTikhonov) -> torch.Tensor:
    """Penalize sharp variations in R and Phi"""
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
    scheduler_type: _TYPES = "Cosine",
    min_mse_index: int = 0,
    max_mse_index: int = -1,
    history = None
):
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
    
    if history is None:
        history = {"train_loss": [], "val_loss": [], "lr": []}
    
    best_val = float("inf")

    for epoch in range(1, n_epochs + 1):
        # Training Step
        model.train()
        optimizer.zero_grad()
        
        pred_train = model(G_train)
        mse_train = loss_fn(pred_train[:,min_mse_index:max_mse_index], F_train[:,min_mse_index:max_mse_index])
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
            mse_val = loss_fn(pred_val[:,min_mse_index:max_mse_index], F_val[:,min_mse_index:max_mse_index])
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

    print_kv(f"Best val MSE", f"{best_val:.4e}")
    print_kv(f"Saved to", f"{checkpoint_path}")
    
    return history, best_val