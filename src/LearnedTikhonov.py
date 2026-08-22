# LearnedTikhonov.py>

import numpy as np
from sympy import Float
import torch
import torch.nn as nn


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

        freq = torch.fft.rfftfreq(n, d=dt)
        omega = 2.0 * torch.pi * freq

        R0 = lambda_tik * omega.pow(2)
        R0 = torch.clamp(R0, min=eps_R)
    
        self.phi_raw = nn.Parameter(torch.zeros(nk))
        #self.log_R   = nn.Parameter(torch.full((nk,), np.log(1e-2)))
        self.log_R = nn.Parameter(torch.log(R0))

    def _get_phi(self) -> torch.Tensor:
        """Forza phi[0]=0 (DC). Se n pari forza anche phi[-1]=0 (Nyquist)."""
        """phi = self.phi_raw.clone() 
        phi[0] = 0.0
        if self.n_even:
            phi[-1] = 0.0"""
        return torch.pi * torch.tanh(self.phi_raw)
        #phi = 2 * torch.pi * torch.sigmoid(self.phi_raw)
        #return phi                                       # (nk,) float32

    def _get_R(self) -> torch.Tensor:
        return torch.exp(self.log_R)                            # (nk,) float32, sempre > 0

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
