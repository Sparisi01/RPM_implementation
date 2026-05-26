import torch
import torch.nn as nn
import numpy as np

from src.print_utils import print_kv, print_line

class BandedSigma(nn.Module):
    def __init__(self, nk:int, M=0):
        """
        Hermitian banded operator in the frequency domain.

        The matrix Sigma is parameterized through its diagonals up to
        bandwidth `M`, enforcing Hermitian symmetry:

            Sigma[i, j] = conj(Sigma[j, i])

        Only the main diagonal and the upper diagonals are explicitly
        learned. Lower diagonals are obtained by complex conjugation.

        Args:
            nk (int):
                Number of frequencies.

            M (int, optional):
                Number of upper/lower off-diagonals.
                - `M = 0` corresponds to a purely diagonal operator.
                - `M > 0` enables frequency coupling.

        Attributes:
            nk (int):
                Number of frequencies.

            M (int):
                Bandwidth of the operator.

            diag_0 (nn.Parameter):
                Real-valued main diagonal.

            diag_re (nn.ParameterList):
                Real part of upper diagonals with shape (M, nk).

            diag_im (nn.ParameterList):
                Imaginary part of upper diagonals with shape (M, nk).
        """
        
        super().__init__()
        self.nk = nk
        self.M = M
        
        self.diag_0 = nn.Parameter(torch.zeros(nk))                 # (nk) real
        
        if M > 0:
            self.diag_re = nn.ParameterList(
                [nn.Parameter(torch.zeros(nk)) for _ in range(M)]   # (M, nk) complex
            )   
            self.diag_im = nn.ParameterList(
                [nn.Parameter(torch.zeros(nk)) for _ in range(M)]   # (M, nk) complex
            )
    
    def forward(self, G_prev):
        """
        Args:
            g_hat (torch.Tensor): 
                Input signal from the previous windows of size (n_samples, nk)
        """
        return  G_prev @ self._build_Sigma()
    
    def _build_Sigma(self):
        """
        Construct the Hermitian banded matrix Sigma.

        The main diagonal is constrained to (-1, 1)
        through a hyperbolic tangent nonlinearity.

        Returns:
            torch.Tensor:
                Complex Hermitian matrix with shape (nk, nk).
        """
        
        nk, M = self.nk, self.M
        
        # Scale values on diagonal in (-1, 1) using tanh
        d0 = torch.tanh(self.diag_0)
        Sigma = torch.diag(d0.to(torch.complex64))
        
        for m in range(1, M + 1):
            d = torch.complex(self.diag_re[m-1], self.diag_im[m-1])
            # over the diagonal: sigma_m
            idx = torch.arange(nk - m)
            Sigma[idx, idx + m] += d[:nk-m]
            # under diagonal: conj(sigma_m) for hermitianity
            Sigma[idx + m, idx] += d[:nk-m].conj()
        
        return Sigma
    
    @staticmethod
    def increase_bandwidth(model, new_M):
        """
        Increase the bandwidth of the operator while preserving
        previously learned parameters.

        Existing diagonals are copied into the new operator,
        while newly introduced diagonals are initialized to zero.

        Args:
            model:
                Model containing a `sigma` attribute of type
                `BandedSigma`.

            new_M (int):
                New bandwidth value.

        Returns:
            The updated model.
        """
        
        old_sigma = model.sigma
        old_M = old_sigma.M
        device = old_sigma.diag_0.device
        
        new_sigma = BandedSigma(old_sigma.nk, new_M).to(device)
        
        # Copia parametri esistenti
        with torch.no_grad():
            new_sigma.diag_0.copy_(old_sigma.diag_0)
            if old_M > 0:
                for m in range(old_M):
                    new_sigma.diag_re[m].copy_(old_sigma.diag_re[m])
                    new_sigma.diag_im[m].copy_(old_sigma.diag_im[m])
        
        model.sigma = new_sigma
        return model
    
    # ==========================================================
    # Utilities Get Parameters
    # ==========================================================
    
    @torch.no_grad()
    def get_Sigma(self) -> np.ndarray:
        return self._build_Sigma().cpu().numpy()
    
    # ==========================================================
    # Utilities freeze / unfreeze
    # ==========================================================
    
    def freeze_diagonal(self, diagonal_index: int):
        if diagonal_index < 0 or diagonal_index > self.M:
            raise ValueError(f"diagonal_index must be in [0, M={self.M}], got {diagonal_index}")
        if diagonal_index == 0:
            self.diag_0.requires_grad_(False)
        else:
            self.diag_re[diagonal_index - 1].requires_grad_(False)
            self.diag_im[diagonal_index - 1].requires_grad_(False)

    def unfreeze_diagonal(self, diagonal_index: int):
        if diagonal_index < 0 or diagonal_index > self.M:
            raise ValueError(f"diagonal_index must be in [0, M={self.M}], got {diagonal_index}")
        if diagonal_index == 0:
            self.diag_0.requires_grad_(True)
        else:
            self.diag_re[diagonal_index - 1].requires_grad_(True)
            self.diag_im[diagonal_index - 1].requires_grad_(True)
        
    def freeze_sigma(self):
        self.diag_0.requires_grad_(False)
        if self.M > 0:
            for p in self.diag_re:
                p.requires_grad_(False)
            for p in self.diag_im:
                p.requires_grad_(False)

    def unfreeze_sigma(self):
        self.diag_0.requires_grad_(True)
        if self.M > 0:
            for p in self.diag_re:
                p.requires_grad_(True)
            for p in self.diag_im:
                p.requires_grad_(True)   
    
class InverseHeatModel(nn.Module):
    
    def __init__(self, 
                H:np.ndarray,   # Impulse response 
                nt: int,        # Number of time steps
                dt: float,      # Time step interval
                M: int=0,       # Number of off diagonals in Sigma
                ):
        
        super().__init__()
        
        self.nt = nt
        self.nk = len(H) 
        self.n  = int(2 * self.nt - 1)
        
        self.dt = dt
        self.n_even = (self.n % 2 == 0)
        
        self.sigma = BandedSigma(self.nk, M)

        H_t = torch.from_numpy(H.astype(np.complex64))
        self.register_buffer("H_abs2", H_t.abs().pow(2))        # float32
        self.register_buffer("H_conj", H_t.conj())              # complex64

        self.phi_raw = nn.Parameter(torch.zeros(self.nk))
        self.log_R   = nn.Parameter(torch.full((self.nk,), np.log(1e-2)))
    
    def forward(self, g : torch.Tensor, g_prev : torch.Tensor) -> torch.Tensor:
        
        # Trapezoidal integration correction
        g_trap              = g.clone()
        g_trap[:, :-1]      = (g[:, :-1] + g[:, 1:]) / 2.0
        g_trap[:,  -1]      = g[:, -1] / 2.0

        # Trapezoidal integration correction
        g_trap_prev         = g_prev.clone()
        g_trap_prev[:, :-1] = (g_prev[:, :-1] + g_prev[:, 1:]) / 2.0
        g_trap_prev[:,  -1] = g_prev[:, -1] / 2.0

        # Correction term
        G_prev              = torch.fft.rfft(g_prev, n=self.n, dim=1)                       # (N, nk)
        G_correction        = self.sigma(G_prev)                                            # (N, nk)
        
        G                   = torch.fft.rfft(g_trap, n=self.n, dim=1)                       # (N, nk)
        W                   = self._build_W()  
        
        F_pred              = W.unsqueeze(0) * (G-G_correction)                             # (N, nk)
        f_pred              = torch.fft.irfft(F_pred, n=self.n, dim=1)[:, :self.nt]
        
        return f_pred / self.dt
    
    def _get_phi(self) -> torch.Tensor:
        """phi[0]=0 (DC). If n even phi[-1]=0 (Nyquist)"""
        phi = self.phi_raw.clone()
        phi[0] = 0.0
        if self.n_even:
            phi[-1] = 0.0
        return phi                                              # (nk,) float32

    def _get_R(self) -> torch.Tensor:
        return torch.exp(self.log_R)                            # (nk,) float32, always > 0
    
    def _build_W(self) -> torch.Tensor:
        """Costruisce il filtro W (nk,) complex64."""
        
        phi = self._get_phi()
        R   = self._get_R()
        
        # torch.polar: device-safe
        exp_i_phi = torch.polar(torch.ones_like(phi), phi)      # (nk,) complex64
        
        return self.H_conj * exp_i_phi / (self.H_abs2 + R + 1e-12)
    
    # ==========================================================
    # Utilities Get Parameters
    # ==========================================================
    
    @torch.no_grad()
    def get_phi(self) -> np.ndarray:
        return self._get_phi().cpu().numpy()

    @torch.no_grad()
    def get_R(self) -> np.ndarray:
        return self._get_R().cpu().numpy()

    @torch.no_grad()
    def get_W_magnitude(self) -> np.ndarray:
        return self._build_W().abs().cpu().numpy()
    
    # ==========================================================
    # Utilities freeze / unfreeze
    # ==========================================================
    
    def freeze_phi(self):
        self.phi_raw.requires_grad_(False)

    def unfreeze_phi(self):
        self.phi_raw.requires_grad_(True)

    def freeze_R(self):
        self.log_R.requires_grad_(False)

    def unfreeze_R(self):
        self.log_R.requires_grad_(True)
        
    def freeze_sigma(self):
        self.sigma.freeze_sigma()

    def unfreeze_sigma(self):
        self.sigma.unfreeze_sigma()
            
    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad_(True)
    
    # ==========================================================
    # Utilities printing informations
    # ==========================================================
        
    def summary(self):
        """
        """
        nk, M = self.nk, self.sigma.M

        print_line(char="=")
        print_kv(f"Structure",f"nk={nk}  M={M}")
        print_line()

        total_params     = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params    = total_params - trainable_params

        print_kv(f"Total Parameters",   f"{total_params:>8,}")
        print_kv(f"Trainable",          f"{trainable_params:>8,}")
        print_kv(f"Frozen",             f"{frozen_params:>8,}")
        print_line()

        # -- Log R -----------------------------------------
        log_R_learn = self.log_R.requires_grad
        log_R_icon  = "Learn" if log_R_learn else "Frozen"
        print_kv("Log_R",log_R_icon)
        
        # -- Phi -------------------------------------------
        phi_raw_learn = self.phi_raw.requires_grad
        phi_raw_icon  = "Learn" if phi_raw_learn else "Frozen"
        print_kv("Phi",phi_raw_icon)
                
        # -- Sigma Diagonal --------------------------------
        d0_learn = self.sigma.diag_0.requires_grad
        d0_icon  = "Learn" if d0_learn else "Frozen"
        print_kv("Diagonal",d0_icon)

        # -- Sigma Off diagonal ----------------------------
        if M > 0:
            for m in range(1, M + 1):
                re = self.sigma.diag_re[m - 1]
                im = self.sigma.diag_im[m - 1]

                re_learn = re.requires_grad
                im_learn = im.requires_grad
                shape    = str(tuple(re.shape))

                if re_learn and im_learn:
                    status = "Learn"
                elif not re_learn and not im_learn:
                    status = "Frozen"

                print_kv(f"m={m}",status)
               
        # -- Sigma sparsity -------------------------------- 
        n_elem_nonzero = nk + 2 * sum(nk - m for m in range(1, M + 1))
        n_elem_total   = nk * nk
        sparsity       = 1.0 - n_elem_nonzero / n_elem_total
        
        print_line()
        print_kv(f"Non-zero elements ",f"{n_elem_nonzero:>8,}/{n_elem_total:,}")
        print_kv(f"Sparsity",f"{sparsity:>7.1%}")
        
        print_line(char="=")