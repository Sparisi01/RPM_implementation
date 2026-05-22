import torch
import torch.nn as nn
import numpy as np

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

            diag_re (nn.Parameter):
                Real part of upper diagonals with shape (M, nk).

            diag_im (nn.Parameter):
                Imaginary part of upper diagonals with shape (M, nk).
        """
        
        super().__init__()
        self.nk = nk
        self.M = M
        
        # Diagonale principale: reale (per simmetria Sigma(w) = Sigma(-w))
        self.diag_0 = nn.Parameter(torch.zeros(nk))
        
        if M > 0:
            self.diag_re = nn.Parameter(torch.zeros(M, nk))  # parte reale
            self.diag_im = nn.Parameter(torch.zeros(M, nk))  # parte immaginaria

        #nn.init.normal_(self.diag_0, mean=0.0, std=0.01)
        #if M > 0:
        #    nn.init.normal_(self.diag_re, mean=0.0, std=0.01)
        #    nn.init.normal_(self.diag_im, mean=0.0, std=0.01)
    
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
        
        # Scala i valori della diagonale in (-1, 1) usando la tangente iperbolica
        d0 = torch.tanh(self.diag_0)
        Sigma = torch.diag(d0.to(torch.complex64))
        
        for m in range(1, M + 1):
            d = torch.complex(self.diag_re[m-1], self.diag_im[m-1])
            # sopra la diagonale: sigma_m
            idx = torch.arange(nk - m)
            Sigma[idx, idx + m] += d[:nk-m]
            # sotto la diagonale: conj(sigma_m) per hermitianità
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
                new_sigma.diag_re[:old_M].copy_(old_sigma.diag_re)
                new_sigma.diag_im[:old_M].copy_(old_sigma.diag_im)
            # Nuove diagonali inizializzate a zero
        
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
    
    def freeze_sigma(self):
        self.diag_0.requires_grad_(False)
        if self.M > 0:
            self.diag_re.requires_grad_(False)
            self.diag_im.requires_grad_(False)

    def unfreeze_sigma(self):
        self.diag_0.requires_grad_(True)
        if self.M > 0:
            self.diag_re.requires_grad_(True)
            self.diag_im.requires_grad_(True)
    
    # ==========================================================
    # Utilities printing informations
    # ==========================================================
    
    def __repr__(self):
        nparams = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return (
            f"{self.__class__.__name__}(\n"
            f"  nk={self.nk},\n"
            f"  M={self.M},\n"
            f"  parameters={nparams:,},\n"
            f"  trainable={trainable:,}\n"
            f")"
        )

    def __str__(self):
        return self.__repr__()
    
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
        
        self.sigma = BandedSigma(self.nk, M)                    # (nk, M) float32

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

        # Correzione IC dalla finestra precedente
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
        for p in self.sigma.parameters():
            p.requires_grad_(False)

    def unfreeze_sigma(self):
        for p in self.sigma.parameters():
            p.requires_grad_(True)
            
    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad_(True)
    
    # ==========================================================
    # Utilities printing informations
    # ==========================================================
    
    def __repr__(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        sigma_params = sum(p.numel() for p in self.sigma.parameters())

        return (
            f"{self.__class__.__name__}(\n"
            f"  nt                      =   {self.nt},\n"
            f"  nk                      =   {self.nk},\n"
            f"  n                       =   {self.n},\n"
            f"  dt                      =   {self.dt},\n"
            f"  sigma_bandwidth         =   {self.sigma.M},\n"
            f"  sigma_parameters        =   {sigma_params:,},\n"
            f"  total_parameters        =   {total_params:,},\n"
            f"  trainable_parameters    =   {trainable_params:,}\n"
            f")"
        )

    def __str__(self):
        return self.__repr__()