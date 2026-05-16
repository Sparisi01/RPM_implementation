# tikhonov_handler.py>

import numpy as np
from src.heat1D_crank_nicolson_solver import Heat1DCrankNicolson
from scipy.optimize import minimize
from typing import Literal, get_args

_TYPES = Literal["closed", "fft", "solver", "mixed"]

class TikhonovHandler:
    
    def __init__(self, solver: Heat1DCrankNicolson):
        self.solver = solver
        
    def _loss_impulse(
        self, 
        f: np.ndarray, 
        g_target: np.ndarray,
        impulse_response: np.ndarray,
        lambda_reg : float = 1e-3
        ) -> float:
        
        g_sim = self.solver.solve_from_impulse_response(f, impulse_response)
        
        misfit = np.linalg.norm(g_sim - g_target)**2
        derivative_regularization = lambda_reg * np.linalg.norm(np.diff(f))**2
        
        return misfit + derivative_regularization

    
    def _loss_standard(
        self, f: np.ndarray,
        g_target: np.ndarray, 
        lambda_reg: float = 1e-3
        ) -> float:
        
        g_sim = self.solver.solve(f)[-1, :]
        
        misfit = np.linalg.norm(g_sim - g_target)**2
        derivative_regularization = lambda_reg * np.linalg.norm(np.diff(f))**2
        
        return misfit + derivative_regularization

    def _build_W(
        self, 
        lambda_reg: float = 1e-3
        ) -> np.ndarray:
        
        n = (2*self.solver.nt - 1)
        
        impulse_response = self.solver.impulse_response()[-1]
        H = np.fft.rfft(impulse_response, n)
    
        omegas = np.fft.rfftfreq(n, d=self.solver.dt) * 2 * np.pi
        
        D2 = np.abs(1 - np.exp(-1j * omegas * self.solver.dt))**2
        
        #return np.conjugate(H) / (np.abs(H)**2 + lambda_reg*omegas**2)
        
        eps = 1e-14
        
        return np.conjugate(H) / (np.abs(H)**2 + lambda_reg*D2 + eps)
            
    def regularize_impulse(
        self,
        g_target:  np.ndarray,
        lambda_reg: float = 1e-3,
        maxiter: int = 100,
        rnd_start_guess_amplitude: float = 0.01
        ) -> np.ndarray:
        
        f_guess = np.random.randn(self.solver.nt) * rnd_start_guess_amplitude
        
        # Compute impulse response
        impulse_response = self.solver.impulse_response()[-1]

        result = minimize(
            self._loss_impulse,
            f_guess,
            args=(g_target, impulse_response, lambda_reg),
            method='L-BFGS-B',
            options={'maxiter': maxiter}
        )
        
        return result.x
    
    def regularize_standard(
        self,
        g_target: np.ndarray, 
        lambda_reg: float = 1e-3, 
        maxiter: int = 100, 
        rnd_start_guess_amplitude: float = 0.01
        ) -> np.ndarray:
        
        f_guess = np.random.randn(self.solver.nt) * rnd_start_guess_amplitude

        result = minimize(
            self._loss_standard,
            f_guess,
            args=(g_target, lambda_reg),
            method='L-BFGS-B',
            options={'maxiter': maxiter}
        )

        return result.x
    
    def regularize_closed_form(
        self, 
        g_target: np.ndarray, 
        lambda_reg: float = 1e-3
        ) -> np.ndarray:
        
        n = (2*self.solver.nt - 1)

        g_trap = np.empty_like(g_target)
        g_trap[:-1] = (g_target[:-1] + g_target[1:]) / 2
        g_trap[-1] = g_target[-1] / 2
                
        G = np.fft.rfft(g_trap, n)
        W = self._build_W(lambda_reg)
         
        F = (W * G) / self.solver.dt

        f_full = np.fft.irfft(F, n)
        
        return f_full[:self.solver.nt] 
    
    def regularize(
        self, 
        g_target: np.ndarray, 
        lambda_reg: float = 1e-3, 
        regularization_type_: _TYPES = "closed", 
        maxiter: int = 100, 
        rnd_start_guess_amplitude: float = 0.01
        ) -> np.ndarray:
        
        options = get_args(_TYPES)
        assert regularization_type_ in options, f"'{regularization_type_}' is not in {options}"
    
        if regularization_type_ == "closed":
            return self.regularize_closed_form(g_target, lambda_reg)
        elif regularization_type_ == "fft":
            return self.regularize_impulse(g_target, lambda_reg, maxiter, rnd_start_guess_amplitude)
        elif regularization_type_ == "solver":
            return self.regularize_standard(g_target, lambda_reg, maxiter, rnd_start_guess_amplitude)
        elif regularization_type_ == "mixed":
            raise NotImplementedError("mixed regularization not implemented")
        

import matplotlib.pyplot as plt

def reconstruction_error_plot(f,g,f_rec, t_array, n_cut = None):
    
    nt = np.size(t_array)
    
    if n_cut is None or n_cut == 0:
        n_cut = 1
    
    fig, axes = plt.subplots(2,1, figsize=(4.4,4.4)) 
    ax1, ax2 = axes
    
    ax1.plot(t_array, f, label=r"$f(t)$", lw=5, alpha=0.6)
    ax1.plot(t_array, f_rec, label=r"Reconstructed $\hat{f}(t)$", c="r")
    ax1.plot(t_array, g, label=r"$\hat{g}(t)$", c="k")
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel("Temperature")
    ax1.set_title(r"Real $f(t)$ vs Reconstructed $\hat{f}(t)$")
    ax1.grid()
    # ax1.ylim([np.min(f_reconstructed[:-n_eff])*1.2, np.max(f_reconstructed[:-n_eff])*1.2])
    ax1.legend()
    ax1.minorticks_on()
    
    loss_array = np.zeros(nt)

    for n in range(nt):
        loss_array[n] = np.linalg.norm(f[0:n] - f_rec[0:n])**2

    ax2.plot(t_array[:-n_cut], loss_array[:-n_cut], label="Cumulative error")
    ax2.set_ylabel(r"$||\hat{f} - f||^2, \hspace{1em} t \in [0,t_f)$")
    ax2.set_xlabel(r"Cumulate time $t_f$")
    ax2.set_title(r"Cumulative reconstruction error as a function of $t_f$")
    ax2.legend()
    ax2.grid()
    
    return loss_array, fig, axes