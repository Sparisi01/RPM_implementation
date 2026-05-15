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