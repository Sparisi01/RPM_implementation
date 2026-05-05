# tikhonov_handler.py>

from matplotlib.pylab import double
import numpy as np
from src.heat1D_crank_nicolson_solver import Heat1DCrankNicolson
from scipy.optimize import minimize

class TikhonovHandler:
    
    def __init__(self, solver: Heat1DCrankNicolson ):
        self.solver = solver
        self.W = None
    
    def _tikhonov_loss_impulse(self, f: np.ndarray, g_target: np.ndarray, impulse_response: np.ndarray, lambda_reg : float = 1e-3) -> double:

        g_sim = self.solver.solve_from_impulse_response(f,impulse_response)
        
        misfit = np.linalg.norm(g_sim - g_target)**2
        derivate_regularizion = lambda_reg * np.linalg.norm(np.diff(f))**2

        return misfit + derivate_regularizion

    def _tikhonov_loss(self, f: np.ndarray, g_target: np.ndarray, lambda_reg : float = 1e-3) -> double:
        
        T = self.solver.solve(f)
        g_sim = T[-1, :]
        
        misfit = np.linalg.norm(g_sim - g_target)**2
        derivate_regularizion = lambda_reg * np.linalg.norm(np.diff(f))**2

        return misfit + derivate_regularizion

    def regularize_impulse_response(self, g_target:  np.ndarray, lambda_reg: float = 1e-3, maxiter: int = 100, rnd_start_guess_amplitude: float = 0.01) -> np.ndarray:
        
        f_guess = np.random.randn(self.solver.nt) * rnd_start_guess_amplitude
        
        # Compute impulse response
        impulse_response = self.solver.impulse_response()[-1]

        result = minimize(
            self._tikhonov_loss_impulse,
            f_guess,
            args=(g_target, impulse_response, lambda_reg),
            method='L-BFGS-B',
            options={'maxiter': maxiter}
        )
        
        return result.x
    
    def regularize_nicholson(self, g_target: np.ndarray, lambda_reg: float = 1e-3, maxiter: int = 100, rnd_start_guess_amplitude: float = 0.01) -> np.ndarray:
        
        f_guess = np.random.randn(self.solver.nt) * rnd_start_guess_amplitude

        result = minimize(
            self._tikhonov_loss,
            f_guess,
            args=(g_target, lambda_reg),
            method='L-BFGS-B',
            options={'maxiter': maxiter}
        )

        return result.x
    
    def regularize_closed_form(self, g_target: np.ndarray, lambda_reg: float = 1e-3, eps: float = 1e-6) -> np.ndarray:
        
        n = (2*self.solver.nt - 1)

        if self.W is None:
            
            impulse_response = self.solver.impulse_response()[-1]
            H = np.fft.rfft(impulse_response, n)
        
            omegas = np.fft.rfftfreq(n, d=self.solver.dt) * 2 * np.pi
            
            # Operator Solution to the Tickonov minimization problem
            self.W = np.conjugate(H) / (np.abs(H)**2 + lambda_reg*omegas**2 + eps)
        
        g_trap = np.empty_like(g_target)
        g_trap[:-1] = (g_target[:-1] + g_target[1:]) / 2
        g_trap[-1] = g_target[-1] / 2
                
        G = np.fft.rfft(g_trap, n)
        
        F = (self.W * G) / self.solver.dt

        f_full = np.fft.irfft(F, n)
        
        return f_full[:self.solver.nt] 
