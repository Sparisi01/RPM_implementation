import numpy as np
from scipy.linalg import lu_factor, lu_solve
from src.print_utils import print_kv, print_section

class Heat1DCrankNicolson:

    def __init__(self, L, Tf, nt, D, h, wanted_r = 0.5):
        self.L = L
        self.Tf = Tf
        self.nt = nt
        self.D = D
        self.wanted_r = wanted_r
        self.h = h

        self.dt = Tf / nt
        self.dx = np.sqrt(D * self.dt / wanted_r)
        self.nx = int(L / self.dx) + 1
        self.dx = L / (self.nx - 1)

        self.r = D * self.dt / (self.dx ** 2)

        self.x_array = np.linspace(0, L, self.nx)
        self.t_array = np.linspace(0, Tf, nt)

        self._build_matrices()
    '''
    def _build_matrices(self):
        self.A = np.zeros((self.nx, self.nx))
        self.B = np.zeros((self.nx, self.nx))
        
        # Nodi interni
        for i in range(1, self.nx - 1):
            self.A[i, i-1] = -self.r / 2
            self.A[i, i]   = 1 + self.r
            self.A[i, i+1] = -self.r / 2

            self.B[i, i-1] = self.r / 2
            self.B[i, i]   = 1 - self.r
            self.B[i, i+1] = self.r / 2

        # Boundary sinistra (Dirichlet)
        self.A[0, 0] = 1
        self.B[0, :] = 0

        # Boundary destra (Robin)
        self.A[-1, -2] = -2 * self.r
        self.A[-1, -1] = 1 + 2 * self.r * (1 + self.h * self.dx)

        self.B[-1, -2] = 2 * self.r
        self.B[-1, -1] = 1 - 2 * self.r * (1 + self.h * self.dx)

    def solve(self, f):
        T = np.zeros((self.nx, self.nt))

        if(np.size(f) != self.nt):
            print("ERORR: Size of f must be the same as nt")
            return 0

        # Condizione iniziale
        T[:, 0] = 0

        # Boundary sinistro
        T[0, 0] = f[0]

        for n in range(1, self.nt):
            # Compute B*Theta
            b = self.B @ T[:, n-1]
            # Summ the vector b
            # b[0] = (f[n] + f[n-1])/2
            b[0] = f[n]
        
            T[:, n] = lu_solve((self.lu, self.piv), b)

        return T
    '''
    def _build_matrices(self):
        
        n = self.nx - 1
        
        self.A = np.zeros((n, n))
        self.B = np.zeros((n, n))
        
        # Nodi interni: i = 1...nx-1
        for i in range(n - 1):
            self.A[i, i]   =  1 + self.r
            self.B[i, i]   =  1 - self.r
            if i > 0:
                self.A[i, i-1] = -self.r / 2
                self.B[i, i-1] =  self.r / 2
            if i < n - 1:
                self.A[i, i+1] = -self.r / 2
                self.B[i, i+1] =  self.r / 2
        
        # Bordo Robin
        alpha = 2 * self.r

        self.A[-1, -2] = -alpha / 2
        self.A[-1, -1] = 1 + alpha/2 * (1 + self.h * self.dx)

        self.B[-1, -2] = alpha / 2
        self.B[-1, -1] = 1 - alpha/2 * (1 + self.h * self.dx)
        
        self.lu, self.piv = lu_factor(self.A)

    def solve(self, f):
        n = self.nx - 1
        T_inner = np.zeros((n, self.nt))
        T = np.zeros((self.nx, self.nt))
        
        # Condizione iniziale zero, bordo sinistro al tempo 0
        T[0, 0] = f[0]
        
        for step in range(1, self.nt):
            b = self.B @ T_inner[:, step-1]
            
            # Contributo bordo sinistro: forma trapezoidale
            b[0] += (self.r / 2) * (f[step-1] + f[step])
            
            T_inner[:, step] = lu_solve((self.lu, self.piv), b)
        
        # Ricostruisci soluzione completa
        T[0, :]  = f
        T[1:, :] = T_inner
        
        return T
    
    def solve_from_impulse_response(self, f, impulse_response):
        
        n = (2*self.nt - 1)
        
        f_trap = np.empty_like(f)
        f_trap[:-1] = (f[:-1] + f[1:]) / 2
        f_trap[-1] = f[-1] / 2

        F = np.fft.rfft(f_trap, n)
        H = np.fft.rfft(impulse_response, n)
        
        if F.shape != H.shape:
            raise ValueError("Frequency-domain shapes do not match")
    
        return np.fft.irfft(F * H,n)[:self.nt] * self.dt   
    
    def impulse_response(self):

        # Definisci impulso
        impulse = np.zeros(self.nt)
        impulse[0] = 2/self.dt
        
        # Solve impulse response
        response = self.solve(impulse)

        return response
    
    
    def info(self):
        print_section(f"Solver info")
        print_kv(f"Time scale coeff:", f"{self.L * self.L / self.D:.2e}")
        print_kv(f"dx:              ", f"{self.dx:.2e}")
        print_kv(f"dt:              ", f"{self.dt:.2e}")
        print_kv(f"nt:              ", f"{self.nt:.0f}")
        print_kv(f"nx:              ", f"{self.nx:.0f}")
        print_kv(f"r:               ", f"{self.r:.2e}")
        print_kv(f"h:               ", f"{self.h:.2e}")