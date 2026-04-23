import numpy as np
from scipy.linalg import lu_factor, lu_solve

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

        self.A = np.zeros((self.nx, self.nx))
        self.B = np.zeros((self.nx, self.nx))

        self._build_matrices()
        self.lu, self.piv = lu_factor(self.A)

    def _build_matrices(self):
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

        # Boundary destra
        T[0, 0] = f[0]

        for n in range(1, self.nt):
            b = self.B @ T[:, n-1]
            #b[0] = f[n]  # Dirichlet
            b[0] = f[n]
            T[:, n] = lu_solve((self.lu, self.piv), b)

        return T

    def info(self):
        print("Time scale coeff:", self.L * self.L / self.D)
        print("dx:", self.dx)
        print("dt:", self.dt)
        print("r:", self.r)