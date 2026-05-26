import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline


# Classe do Model de Ising
class IsingModel:

    def __init__(self, L, T):
        # parametros do modelo
        self.L = L
        self.T = T

        # array com os spins
        self.spins = np.ones((L, L))

        # arrays para guardar evolução das variáveis
        self.energies = []
        self.magnetizations = []

    def calc_ener_spin(self, i, j):
        energy = 0
        return energy

    def calc_ener(self):
        # calcular a energia por spin do sistema
        # e = E / L^2
        # ...
        energy = 0
        return energy

    def calc_mag(self):
        # calcular a magnetização por spin do sistema
        # m = M / L^2
        # ...
        mag = 0
        return mag

    def iter_monte_carlo(self, n_iter):
        # iterar com o método de Metropolis Hastings
        for i in tqdm(range(n_iter), desc=f"L={self.L:6d}, T={self.T:8f}"):
            # ...
            pass

    @property
    def energy(self):
        # usa para aceder ao array com as energias
        return np.array(self.energies)

    @property
    def magnetization(self):
        # usa para aceder ao array com as magnetizações
        return np.array(self.magnetizations)