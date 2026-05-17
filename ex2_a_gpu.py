import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
from numba import njit

# ── Constantes do modelo ─────────────────────────────────────────────────────
J  = 1
h  = 0
kb = 1
rnd = np.random.default_rng(seed=1)


@njit(cache=True, fastmath=True)
def _metropolis_sweep(spins, n_iter, T, J, h,
                      i_vals, j_vals, u_vals,
                      energies, magnetizations,
                      E0, mag0):
    L = spins.shape[0]
    inv_L2 = 1.0 / (L * L)
    E = E0
    mag = mag0
    energies[0] = E
    magnetizations[0] = mag

    # tabela de Boltzmann: só há 5 valores possíveis de sumviz (-4,-2,0,2,4)
    # e own ∈ {-1,+1}. Pré-calculamos exp(-delta/T) indexado por (own+1, sumviz+4).
    exp_table = np.empty((2, 9))
    for own_idx in range(2):
        own_val = -1.0 if own_idx == 0 else 1.0
        for s_idx in range(9):
            sumviz = s_idx - 4.0
            delta = 2.0 * own_val * (J * sumviz + h)
            exp_table[own_idx, s_idx] = np.exp(-delta / T)

    for k in range(n_iter):
        i = i_vals[k]
        j = j_vals[k]
        u = u_vals[k]
        own = spins[i, j]

        ip1 = i + 1 if i + 1 < L else 0
        im1 = i - 1 if i > 0     else L - 1
        jp1 = j + 1 if j + 1 < L else 0
        jm1 = j - 1 if j > 0     else L - 1

        sumviz = spins[i, jp1] + spins[i, jm1] + spins[ip1, j] + spins[im1, j]
        delta  = 2.0 * own * (J * sumviz + h)

        own_idx = 0 if own < 0 else 1
        s_idx   = int(sumviz) + 4

        if delta < 0.0 or exp_table[own_idx, s_idx] > u:
            spins[i, j] = -own
            E   += delta
            mag -= 2.0 * J * own * inv_L2

        energies[k + 1]       = E
        magnetizations[k + 1] = mag

    return E, mag


# ── Classe (wrapper fino) ────────────────────────────────────────────────────
class IsingModel:

    def __init__(self, L, T):
        self.L = L
        self.T = T
        self.spins = np.ones((L, L), dtype=np.int8)
        self.energies       = np.empty(0)
        self.magnetizations = np.empty(0)

    def calc_ener(self) -> float:
        right = np.roll(self.spins, -1, axis=1)
        down  = np.roll(self.spins, -1, axis=0)
        return -J * np.sum(self.spins * (right + down)) - h * np.sum(self.spins)

    def calc_mag(self) -> float:
        return J * np.sum(self.spins) / self.L**2

    def iter_monte_carlo(self, n_iter):
        L = self.L

        self.energies       = np.empty(n_iter + 1)
        self.magnetizations = np.empty(n_iter + 1)

        E0   = float(self.calc_ener())
        mag0 = float(self.calc_mag())

        i_vals = rnd.integers(0, L, size=n_iter, dtype=np.int64)
        j_vals = rnd.integers(0, L, size=n_iter, dtype=np.int64)
        u_vals = rnd.uniform(0.0, 1.0, size=n_iter)

        _metropolis_sweep(
            self.spins, n_iter, float(self.T), float(J), float(h),
            i_vals, j_vals, u_vals,
            self.energies, self.magnetizations,
            E0, mag0,
        )

    @property
    def energy(self):
        return self.energies

    @property
    def magnetization(self):
        return self.magnetizations


# ── Modelo de termalização (necessário para reconstruir interp_tau) ─────────
def therm_model(N, ef, tau):
    e0 = -2.0
    return ef + (e0 - ef) * np.exp(-N / tau)


# ═════════════════════════════════════════════════════════════════════════════
# Reconstruir interp_tau (pipeline do ex. 1, sem plots)
# ═════════════════════════════════════════════════════════════════════════════

L_grid = np.array([16, 32, 64, 128])
T_grid = np.array([1.0, 2.0, 3.0, 4.0])
n_sweeps = 10

tau_mat = np.zeros((len(L_grid), len(T_grid)))

for iL, L in enumerate(L_grid):
    for iT, T in enumerate(T_grid):
        model = IsingModel(L, T)
        model.iter_monte_carlo(n_sweeps * L**2)
        e = model.energy / L**2
        N = np.arange(len(e), dtype=float)
        step  = max(1, len(e) // 500)
        e_sub, N_sub = e[::step], N[::step]
        n_tail = max(1, len(e_sub) // 10)
        try:
            popt, _ = curve_fit(
                therm_model, N_sub, e_sub,
                p0     = [float(e_sub[-n_tail:].mean()), float(L**2)],
                bounds = ([-2.5, 1.0], [0.5, np.inf]),
                maxfev = 10000,
            )
            tau_mat[iL, iT] = popt[1]
        except (RuntimeError, ValueError):
            tau_mat[iL, iT] = np.nan

interp_tau = RegularGridInterpolator((L_grid, T_grid), tau_mat)


# ═════════════════════════════════════════════════════════════════════════════
# Exercício 2(a)
# ═════════════════════════════════════════════════════════════════════════════

L_values_2a = [16, 32, 64, 128]
T_values_2a = np.arange(1.0, 4.0 + 1e-9, 0.2)
n_meas      = 1_000_000

e_mean = np.zeros((len(L_values_2a), len(T_values_2a)))
e_std  = np.zeros_like(e_mean)
m_mean = np.zeros_like(e_mean)
m_std  = np.zeros_like(e_mean)

for iL, L in enumerate(L_values_2a):
    for iT, T in enumerate(T_values_2a):
        n_term = max(1, int(4 * float(interp_tau([[L, T]])[0])))

        model = IsingModel(L, T)
        model.iter_monte_carlo(n_term)        # termalização
        model.iter_monte_carlo(n_meas)        # medição

        e_arr = model.energy / L**2
        m_arr = model.magnetization

        e_mean[iL, iT] = e_arr.mean()
        e_std [iL, iT] = e_arr.std()
        m_mean[iL, iT] = m_arr.mean()
        m_std [iL, iT] = m_arr.std()

# ── Gráficos ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    r"Exercício 2(a) — $\langle e \rangle(T)$ e $\langle m \rangle(T)$ para vários $L$",
    fontsize=13
)
cores = ["C0", "C1", "C2", "C3"]

for iL, L in enumerate(L_values_2a):
    axes[0].errorbar(T_values_2a, e_mean[iL], yerr=e_std[iL],
                     marker="o", ms=4, lw=1, capsize=2,
                     color=cores[iL], label=f"$L = {L}$")
    axes[1].errorbar(T_values_2a, m_mean[iL], yerr=m_std[iL],
                     marker="o", ms=4, lw=1, capsize=2,
                     color=cores[iL], label=f"$L = {L}$")

Tc_inf = 2.0 / np.log(1.0 + np.sqrt(2.0))
axes[1].axvline(Tc_inf, color="k", ls=":", lw=1, label=f"$T_c^\\infty$ ≈ {Tc_inf:.3f}")

axes[0].set(xlabel="$T$", ylabel=r"$\langle e \rangle$", title="Energia média por spin")
axes[1].set(xlabel="$T$", ylabel=r"$\langle m \rangle$", title="Magnetização média por spin")
for ax in axes:
    ax.grid(True, ls="--", alpha=0.4); ax.legend()

plt.tight_layout(); plt.show()

# ── Guardar resultados para 2(b), 2(c), 2(d) ─────────────────────────────────
np.savez(
    "results_2a.npz",
    L_values = np.array(L_values_2a),
    T_values = T_values_2a,
    e_mean   = e_mean, e_std = e_std,
    m_mean   = m_mean, m_std = m_std,
)