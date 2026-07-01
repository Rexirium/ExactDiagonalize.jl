import numpy as np
from quspin.operators import hamiltonian, quantum_LinearOperator
from quspin.basis import spin_basis_1d
from quspin.tools.evolution import ED_state_vs_time
from pxp_basis import pxp_basis_1d
import matplotlib.pyplot as plt

# Parameters
L = 20
b = L // 2
g = -0.4

# Basis construction
basis = pxp_basis_1d(L, a=2, kblock=0)
basis_full = pxp_basis_1d(L)
eltype = np.float64

# Make Hamiltonian
x_list = [[1.0, i] for i in range(L)]
z_list = [[g * (-1)**i, i] for i in range(L)]
zz_list = [[- g/2 * (-1)**i, i, (i + 1) % L] for i in range(L)]
zzz_list = [[g/4 * (-1)**i, (i - 1) % L, i, (i + 1) % L] for i in range(L)]

static = [
    ["x", x_list],
    ["z", z_list], 
    #["zz", zz_list], 
    ["zzz", zzz_list],
]

no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
H_pxp = hamiltonian(static, [], basis=basis, dtype=eltype, **no_checks)

# Diagonalizing
E, U = H_pxp.eigh()
print("PXP spectrum solved")

# Define initial state
Z2state = np.zeros(basis.Ns, dtype=eltype)
Z2idx = basis.index("10" * (L // 2))
Z2state[Z2idx] = 1.0

overlaps = np.abs(np.matvec(U.conj().T, Z2state)) ** 2

marksizes = [15 if overlaps[n] > 0.01 else 5 for n in range(basis.Ns)]

energy_expected = np.dot(E, overlaps)

full_states = basis.pxp_project_from(U, basis_full, sparse=False)
entropies = np.empty(basis.Ns)
for n in range(basis.Ns):
    entropies[n] = basis_full.my_ent_entropy(full_states[:, n])

ts = np.linspace(0.0, 40.0, 2001)
states_t = ED_state_vs_time(Z2state, E, U, ts, iterate=False)

zz_local = quantum_LinearOperator(
    [["zz", [[1.0, b, (b + 1) % L] for b in range(L)]]],
    basis=basis, dtype=eltype, **no_checks,
)
zz_correlation_t = np.real(zz_local.expt_value(states_t)) / L
z2_overlap_t = np.abs(states_t[Z2idx]) ** 2

plt.rcParams.update({
    #"text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"], 
    "font.size": 14, 
    "xtick.direction": "in",
    "ytick.direction": "in", 
    "legend.frameon": False,
    "legend.edgecolor": 'none'
})


fig = plt.figure(figsize=(6, 8),layout="constrained")
grid = fig.add_gridspec(2, 1, height_ratios=(1.3, 2))
time_grid = grid[1].subgridspec(2, 1, hspace=0)
time_axes = time_grid.subplots(sharex=True)
axes = np.concatenate(([fig.add_subplot(grid[0])], time_axes))

spectrum = axes[0].scatter(
    E, overlaps, c=entropies, cmap='plasma', s=marksizes
)
axes[0].axvline(
    energy_expected, color="black", linestyle="--",
)

colorbar = fig.colorbar(spectrum, ax=axes[0], location="right", pad=0.02)
colorbar.ax.tick_params(length=0, labelsize=10)

axes[0].set(
    xlabel=r"$E_n$", ylabel=r"$|\langle \mathbb{Z}_2 |\psi \rangle|^2$", 
)
axes[0].set_yscale("log")
axes[0].set_ylim(1e-12, 1.0)

axes[1].plot(ts, zz_correlation_t)
axes[1].set(
    ylabel=r"$\langle Z_{i}Z_{i+1}\rangle$",
)
axes[1].tick_params(axis="x", labelbottom=False, bottom=False)

axes[2].plot(ts, z2_overlap_t)
axes[2].set(
    xlabel=r"$t$",
    ylabel=r"$|\langle \mathbb{Z}_2|\psi(t)\rangle|^2$",
)

plt.savefig(f"manybodyscars/pxp_zandzzz_L={L}_g={g}.png")
