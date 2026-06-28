import numpy as np
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from pxp_basis import pxp_basis_1d
import matplotlib.pyplot as plt

L = 16
b = L // 2
g = - 0.2

basis = pxp_basis_1d(L, a=2, kblock=0)
basis_full = pxp_basis_1d(L)

h_list = [[1.0, i] for i in range(L)]
hg_list = [[g * (-1)**i, i] for i in range(L)]
hn_list = [[g * (-1)**i, i, (i + 1) % L] for i in range(L)]
hnn_list = [[g, i, (i + 2) % L] for i in range(L)]

static = [["x", h_list], ["z", hg_list]]

no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
H_pxp = hamiltonian(static, [], basis=basis, dtype=np.float64, **no_checks)

E, U = H_pxp.eigh()
print("PXP spectrum solved")

Z2state = np.zeros(basis.Ns)
Z2idx = basis.index("10" * b)

Z2state[Z2idx] = 1.0

overlaps = np.abs(np.matvec(U.T, Z2state)) ** 2

marksizes = [12 if overlaps[n] > 0.01 else 5 for n in range(basis.Ns)]

entropies = np.zeros(basis.Ns)
for n in range(basis.Ns):
    full_state = basis.pxp_project_from(U[:, n], basis_full, sparse=False)
    entropies[n] = basis_full.my_ent_entropy(full_state)

#mask = overlaps > 0

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

fig, ax = plt.subplots()

ax.scatter(E, overlaps, c=entropies, cmap='plasma', s=marksizes)
ax.set(
    title=rf"$L={L}, g={g}$ add Z_2 zz term", 
    xlabel=r"$E_n$", ylabel=r"$S(L/2)$", 
)
ax.set_yscale("log")
ax.set_ylim(1e-10, 2.0)
plt.show()
#plt.savefig(f"manybodyscars/pxp_constrained_zz2_L={L}_g={g}.png")
