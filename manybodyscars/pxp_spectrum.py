import numpy as np
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from pxp_basis import pxp_basis_1d
import matplotlib.pyplot as plt

L = 16
b = L // 2
g = 0.1

basis = pxp_basis_1d(L)

h_list = [[1.0, i] for i in range(L)]
hh_list = [[g, i, (i + 1) % L] for i in range(L)]

static = [["x", h_list], ["xx", hh_list], ["yy", hh_list], ["zz", hh_list]]

no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
H_pxp = hamiltonian(static, [], basis=basis, dtype=np.float64, **no_checks)

E, U = H_pxp.eigh()
print("PXP spectrum solved")

Z2state = np.zeros(basis.Ns)
Z2idx = basis.index("10" * b)

Z2state[Z2idx] = 1.0

overlaps = np.abs(np.matvec(U.T, Z2state)) ** 2


# entropies = basis.ent_entropy(U, density=False, enforce_pure=True)['Sent_A']

entropies = np.zeros(basis.Ns)

for n in range(basis.Ns):
    entropies[n] = basis.my_ent_entropy(U[:, n])

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

ax.scatter(E, entropies, c=overlaps, cmap='plasma', s=10)
#ax.set_yscale("log")
#ax.set_ylim(1e-5, 10.0)

plt.savefig(f"manybodyscars/pxp_python_L={L}.png")
