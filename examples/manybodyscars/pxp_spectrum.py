import numpy as np
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from pxp_basis import pxp_basis_1d
import matplotlib.pyplot as plt

L = 20
b = L // 2

basis = pxp_basis_1d(L)
basis_nosym = pxp_basis_1d(L)

h_list = [[1.0, i] for i in range(L)]
static = [["x", h_list]]

no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
H_pxp = hamiltonian(static, [], basis=basis, dtype=np.float64, **no_checks)

E, U = H_pxp.eigh()
print("PXP spectrum solved")

Z2state = np.zeros(basis.Ns)
Z2idx = basis.index("10" * b)

Z2state[Z2idx] = 1.0

pxp_idx = ((1 << L) - 1) - basis_nosym.states

overlaps = np.abs(np.matvec(U.T, Z2state)) ** 2
entropies = np.zeros(basis.Ns)

#psi_full = np.zeros(1 << L)
#psi_nosym = np.zeros(basis_nosym.Ns)
for n in range(basis.Ns):
    #psi_full[:] = basis.project_from(U[:, n], sparse=False)
    #psi_nosym[:] = psi_full[pxp_idx]
    #entropies[n] = basis_nosym.my_ent_entropy(psi_nosym, density=False)
    entropies[n] = basis.my_ent_entropy(U[:, n], density=False)

mask = overlaps > 0

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

ax.scatter(E[mask], overlaps[mask], c=entropies[mask], cmap='plasma', s=10)
ax.set_yscale("log")
ax.set_ylim(1e-10, 1.0)

plt.savefig("examples/manybodyscars/spec_overlap.png")
