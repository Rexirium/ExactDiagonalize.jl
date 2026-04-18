import numpy as np
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian, quantum_LinearOperator
import matplotlib.pyplot as plt

J = 1.0
hx, hz = 0.9045, 0.8090

def get_expvals(L:int):
    basis = spin_basis_1d(L=L, a=1, kblock=0, pblock=1)

    J_zz = [[J, i, (i+1) % L] for i in range(L)]
    h_x = [[hx, i] for i in range(L)]
    h_z = [[hz, i] for i in range(L)]

    static = [["zz", J_zz], ["x", h_x], ["z", h_z]]

    h_mI = hamiltonian(static, [], basis=basis, dtype=np.float64)

    energies, eigstates = h_mI.eigh()

    ozz = [[1.0 / L, i, (i+1) % L] for i in range(L)]
    ostatic = [["xx", ozz]]

    opzz = quantum_LinearOperator(ostatic, basis=basis, dtype=np.float64)

    opzzvals = opzz.expt_value(eigstates, enforce_pure=True)
    return energies / L, opzzvals

def get_matrixele(L:int):
    basis = spin_basis_1d(L=L, a=1, kblock=0, pblock=1)

    J_zz = [[J, i, (i+1) % L] for i in range(L)]
    h_x = [[hx, i] for i in range(L)]
    h_z = [[hz, i] for i in range(L)]

    static = [["zz", J_zz], ["x", h_x], ["z", h_z]]

    h_mI = hamiltonian(static, [], basis=basis, dtype=np.float64)

    energies, eigstates = h_mI.eigh()
    omega = energies.reshape(-1, 1) - energies
    
    ozz = [[1.0 / L, i, (i+1) % L] for i in range(L)]
    ostatic = [["xx", ozz]]

    opzz = quantum_LinearOperator(ostatic, basis=basis, dtype=np.float64)
    
    opzzmat = opzz.matrix_ele(eigstates, eigstates)
    
    return omega, opzzmat
    

L_list = [10, 14, 18]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "xtick.direction": "in",
    "ytick.direction": "in"
})

fig, ax = plt.subplots()
for L in L_list:
    E, expvals = get_expvals(L)
    ax.scatter(E, expvals, s=1, label="$L = {:d}$".format(L))

ax.set_xlabel(r"$E_n/L$")
ax.set_ylabel(r"$\langle m \vert O \vert m \rangle$")

ax.legend(markerscale=2)
plt.show()
