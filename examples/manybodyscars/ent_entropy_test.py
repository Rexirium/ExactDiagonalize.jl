from spin1xy import *
import matplotlib.pyplot as plt

Ls = [6, 8, 10, 12]
nmaxs = [6, 8, 10, 12, 14]

entropies = np.zeros(len(Ls))

for il, L in enumerate(Ls):
    b = L // 2
    """
    nmax = nmaxs[il]
    nbasis = nmax + 1
    bases = [spin_basis_1d(L=L, S="1", Nup = 2 * n) for n in range(nbasis)]
    
    # initial states
    psi0s, coeffs = make_initial(bases, nmax)
    
    # Combine the sym bases and combine the state vector
    basis_tot = np.concatenate([bases[n].states for n in range(nbasis)])
    dim = basis_tot.size
    psi_tot = np.zeros(dim, dtype=complex)
    
    # indices to update psi in doing the time evolution
    split_points = np.cumsum([bases[n].Ns for n in range(nbasis - 1)])
    inds = np.split(np.arange(dim), split_points)
    
    for n, psi in enumerate(psi0s):
        psi_tot[inds[n]] = coeffs[n] * psi
    """
    
    basis = spin_basis_1d(L=L, S="1")
    psi0 = make_initial_total(basis, 2, sign=1)
    entropies[il] = my_ent_entropy(basis.states, 3, psi0, b, density=False)
    # entropies[il] = basis.ent_entropy(psi0, sub_sys_A=None, return_rdm=None, density=False)["Sent_A"]
        
    print("L = {} entropy obtained".format(L))

fig, ax = plt.subplots()

ax.plot(Ls, entropies)
ax.set(xlabel="L", ylabel="Entropy")

plt.show()



