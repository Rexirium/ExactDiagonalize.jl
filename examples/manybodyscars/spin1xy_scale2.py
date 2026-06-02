from spin1xy import *

Ls = [6, 8, 10]
J, h = 1.0, 1.0
nt = 121
ts = np.linspace(0.0, 120, nt)


entropies = np.zeros((nt, len(Ls)))

for il, L in enumerate(Ls):
    nbasis = L + 1
    nmax = L
    b = L // 2
    bases = [spin_basis_1d(L=L, S="1", Nup = 2 * n) for n in range(nbasis)]
    Es, Us = spin1xy_spectrum(bases, L, J, h)
    print("L = {} spectrum solved".format(L))
    
    # initial states
    psi0s, coeffs = make_initial(bases, nmax)
    # Generator for time evolution
    psi_t_list = []
    for n in range(nbasis):
        if len(Es[n]) == 1:
            psi_t = ED_state_vs_time_1D(psi0s[n][0], Es[n][0], ts, iterate=True)
        else:
            psi_t = ED_state_vs_time(psi0s[n], Es[n], Us[n], ts, iterate=True)
        psi_t_list.append(psi_t)
    
    # Combine the sym bases and combine the state vector
    basis_tot = np.concatenate([bases[n].states for n in range(nbasis)])
    dim = basis_tot.size
    psi_tot = np.zeros(dim, dtype=complex)
    # indices to update psi in doing the time evolution
    split_points = np.cumsum([bases[n].Ns for n in range(nbasis - 1)])
    inds = np.split(np.arange(dim), split_points)
    
    for i, psis in enumerate(zip(*psi_t_list)):
        # update the combined
        for n, psi in enumerate(psis):
            psi_tot[inds[n]] = coeffs[n] * psi
        # psi_tot /= sla.norm(psi_tot)
            
        entr = my_ent_entropy(basis_tot, 3, psi_tot, b, density=False)
        entropies[i, il] = entr
        
    print("L = {} entropy obtained".format(L))

np.savez(f"examples/manybodyscars/spin1xy_Lmax={Ls[-1]}_total_linear.npz", 
        Ls = np.array(Ls), ts = ts,
        params = np.array([J, h]),  
        entropies=entropies
    )

