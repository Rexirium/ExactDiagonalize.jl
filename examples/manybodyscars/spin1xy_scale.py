from spin1xy import *

Ls = [6 + 2*n for n in range(7)]
Stot = 1
J, h = 1.0, 1.0
nt = 1201
ts = np.linspace(0.0, 120.0, nt)

dts = 2e-2 * np.ones(nt)
profile = [11 for _ in range(nt)]
profile[-1] = 101
ts_full = time_expand(ts, dts, profile)

entropies = np.zeros((nt, len(Ls)))
entropies_full = np.zeros_like(ts_full)

for n, L in enumerate(Ls):
    b = L // 2
    basis = spin_basis_1d(L=L, S="1", Nup = Stot)
    E, U = spin1xy_spectrum(basis, L, J, h)
    print("L = {} spectrum solved".format(L))
    
    psi0 = np.zeros(basis.Ns)
    psi0[basis.Ns // 2] = 1.0
    
    psi_t = ED_state_vs_time(psi0, E, U, ts_full, iterate=True)
    
    subA = tuple(range(L // 2))
    for i, psi in enumerate(psi_t):
        entr = my_ent_entropy(basis.states, 3, psi, b, density=False)
        entropies_full[i] = entr
        
    entropies[:, n] = latetime_average(entropies_full, profile)
    print("L = {} entropy obtained".format(L))

np.savez(f"examples/manybodyscars/spin1xy_Sz={Stot}_Lmax={Ls[-1]}.npz", 
        Ls = np.array(Ls), ts = ts,
        params = np.array([J, h]),  
        entropies=entropies
    )