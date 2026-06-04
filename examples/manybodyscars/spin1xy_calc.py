from spin1xy import *

L = 6
nt = 501
J, h = 1.0, 0.5
b = L // 2
    
bases = [spin_basis_1d(L=L, S="1", Nup=n) for n in range(2*L +1)]
    
Es, Us = spin1xy_spectrum(bases, L, J, h)
print("spectrum solved")
    
ts = np.geomspace(0.1, 1e7, nt)
subA = tuple(range(b))

psis = ["NN", "NF", "SZ"]

basis_full = spin_basis_1d(L=L, S="1")
psi = np.zeros(pow(3, L), dtype=complex)
entropies = np.zeros((len(ts), 3))

for k, ss in enumerate(psis):
    psi0 = make_initialstate(bases, L, ss, 1)
    
    psi_t_list = []
    for n in range(2 * L + 1):
        if len(Es[n]) == 1:
            psi_t = ED_state_vs_time_1D(psi0[n][0], Es[n][0], ts)
        else: 
            psi_t = ED_state_vs_time(psi0[n], Es[n], Us[n], ts, iterate=True)
        psi_t_list.append(psi_t)
        
    for i, psis in enumerate(zip(*psi_t_list)):
        for n, basis in enumerate(bases):
            psi[basis.states] = psis[n]
        
        entropies[i, k] = basis.my_ent_entropy(psi, b, density=False)
    
np.savez(f"examples/manybodyscars/spin1xy_L={L}.npz", ts = ts, entropies = entropies)