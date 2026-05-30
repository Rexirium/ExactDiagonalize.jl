from spin1xy import *

L = 8
J, h = 1.0, 0.5
b = L // 2
    
basis = spin_basis_1d(L=L, S="1")
    #bases = [spin_basis_1d(L=L, S="1", Nup=n) for n in range(2*L +1)]
    
E, U = spin1xy_spectrum(basis, L, J, h)
print("spectrum solved")
    
ts = np.geomspace(0.1, 1e7, 501)
subA = tuple(range(b))

psis = ["NN", "NF", "SZ"]
    
entropies = np.zeros((len(ts), 3))
for n, ss in enumerate(psis):
    psi0 = make_initialstate(basis, L, ss, 1)
    psi_t = ED_state_vs_time(psi0, E, U, ts, iterate=True)
        
    for i, psi in enumerate(psi_t):
        entr = basis.ent_entropy(psi, sub_sys_A=subA, return_rdm=None, sparse_diag=True)
        entropies[i, n] = entr["Sent_A"]
    
np.savez("examples/manybodyscars/spin1xy_L=8.npz", ts = ts, entropies = entropies)