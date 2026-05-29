import numpy as np
from functools import reduce
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
from quspin.tools.measurements import ED_state_vs_time

def spin1xy_spectrum(basis, J:float, h:float):
    # 构造哈密顿量
    lsize = basis.L
    # S^x S^x + S^y S^y 项
    static = [["+-", [[J/2, i, (i+1) % lsize] for i in range(lsize)]]]
    static += [["-+", [[J/2, i, (i+1) % lsize] for i in range(lsize)]]]
    # h S^z 项
    static += [["z", [[h, i] for i in range(lsize)]]]
    H = hamiltonian(static, [], basis=basis, dtype=np.float64)
    return H.eigh()

def make_initialstate(basis, statestr:str, s:int):
    lsize = basis.L
    
    vp = np.array([1., 0., 1.]) / np.sqrt(2)
    vm = np.array([-1., 0., 1.]) / np.sqrt(2)
    
    if s not in [-1, 0, 1]:
        raise ValueError("Invalid s value, it must be one of -1, 0, 1")
    
    if statestr == "NN":
        if s == 1:
            vs = np.kron(vp, vm)
        elif s == -1:
            vs = np.kron(vm, vp)
        else:
            raise ValueError("s = 0 incompactible with Nematic Neel state")
        
        psi_full = reduce(np.kron, [vs] * (lsize // 2))
    elif statestr == "NF":
        if s == 1:
            psi_full = reduce(np.kron, [vp] * lsize)
        elif s == -1:
            psi_full = reduce(np.kron, [vm] * lsize)
        else:
            raise ValueError("s = 0 incompactible with Nematic Ferro state")  
    elif statestr == "SZ":
        vs = np.zeros(3)
        vs[s + 1] = 1.0
        psi_full = reduce(np.kron, [vs] * lsize)
    
    return psi_full[basis.states]

def spin1xy_dynamic(basis, psi0:np.ndarray, ts:np.ndarray, J:float, h:float):
    E, U = spin1xy_spectrum(basis, J, h)
    print("spectrum solved")
    b = basis.L // 2
    subA = tuple(range(b))
    
    psi_t = ED_state_vs_time(psi0, E, U, ts, iterate=True)
    
    entropies = np.zeros_like(ts)
    for i, psi in enumerate(psi_t):
        entr = basis.ent_entropy(psi, sub_sys_A=subA, return_rdm=None, sparse_diag=True)
        entropies[i] = entr["Sent_A"]
    
    return entropies
        
    
# --- 主程序 ---
    
if __name__=="__main__":
    L = 8
    J, h = 1.0, 0.5
    b = L // 2
    
    basis = spin_basis_1d(L=L, S="1")
    E, U = spin1xy_spectrum(basis, J, h)
    print("spectrum solved")
    
    ts = np.geomspace(0.1, 1e7, 501)
    subA = tuple(range(b))
    
    psis = ["NN", "NF", "SZ"]
    
    entropies = np.zeros((len(ts), 3))
    for n, ss in enumerate(psis):
        psi0 = make_initialstate(basis, ss, 1)
        psi_t = ED_state_vs_time(psi0, E, U, ts, iterate=True)
        
        for i, psi in enumerate(psi_t):
            entr = basis.ent_entropy(psi, sub_sys_A=subA, return_rdm=None, sparse_diag=True)
            entropies[i, n] = entr["Sent_A"]
    
    np.savez("examples/manybodyscars/spin1xy_L=8.npz", ts = ts, entropies = entropies)



    
    
