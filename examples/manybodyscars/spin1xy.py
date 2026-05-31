import numpy as np
import numpy.linalg as LA
from functools import reduce
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
from quspin.tools.measurements import ED_state_vs_time
import time

rng = np.random.default_rng()

def spin1xy_spectrum(bases, lsize:int, J:float, h:float):
    # 构造哈密顿量
    # S^x S^x + S^y S^y 项
    static = [["+-", [[J/2, i, (i+1) % lsize] for i in range(lsize)]]]
    static += [["-+", [[J/2, i, (i+1) % lsize] for i in range(lsize)]]]
    # h S^z 项
    static += [["z", [[h, i] for i in range(lsize)]]]
    
    if isinstance(bases, list) == False:
        H = hamiltonian(static, [], basis=bases, dtype=np.float64, 
                check_herm=False, check_pcon=False, check_symm=False)
        return H.eigh()

    Es, Us = [], []
    for basis in bases:
        H = hamiltonian(static, [], basis=basis, dtype=np.float64, 
                check_herm=False, check_pcon=False, check_symm=False)
        E, U = H.eigh()
        Es.append(E)
        Us.append(U)

    return Es, Us

def make_initialstate(bases, lsize:int, statestr:str, s:int):
    
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
    
    if isinstance(bases, list) == False:
        return psi_full[bases.states]
    
    psis = []
    for basis in bases:
        psi = psi_full[basis.states]
        psis.append(psi)
    
    return psis

def make_arealaw_initial(bases, lsize:int):
    return 

def ED_state_vs_time_1D(psi, E, ts):
    psi_t = psi * np.exp( -(1j * E) * ts)
    yield from psi_t
      
# --- 主程序 ---
    
if __name__=="__main__":
    import matplotlib.pyplot as plt
    
    L = 10
    J, h = 1.0, 1.0
    b = L // 2
    
    basis = spin_basis_1d(L=L, S="1", Nup=1)
    print(basis.Ns)
    
    E, U = spin1xy_spectrum(basis, L, J, h)
    print("spectrum solved")
    
    psi0 = np.zeros(basis.Ns)
    psi0[0] = 1.0
    psi0 = psi0 / LA.norm(psi0)
    ts = np.linspace(0, 10, 101)
    
    
    psi_t = ED_state_vs_time(psi0, E, U, ts, iterate=True)
    
    subA = tuple(range(b))
    entropies = np.zeros_like(ts)
    for i, psi in enumerate(psi_t):
        entr = basis.ent_entropy(psi, sub_sys_A=subA, return_rdm=None, sparse_diag=True)
        entropies[i] = entr["Sent_A"]
        
    fig, ax = plt.subplots()
    ax.plot(ts, entropies)
    # ax.set_xscale("log")
    plt.show()
    
    
