import numpy as np
import scipy.linalg as sla
from scipy.special import comb
# from scipy.sparse.linalg import svds
from functools import reduce
from itertools import combinations
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
from quspin.tools.measurements import ED_state_vs_time
from utils import *
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

def make_athermal_initial(basis, n:int, sign:int = 1):
    lsize = basis.L
    psi = np.zeros(basis.Ns)
    combs = np.array(list(combinations(np.arange(lsize), n)), dtype=int)
    
    states = np.sum(2 * np.pow(3, combs), axis=1)
    for j, stateint in enumerate(states):
        idx = basis.index(stateint)
        phase = np.abs(lsize - np.sum(combs[j, :], dtype=int))
        psi[idx] = np.pow(-1, phase) * sign
    
    return psi / sla.norm(psi)

def make_initial(bases:list, nmax:int, sign:int = 1):
    psis = []
    coeffs = []
    for (n, basis) in enumerate(bases):
        psi = make_athermal_initial(basis, n, sign=sign)
        cc = np.sqrt(1 / 2**nmax * comb(nmax, n))
        psis.append(psi)
        coeffs.append(cc)
        
    return psis, coeffs

def make_initial_total(basis, nmax:int=None, sign:int = 1):
    lsize = basis.L
    psi = np.zeros(basis.Ns, dtype=complex)
    
    if nmax is None:
        nmax = basis.L

    for n in range(nmax + 1):
        combs = np.array(list(combinations(np.arange(lsize), n)), dtype=int)
        states = np.sum(2 * np.pow(3, combs), axis=1)
        
        for j, stateint in enumerate(states):
            idx = basis.Ns - 1 - stateint
            phase = np.abs(lsize - np.sum(combs[j, :], dtype=int))
            coeff = np.sqrt(1 / 2**lsize * comb(lsize, n))
            psi[idx] = np.pow(-1, phase) * sign * coeff
        
    return psi / sla.norm(psi)

      
# --- 主程序 ---
    
if __name__=="__main__":
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(123)
    
    L = 10
    J, h = 1.0, 1.0
    b = L // 2
    n_mag = 2
    
    basis = spin_basis_1d(L=L, S="1", Nup=2*n_mag)
    print(basis.Ns)
    
    E, U = spin1xy_spectrum(basis, L, J, h)
    print("spectrum solved")
    
    psi0 = rng.normal(size=basis.Ns)
    psi0 /= sla.norm(psi0)
    
    nt = 201
    ts = np.geomspace(0.1, 1e8, nt)
    dts = np.geomspace(1e-3, 1e3, nt)
    profile = [3 if i < (nt // 2) else 5 for i in range(nt)]
    ts_full = time_expand(ts, dts, profile)
    
    
    psi_t = ED_state_vs_time(psi0, E, U, ts_full, iterate=True)
    
    start = time.perf_counter()
    entropies_full = np.zeros_like(ts_full)
    for i, psi in enumerate(psi_t):
        # entr = basis.ent_entropy(psi, sub_sys_A=subA, return_rdm=None)["Sent_A"]
        entropies_full[i] = basis.my_ent_entropy(psi, b, density=False)
    stop = time.perf_counter()
    print("Entropy calc time {}".format(stop - start))
      
    entropies = latetime_average(entropies_full, profile)
        
    fig, ax = plt.subplots()
    ax.plot(ts, entropies)
    ax.set_xscale("log")
    # plt.savefig("examples/manybodyscars/my_entropy.png")
    plt.show()
    
    
