import numpy as np
from quspin.basis import spin_basis_1d
from scipy.linalg import svdvals

def merge_basis_index(bases:list[spin_basis_1d]):
    
    merged_states = np.concatenate([basis.states for basis in bases])
    sort_idx = np.argsort(merged_states)[::-1]
    # 4. 计算逆排列 (求出原数组每个元素在新数组中的指标)
    new_indices = np.empty_like(sort_idx)
    new_indices[sort_idx] = np.arange(len(sort_idx))
    
    # 5. 按照原数组的长度，将 new_indices 切分回原来的结构
    lengths = [basis.Ns for basis in bases]
    split_points = np.cumsum(lengths)[:-1]  # 计算切分点
    basis_indices = np.split(new_indices, split_points)
    
    return basis_indices


def my_ent_entropy(self, psi:np.ndarray, b:int, density:bool=True):
    # 3进制表示特有
    pow_sps = np.pow(self.sps, b)
     # 1. 向量化计算所有态的左边和右边部分
    left_parts = self.states // pow_sps
    right_parts = self.states % pow_sps
    
    # 2. 获取去重后的状态，以及每个原始状态在新列表中的索引
    # l_idx 和 r_idx 的长度与 basis_states 完全一致
    lstates, l_idx = np.unique(left_parts, return_inverse=True)
    rstates, r_idx = np.unique(right_parts, return_inverse=True)
    
    mat = np.zeros((lstates.size, rstates.size), dtype=psi.dtype)
    # 4. NumPy 高级索引 (Fancy Indexing)，一步到位完成所有数据的映射
    mat[l_idx, r_idx] = psi
    
    S = svdvals(mat, overwrite_a=True, check_finite=False)
    # S = svds(mat, k = min(*mat.shape)-1, tol= 1e-12, return_singular_vectors=False)
    ps = S * S
    ps = ps[ps > 1e-33]
    
    if density:
        return - np.sum(ps * np.log(ps)) / b
    else:
        return - np.sum(ps * np.log(ps))

spin_basis_1d.my_ent_entropy = my_ent_entropy

def ED_state_vs_time_1D(psi, E, ts, iterate=True):
    psi_t = psi * np.exp( -(1j * E) * ts)
    if iterate:
        yield from psi_t
    else:
        return psi_t
    
def time_expand(ts:np.ndarray, dts, profile:list):
    ts_full = np.zeros(np.sum(profile))
    idx = 0
    for t, dt, n in zip(ts, dts, profile):
        ts_full[idx : idx + n] = np.linspace(t - dt, t + dt, n)
        idx += n
    
    return ts_full

def latetime_average(arr:np.ndarray, profile:list):
    res = np.zeros(len(profile))
    idx = 0
    for i, n in enumerate(profile):
        res[i] = np.sum(arr[idx : idx + n]) / n
        idx += n
    
    return res