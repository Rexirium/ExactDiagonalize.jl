import warnings

import numpy as np
import scipy.sparse as spp
from scipy.linalg import svdvals
from quspin.basis import spin_basis_1d

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


def my_ent_entropy(self, psi, b=None):
    
    if hasattr(self, "blocks") and len(self.blocks) > 0:
        warnings.warn(
            "Use a symmetry-free basis for reliable entanglement entropy.",
            UserWarning,
            stacklevel=2,
        )
    
    if b is None:
        b = self.N // 2

    cache = getattr(self, "_ent_entropy_cache", {})
    partition = cache.get(b)
    if partition is None:
        shift = self.N - b
        if self.sps == 2:
            mask = (1 << shift) - 1
            left_parts = self.states >> shift
            right_parts = self.states & mask
        else:
            pow_sps = np.pow(self.sps, shift)
            left_parts = self.states // pow_sps
            right_parts = self.states % pow_sps

        lstates, l_idx = np.unique(left_parts, return_inverse=True)
        rstates, r_idx = np.unique(right_parts, return_inverse=True)
        partition = (l_idx, r_idx, lstates.size, rstates.size)
        cache[b] = partition
        self._ent_entropy_cache = cache

    l_idx, r_idx, n_left, n_right = partition
    mat = np.zeros((n_left, n_right), dtype=psi.dtype)
     # 4. 根据 psi 的类型进行智能映射 (支持 Sparse 和 Dense)
    # ==============================================================
    if spp.issparse(psi):
        # 转换为 COO 格式，便于直接提取 "非零元素的坐标和数据"
        psi_coo = psi.tocoo()
        # 这里仅提取出非零态的左右子空间坐标进行赋值，极其高效！
        mat[l_idx[psi_coo.row], r_idx[psi_coo.row]] = psi_coo.data
    else:
        # np.ravel 确保: 即便传入的是 (N, 1) 的 dense array，也会被展平为 (N,)
        # 从而匹配 mat[l_idx, r_idx] 所需的一维形态
        mat[l_idx, r_idx] = np.ravel(psi)
    # ==============================================================
    
    S = svdvals(mat, overwrite_a=True, check_finite=False)
    # S = svds(mat, k = min(*mat.shape)-1, tol= 1e-12, return_singular_vectors=False)
    ps = S * S
    ps = ps[((ps > 1e-300) & (ps < 1.0))]
    
    entropy = -np.sum(ps * np.log(ps))
    return entropy

spin_basis_1d.my_ent_entropy = my_ent_entropy



if __name__ == "__main__":
    rng = np.random.default_rng()
    from numpy.linalg import norm
    
    basis = spin_basis_1d(L=10, Nup=5)
    basis_full = spin_basis_1d(L=10)
    b = 2
    
    psi = rng.normal(size = basis.Ns)
    psi /= norm(psi)
    psi_full = basis.project_from(psi, sparse=True)
    

    subA = tuple(range(b))
    myent = basis.my_ent_entropy(psi, b, density=False)
    myent_full = basis_full.my_ent_entropy(psi_full, b, density=False)
    qsent = basis.ent_entropy(psi, subA, density=False)
    qsent_full = basis_full.ent_entropy(psi_full, subA, density=False)
    
    print(f"my entropy is {myent}")
    print(f"my entropy full is {myent_full}")
    print(f"quspin entropy is {qsent["Sent_A"]}")
    print(f"quspin entropy full is {qsent_full["Sent_A"]}")
