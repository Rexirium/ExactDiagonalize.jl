# ==============================================
# Entanglement entropy calculation for pure states
# ===============================================

function matrixize(basis::AbstractBasis, psi::Vector{T}, b::Int) where T <: Number

    left_parts, right_parts = splitbasis(basis.bitsvec, basis.lsize - b)

    lbits = unique(left_parts)
    rbits = unique(right_parts)
    M, N = length(lbits), length(rbits)

    ldict = Dict(v => i for (i, v) in enumerate(lbits))
    rdict = Dict(v => i for (i, v) in enumerate(rbits))

    mat = zeros(T, M, N)
    @inbounds for i in 1 : basis.dim
        lidx = ldict[left_parts[i]]
        ridx = rdict[right_parts[i]]
        mat[lidx, ridx] = psi[i]
    end
    return mat
end

function ent_entropy(basis::AbstractBasis, psi::Vector, b::Int=basis.lsize ÷ 2)
    (b <= 0 || b >= basis.lsize) && return 0.0
    mat = matrixize(basis, psi, b)
    Σ = svdvals!(mat)
    SvN = 0.0

    @inbounds for s in Σ
        p = s*s
        if 1e-300 < p < 1.0
            SvN -= p * log(p)
        end
    end
    return SvN
end

ent_entropy(psi::QState, b::Int=psi.basis.lsize ÷ 2) = ent_entropy(psi.basis, psi.vector, b)

function reduced_density_matrix(basis::AbstractBasis, psi::Vector, b::Int=basis.lsize ÷ 2; subsys::Char='A')
    mat = matrixize(basis, psi, b)
    if subsys == 'A'
        return mat * mat'
    elseif subsys == 'B'
        return mat' * mat
    end
    return mat * mat'
end

reduced_density_matrix(psi::QState, b::Int=basis.lsize ÷ 2; subsys::Char='A') = reduced_density_matrix(psi.basis, psi.vector, b; subsys=subsys)