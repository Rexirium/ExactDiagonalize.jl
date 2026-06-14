# ==============================================
# Entanglement entropy calculation for pure states
# ===============================================

function matrixize(basis::AbstractBasis, psi::Vector{T}, b::Int) where T <: Number
    shift = basis.lsize - b

    left_parts = basis.bitsvec .>> shift
    right_parts = basis.bitsvec .& ((0x00001 << shift) - 0x00001)

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
        if 1e-300 < p <= 1.0
            SvN -= p * log(p)
        end
    end
    return SvN
end

ent_entropy(psi::QState, b::Int=psi.basis.lsize ÷ 2) = ent_entropy(psi.basis, psi.vector, b)