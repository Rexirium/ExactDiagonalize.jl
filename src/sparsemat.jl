"""
Sparse matrix exponential time evolution
"""
struct SpMatAlg
    order::Int
end
# Tag for sparse matrix method
spmat(; order::Int=4) = SpMatAlg(order)

# In-place Taylor expansion update for exp(-iHdt)ψ
function updating!(psi::Vector{T}, hmat::SpMatrix, dt::Real, order::Int) where T <: Number
    fac = one(T)
    tmp = copy(psi)
    Hpsi = Vector{T}(undef, length(psi))
    for k in 1:order  # calculate the (-iHt)^k / k! term iteratively
        fac *= -im * dt / k
        mul!(Hpsi, hmat, tmp)
        psi .+= fac .* Hpsi
        tmp .= Hpsi
    end
end

# Evolve state using Taylor expansion of exp(-iHt) with sparse matrix
function timeEvolve(ops::OpSum, basis::AbstractBasis, psi0::Vector, ts::AbstractVector, obs::AbstractObserver, alg::SpMatAlg)
    hmat = makeHamiltonian(ops, basis; sparsed=true)
    psi = ComplexF64.(psi0)

    for i in eachindex(ts)
        record!(obs, psi, i)
        if i == length(ts)
            break
        end

        dt = ts[i+1] - ts[i]
        updating!(psi, hmat, dt, alg.order)
    end
    return psi
end

function timeEvolve(ops::OpSum, init::QState, ts::AbstractVector, obs::AbstractObserver, alg::SpMatAlg)
    psi = timeEvolve(ops, init.basis, init.vector, ts, obs, alg)
    return QState(init.basis, psi)
end