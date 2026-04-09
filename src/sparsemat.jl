"""
Sparse matrix exponential time evolution
"""
# Tag for sparse matrix method
spmat() = Val(:spmat)

# In-place Taylor expansion update for exp(-iHdt)ψ
function updating!(psi::Vector{T}, hmat::SpMatrix, dt::Real, order::Int) where T <: Number
    fac = one(T)
    tmp = copy(psi)
    Hpsi = Vector{T}(undef, length(psi))
    for k in 1:order  # calculate the (-iHt)^k / k! term iteratively
        fac *= -im * dt / k
        mul!(Hpsi, hmat, tmp)
        psi .+= fac * Hpsi
        tmp .= Hpsi
    end
end

# Evolve state using Taylor expansion of exp(-iHt) with sparse matrix
function timeEvolve(ops::OpSum, init::QState, ts::AbstractVector, obs::AbstractObserver, ::Val{:spmat}; order::Int=4)
    hmat = makeHamiltonian(ops, init.basis; sparsed=true)
    psi = ComplexF64.(init.vector)

    for (i, t) in enumerate(ts)
        record!(obs, psi, i)
        if i == length(ts)
            break
        end
        dt = ts[i + 1] - t
        updating!(psi, hmat, dt, order)
    end
    return QState(init.basis, psi)
end