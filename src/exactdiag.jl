"""
Exact diagonalization and time evolution routines
"""
# Tag for exact diagonalization
exact() = Val(:exact)

# Compute spectrum for given operator sum and basis
function spectrum(ops::OpSum, basis::AbstractBasis; retvecs::Bool=false)
    hmat = makeHamiltonian(ops, basis)
    if retvecs 
        return eigen!(Hermitian(hmat))  
    else 
        return eigvals!(Hermitian(hmat))
    end
end

# Compute full spectrum by summing over all particle numbers
function spectrum(ops::OpSum, lsize::Int)
    energies = Float64[]
    sizehint!(energies, 1 << lsize)
    for num in 0:lsize
        basis = SpinBasis(lsize; num = num)
        hmat = makeHamiltonian(ops, basis)
        eigs = eigvals!(Hermitian(hmat))
        append!(energies, eigs)
    end
    return energies
end

# Evolve state for multiple time steps and record observables
function timeEvolve(ops::OpSum, basis::AbstractBasis, psi0::Vector, ts::AbstractVector, obs::AbstractObserver, ::Val{:exact})
    hmat = makeHamiltonian(ops, basis)
    eigs, U = eigen!(Hermitian(hmat))

    psi = ComplexF64.(psi0)
    psi_trans = U' * psi
    phases = similar(psi)
    psi_phases = similar(psi)

    for (i, t) in enumerate(ts)
        if i == 1
            record!(obs, psi, 1)
            continue
        end
        phases = cis.( -t * eigs)
        psi_phases .= phases .* psi_trans
        mul!(psi, U, psi_phases)
        record!(obs, psi, 1)
    end
    return psi
end

function timeEvolve(ops::OpSum, init::QState, ts::AbstractVector, obs::AbstractObserver, ::Val{:exact})
    psi = timeEvolve(ops, init.basis, init.vector, ts, obs, Val(:exact))
    return QState(init.basis, psi)
end

# Default method: exact diagonalization
timeEvolve(ops::OpSum, basis::AbstractBasis, psi0::Vector, ts::AbstractVector, obs::AbstractObserver) =
    timeEvolve(ops, basis, psi0, ts, obs, Val(:exact))
    
timeEvolve(ops::OpSum, init::QState, ts::AbstractVector, obs::AbstractObserver) = 
    timeEvolve(ops, init, ts, obs, Val(:exact))