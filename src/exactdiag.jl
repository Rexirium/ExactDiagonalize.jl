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

# Evolve state to time tf using exact diagonalization
function timeEvolve(ops::OpSum, init::QState, tf::Real)
    hmat = makeHamiltonian(ops, init.basis)
    eigs, U = eigen(Hermitian(hmat))
    
    phases = cis.( -tf * eigs)
    expEt = Diagonal(phases)
    final = U * expEt * U' * (init.vector)
    return QState(init.basis, final)
end

# Evolve state for multiple time steps and record observables
function timeEvolve(ops::OpSum, init::QState, ts::AbstractRange, obs::AbstractObserver, ::Val{:exact})
    hmat = makeHamiltonian(ops, init.basis)
    eigs, U = eigen!(Hermitian(hmat))

    psi = ComplexF64.(init.vector)
    psi_trans = U' * psi

    dt = step(ts)
    step_phases = cis.(-dt * eigs)

    record!(obs, psi, 1)
    for i in 2:length(ts)
        psi_trans .*= step_phases
        mul!(psi, U, psi_trans)
        record!(obs, psi, i + 1)
    end
    return QState(init.basis, psi)
end

# Default method: exact diagonalization
timeEvolve(ops::OpSum, init::QState, ts::AbstractRange, obs::AbstractObserver) = timeEvolve(ops, init, ts, obs, Val(:exact))