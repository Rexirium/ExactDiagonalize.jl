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
        basis = NumBasis(lsize, num)
        hmat = makeHamiltonian(ops, basis)
        eigs = eigvals!(Hermitian(hmat))
        append!(energies, eigs)
    end
    return energies
end

# Evolve state to time tf using exact diagonalization
function timeEvolve(ops::OpSum, init::AbstractState, tf::Real)
    hmat = makeHamiltonian(ops, init.basis)
    eigs, U = eigen(Hermitian(hmat))
    replace!(x -> isapprox(x, 0; atol=2*eps(Float64)) ? 0.0 : x, U)
    
    phases = exp.( - im * tf * eigs)
    expEt = Diagonal(phases)
    final = U * expEt * U' * (init.vector)
    return State(init.basis, final)
end

# Evolve state for multiple time steps and record observables
function timeEvolve(ops::OpSum, init::AbstractState, ts::AbstractVector, obs::AbstractObserver, ::Val{:exact})
    hmat = makeHamiltonian(ops, init.basis)
    eigs, U = eigen!(Hermitian(hmat))
    dim = length(eigs)
    replace!(x -> isapprox(x, 0; atol=2*eps(Float64)) ? 0.0 : x, U)

    psi = ComplexF64.(init.vector)
    init_trans = U' * psi
    phases = Vector{ComplexF64}(undef, dim)
    psi_trans = Vector{ComplexF64}(undef, dim)

    record!(obs, psi, 1)
    for (i, t) in enumerate(ts[2:end])
        phases .= exp.( - im * t * eigs)
        psi_trans .= phases .* init_trans
        mul!(psi, U, psi_trans)
        record!(obs, psi, i + 1)
    end
    return State(init.basis, psi)
end

# Default method: exact diagonalization
timeEvolve(ops::OpSum, init::AbstractState, ts::AbstractVector, obs::AbstractObserver) = timeEvolve(ops, init, ts, obs, Val(:exact))