exact() = Val(:exact)

function spectrum(ops::OpSum, basis::AbstractBasis)
    hmat = makeHamiltonian(ops, basis)
    return eigvals(Hermitian(hmat))
end

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

function timeEvolve(ops::OpSum, init::AbstractState, tf::Real)
    hmat = makeHamiltonian(ops, init.basis)
    eigs, U = eigen(Hermitian(hmat))
    phases = complex.(cos.(tf * eigs), - sin.(tf * eigs))
    expEt = Diagonal(phases)
    final = U * expEt * U' * (init.vector)
    return State(init.basis, final)
end

function timeEvolve(ops::OpSum, init::AbstractState, ts::AbstractVector, obs::AbstractObserver, ::Val{:exact})
    hmat = makeHamiltonian(ops, init.basis)
    eigs, U = eigen(Hermitian(hmat))
    dim = length(eigs)

    psi = ComplexF64.(init.vector)
    init_trans = U' * psi
    phases = Vector{ComplexF64}(undef, dim)
    psi_trans = Vector{ComplexF64}(undef, dim)

    record!(obs, psi)
    for t in ts[2:end]
        phases .= complex.(cos.(t * eigs), - sin.(t * eigs))
        psi_trans .= phases .* init_trans
        mul!(psi, U, psi_trans)
        record!(obs, psi)
    end
    return State(init.basis, psi)
end

#default method is ED
timeEvolve(ops::OpSum, init::AbstractState, ts::AbstractVector, obs::AbstractObserver) = timeEvolve(ops, init, ts, obs, Val(:exact))