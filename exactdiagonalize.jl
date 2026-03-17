using CairoMakie
include("operators.jl")

function spectrum(ops::AbstractOpSum, basis::AbstractBasis)
    hmat = makeHamiltonian(ops, basis)
    return eigvals(hmat)
end

function spectrum_numconserved(ops::AbstractOpSum, lsize::Int)
    energies = Float64[]
    sizehint!(energies, 1 << lsize)
    for num in 0:lsize
        basis = NumBasis(lsize, num)
        hmat = makeHamiltonian(ops, basis)
        eigs = eigvals!(hmat)
        append!(energies, eigs)
    end
    return energies
end

function timeEvolve(ops::AbstractOpSum, init::AbstractState, tf::Real)
    hmat = makeHamiltonian(ops, init.basis)
    eigenergy, U = eigen(hmat)
    phases = complex.(cos.(tf * eigenergy), - sin.(tf * eigenergy))
    expEt = Diagonal(phases)
    final = U * expEt * U' * (init.vector)
    return State(init.basis, final)
end

function timeEvolve(ops::AbstractOpSum, init::AbstractState, ts::AbstractVector, obs::AbstractObserver)
    hmat = makeHamiltonian(ops, init.basis)
    dim = length(init.vector)
    eigenergy, U = eigen(hmat)

    phases = Vector{ComplexF64}(undef, dim)
    expEt = Diagonal{ComplexF64}(undef, dim)
    psi = Vector{ComplexF64}(undef, dim)

    for t in ts
        phases .= complex.(cos.(t * eigenergy), - sin.(t * eigenergy))
        expEt.diag .= phases
        psi .= U * expEt * U' * (init.vector)
        record!(obs, psi)
    end
    return State(init.basis, psi)
end

let 
    L, N = 10, 1
    
    init = NumState("0000000001")

    os = Tuple[]
    for j in 1:L
        nj = mod1(j+1, L)
        push!(os, (1.0, :Z, j, :Z, nj))
        push!(os, (1.0, :X, j, :X, nj))
        push!(os, (-1.0, :iY, j, :iY, nj))
    end
    # push!(os, (1.0, :X, L))
    ops = SpinOpSum(Float64, os)

    obs = OperatorObserver((1.0, :Z, 1), init.basis)
    
    
    ts = 0.0:0.01:10.0
    timeEvolve(ops, init, ts, obs)
    
    fig, ax = lines(ts, obs.data)
    
end