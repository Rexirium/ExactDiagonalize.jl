using MKL, LinearAlgebra
using SparseArrays

include("utils.jl")
include("state_basis.jl")

abstract type AbstractOpSum end

mutable struct SpinOpSum <: AbstractOpSum
    type::DataType
    opvec::AbstractVector
end

function act(opstr::Symbol, loc::Int, bits::Int, T::DataType)
    """
    act a single qubit operator on the state `bits`=|1001011⟩ for bits=(1001011)₂
    |1⟩ = (1, 0)ᵀ = |↑⟩, |0⟩ = (0, 1)ᵀ = |↓⟩
    I do not specify the Y operator (has complex element) to keep type stability, but use iY instead.
    """
    if opstr == :Z
        return bits, T(2 * readbit(bits, loc) - 1)
    elseif opstr == :X
        return flip(bits, loc), one(T)
    elseif opstr == :iY # means simplectic matrix [0 1 ; -1 0], Sp = iY
        return flip(bits, loc), T(1 - 2 * readbit(bits, loc))
    elseif opstr == :σp
        return flip(bits, loc), T(! readbit(bits, loc))
    elseif opstr == :σm
        return flip(bits, loc), T(readbit(bits, loc))
    else
        error("Operator not specified yet!")
    end
end

function act(opstr::String, loc::Tuple{Int, Int}, bits::Int, T::DataType)
    if opstr == :CX
        c, t = loc
        bitc = readbit(bits, c)
        return flip(bits, t, bitc), one(T)
    elseif opstr == :CZ
        c, t = loc
        i1, i2 = minmax(c, t)
        b1, b2 = readbit(bits, i1, i2)
        return bits, T(2 * (b1 ^ b2) - 1)
    else
        error("Operator not specified yet!")
    end
end

function apply(op::Tuple, bits::Int, T::DataType)
    element = op[1]
    newbits = bits
    oplen = length(op)

    for s in 2:2:oplen
        tmp = act(op[s], op[s+1], newbits, T)
        newbits = tmp[1]
        element *= tmp[2]
    end
    return newbits, element
end

function makeHamiltonian(ops::AbstractOpSum, basis::AbstractBasis)
    dim = length(basis.bitsvec)
    hmat = zeros(ops.type, dim, dim)
    for (j, bits) in enumerate(basis.bitsvec)
        for op in ops.opvec
            newbits, element = apply(op, bits, ops.type)
            i = findindex(basis, newbits)
            (i == -1 || element == 0) && continue
            hmat[i, j] += element
        end
    end
    return Hermitian(hmat)
end

function timeEvolve(ops::AbstractOpSum, init::AbstractState, tf::Real)
    hmat = makeHamiltonian(ops, init.basis)
    eigenergy, U = eigen(hmat)
    phases = cos.(tf * eigenergy) .- im * sin.(tf * energy)
    expEt = Diagonal(phases)
    final = U * expEt * U' * (init.statevec)
    return State(init.basis, final)
end

function timeEvolve(ops::AbstractOpSum, init::AbstractState, ts::Vector{<:Real})
    hmat = makeHamiltonian(ops, init.basis)
    eigenergy, U = eigen(hmat)
    phases = cos.(eigenergy) .- im * sin.(eigenergy)
    expEt = Diagonal(phases)
    final = Vector{ComplexF64}(undef, length(init.statevec))

    for t in ts
        phases .= cos.(t * eigenergy) .- im * sin.(t * eigenergy)
        expEt[diagind(expEt)] .= phases
        final .= U * expEt * U' * (init.statevec)
    end
end

let 
    L, N = 6, 3

    os = Tuple[]
    for j in 1:L
        nj = mod1(j+1, L)
        push!(os, (1.0, :Z, j, :Z, nj))
        push!(os, (1.0, :X, j, :X, nj))
        push!(os, (-1.0, :iY, j, :iY, nj))
    end
    # push!(os, (1.0, :X, L))
    ops = SpinOpSum(Float64, os)

    basis = NumBasis(L, N)
    @time makeHamiltonian(ops, basis)
    
end
