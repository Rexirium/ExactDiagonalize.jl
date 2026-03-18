using MKL, LinearAlgebra
using SparseArrays

include("utils.jl")
include("state_basis.jl")

global SpMatrix = SparseMatrixCSC

abstract type AbstractOpSum end

mutable struct SpinOpSum{T <: Number} <: AbstractOpSum
    opvec::Vector{<:Tuple}
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

function act(opstr::Symbol, loc::Tuple{Int, Int}, bits::Int, T::DataType)
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

function apply(op::Tuple, psi::AbstractState)
    opmat = op2mat(op, psi.basis)
    vector = opmat * psi.vector
    return State(psi.basis, vector)
end

function apply!(op::Tuple, psi::AbstractState)
    opmat = op2mat(op, psi.basis)
    lmul!(opmat, psi.vector)
end

function op2mat(op::Tuple, basis::AbstractBasis; sparsed::Bool=true)
    dim = length(basis.bitsvec)
    T = typeof(op[1])
    opmat = sparsed ? spzeros(T, dim, dim) : zeros(T, dim, dim)
    for (j, bits) in enumerate(basis.bitsvec)
        newbits, element = apply(op, bits, T)
        i = findindex(basis, newbits)
        (i <= 0 || iszero(element)) && continue
        opmat[i, j] += element
    end
    return opmat
end

function expected(op::Tuple, psi::AbstractState)
    opmat = op2mat(op, psi.basis)
    v = psi.vector
    return real(v' * opmat * v)
end

function inner(x::S, op::Tuple, y::S) where S <: AbstractState
    length(x.vector) == length(y.vector) || error("wrong dimension of two states!")
    opmat = op2mat(op, y.basis)
    return x.vector' * opmat * y.vector
end

function makeHamiltonian(ops::SpinOpSum{T}, basis::AbstractBasis; sparsed::Bool=false) where T <: Number
    dim = length(basis.bitsvec)
    hmat = sparsed ? spzeros(T, dim, dim) : zeros(T, dim, dim) 
    for (j, bits) in enumerate(basis.bitsvec)
        for op in ops.opvec
            newbits, element = apply(op, bits, T)
            i = findindex(basis, newbits)
            (i <= 0 || iszero(element)) && continue
            hmat[i, j] += element
        end
    end
    return hmat
end

#=============Obsrever system to record quantities during evolution=============#
abstract type AbstractObserver end

mutable struct OperatorObserver{T <: Number} <: AbstractObserver
    opmat::SpMatrix{T}
    data::Vector{Float64}

    OperatorObserver(op::Tuple, basis::AbstractBasis; sparsed::Bool=true) = new{typeof(op[1])}(
        op2mat(op, basis; sparsed=sparsed), Vector{Float64}()
    )
end

function record!(obs::OperatorObserver, psi::AbstractVector)
    val = real(psi' * obs.opmat * psi)
    push!(obs.data, val)
end

mutable struct OpSumObserver{T <: Number} <: AbstractObserver
    opsmat::SpMatrix{T}
    data::Vector{Float64}

    OpSumObserver(ops::SpinOpSum{T}, basis::AbstractBasis; sparsed::Bool=true) where T <: Number = new{T}(
        makeHamiltonian(ops, basis; sparsed=sparsed), Vector{Float64}()
    )
end

function record!(obs::OpSumObserver, psi::AbstractVector)
    val = real(psi' * obs.opsmat * psi)
    push!(obs.data, val)
end