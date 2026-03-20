using MKL, LinearAlgebra
using SparseArrays

include("utils.jl")
include("state_basis.jl")

const SpMatrix = SparseMatrixCSC
const _systype = Ref{Val}(Val(:Spin))

function set_systype(val::Symbol)
    _systype[] = Val(val)
end

get_systype() = _systype[]

abstract type AbstractOp end

struct SpinOp <: AbstractOp
    name::Symbol
    loc::Union{Int, Tuple{Int, Int}}
end
Operator(name::Symbol, loc::Union{Int, Tuple{Int, Int}}, ::Val{:Spin}) = SpinOp(name, loc)

mutable struct OpSum{T <: Number}
    covec::Vector{T}
    opvec::Vector{Vector{<:AbstractOp}}
end

function os2ops(os::Tuple)
    len = length(os)
    ops = SpinOp[]
    sizehint!(ops, len ÷ 2)
    for s in 2:2:len
        push!(ops, Operator(os[s], os[s+1], _systype[]))
    end
    return ops
end

function OpSum(osvec::Vector{<:Tuple}, eltype::DataType)
    covec = Vector{eltype}()
    opvec = Vector{SpinOp}[]
    for os in osvec
        ops = os2ops(os)
        push!(covec, os[1])
        push!(opvec, ops)
    end
    return OpSum{eltype}(covec, opvec)
end

@inline function act(op::SpinOp, bits::Int, T::DataType)
    """
    act a single qubit operator on the state `bits`=|1001011⟩ for bits=(1001011)₂
    |1⟩ = (1, 0)ᵀ = |↑⟩, |0⟩ = (0, 1)ᵀ = |↓⟩
    I do not specify the Y operator (has complex element) to keep type stability, but use iY instead.
    """
    if op.name == :Z
        return bits, T(2 * readbit(bits, op.loc) - 1)
    elseif op.name == :X
        return flip(bits, op.loc), one(T)
    elseif op.name == :iY # means simplectic matrix [0 1 ; -1 0] = iY
        return flip(bits, op.loc), T(1 - 2 * readbit(bits, op.loc))
    elseif op.name == :σp
        return flip(bits, op.loc), T(! readbit(bits, op.loc))
    elseif op.name == :σm
        return flip(bits, op.loc), T(readbit(bits, op.loc))
    elseif op.name == :CX
        c, t = op.loc
        return flip(bits, t, readbit(bits, c)), one(T)
    elseif op.name == :CZ
        b1, b2 = readbit(bits, minmax(op.loc...))
        return bits, T(2 * (b1 ^ b2) - 1)
    else
        error("Operator not specified yet!")
    end
end


function apply(coef::T, ops::Vector{<:AbstractOp}, bits::Int) where T <: Number
    element = coef
    newbits = bits

    for op in ops
        tmp = act(op, newbits, T)
        newbits = tmp[1]
        element *= tmp[2]
    end
    return newbits, element
end

function op2mat(coeff::T, ops::Vector{<:AbstractOp}, basis::AbstractBasis; sparsed::Bool=true) where T <: Number
    dim = length(basis.bitsvec)
    opmat = sparsed ? spzeros(T, dim, dim) : zeros(T, dim, dim)

    for (j, bits) in enumerate(basis.bitsvec)
        newbits, element = apply(coeff, ops, bits)
        i = findindex(basis, newbits)
        (i <= 0 || iszero(element)) && continue
        opmat[i, j] += element
    end
    return opmat
end

function apply(ops::Vector{<:AbstractOp}, psi::AbstractState, coeff::Number=1.0)
    opmat = op2mat(coeff, ops, psi.basis)
    vector = opmat * psi.vector
    return State(psi.basis, vector)
end

function apply!(ops::Vector{<:AbstractOp}, psi::AbstractState, coeff::Number=1.0)
    opmat = op2mat(coeff, ops, psi.basis)
    lmul!(opmat, psi.vector)
end

function expected(ops::Vector{<:AbstractOp}, psi::AbstractState, coeff::Number=1.0)
    opmat = op2mat(coeff, ops, psi.basis)
    v = psi.vector
    return real(v' * opmat * v)
end

function inner(x::S, ops::Vector{<:AbstractOp}, y::S, coeff::Number=1.0) where S <: AbstractState
    length(x.vector) == length(y.vector) || error("wrong dimension of two states!")
    opmat = op2mat(coeff, ops, y.basis)
    return x.vector' * opmat * y.vector
end

function makeHamiltonian(opsum::OpSum{T}, basis::AbstractBasis; sparsed::Bool=false) where T <: Number
    dim = length(basis.bitsvec)
    opnum = length(opsum.covec)
    covec = opsum.covec
    opvec = opsum.opvec

    hmat = sparsed ? spzeros(T, dim, dim) : zeros(T, dim, dim) 
    @inbounds for (j, bits) in enumerate(basis.bitsvec)
        for s in 1:opnum
            newbits, element = apply(covec[s], opvec[s], bits)
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

    OperatorObserver(os::Tuple, basis::AbstractBasis; sparsed::Bool=true) = new{typeof(os[1])}(
        op2mat(os[1], os2ops(os), basis; sparsed=sparsed), Vector{Float64}()
    )
end

function record!(obs::OperatorObserver, psi::AbstractVector)
    val = real(psi' * obs.opmat * psi)
    push!(obs.data, val)
end

mutable struct OpSumObserver{T <: Number} <: AbstractObserver
    opsmat::SpMatrix{T}
    data::Vector{Float64}

    OpSumObserver(ops::OpSum{T}, basis::AbstractBasis; sparsed::Bool=true) where T <: Number = new{T}(
        makeHamiltonian(ops, basis; sparsed=sparsed), Vector{Float64}()
    )
end

function record!(obs::OpSumObserver, psi::AbstractVector)
    val = real(psi' * obs.opsmat * psi)
    push!(obs.data, val)
end