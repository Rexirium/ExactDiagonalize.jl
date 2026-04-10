# Include utility and basis definitions
include("utils.jl")
include("state_basis.jl")

# System type (e.g., :Spin)
const _systype = Ref{Val}(Val(:Spin))

# Set the system type
function set_systype(val::Symbol)
    _systype[] = Val(val)
end

# Get the system type
get_systype() = _systype[]

# Abstract operator type
abstract type AbstractOp end

# Spin operator (e.g., X, iY, Z, CX, CZ)
struct SpinOp <: AbstractOp
    name::Symbol
    loc::Union{UInt8, Tuple{UInt8, UInt8}}

    SpinOp(name::Symbol, loc::Int) = new(name, loc % UInt8)
    SpinOp(name::Symbol, loc::Tuple{Int, Int}) = new(name, loc .% UInt8)
end

# Decide which type of operator to take
get_optype(::Val{:Spin}) = SpinOp

# Sum of operators with coefficients
mutable struct OpSum{T <: Number, O <: AbstractOp}
    covec::Vector{T}
    opvec::Vector{Vector{O}}
end
OpSum(T::DataType, O) = OpSum{T, O}(T[], Vector{O}[])
OpSum(T::DataType) = OpSum(T, get_optype(_systype[]))

# Convert tuple to list of SpinOps
function os2ops(os::Tuple, optype::DataType)
    len = length(os)
    ops = optype[]
    sizehint!(ops, len ÷ 2)
    for s in 2:2:len
        push!(ops, optype(os[s], os[s+1]))
    end
    return ops
end

# Construct OpSum from tuples and element type
function OpSum(osvec::Vector{<:Tuple}, eltype::DataType)
    covec = Vector{eltype}()
    optype = get_optype(_systype[])
    opvec = Vector{optype}[]

    for os in osvec
        ops = os2ops(os, optype)
        push!(covec, os[1])
        push!(opvec, ops)
    end
    return OpSum{eltype, optype}(covec, opvec)
end

function Base.:+(opsum::OpSum{T, O}, os::Tuple) where {T <: Number, O <: AbstractOp}
    push!(opsum.covec, os[1])
    push!(opsum.opvec, os2ops(os, O))
    opsum
end

function Base.:+(ops1::OpSum{T, O}, ops2::OpSum{S, O}) where {T <: Number, S <: Number, O <: AbstractOp}
    covec = vcat(ops1.covec, ops2.covec)
    opvec = vcat(ops1.opvec, ops2.opvec)
    return OpSum{promote_type(T, S), O}(covec, opvec)
end


"""
act a single qubit operator on the state `bits`=|1001011⟩ for bits=(1001011)₂
|1⟩ = (1, 0)ᵀ = |↑⟩, |0⟩ = (0, 1)ᵀ = |↓⟩
I do not specify the Y operator (has complex element) to keep type stability, but use iY instead.
"""
# Wait for later development on Fermion Operators
@inline function act(op::SpinOp, bits::UInt32)::Tuple{UInt32, Int}
    if op.name == :Z
        return bits, 2 * readbit(bits, op.loc) - 1
    elseif op.name == :X
        return flip(bits, op.loc), 1
    elseif op.name == :iY # means simplectic matrix [0 1 ; -1 0] = iY
        return flip(bits, op.loc), 1 - 2 * readbit(bits, op.loc)
    elseif op.name == :σp
        return flip(bits, op.loc), Int(! readbit(bits, op.loc))
    elseif op.name == :σm
        return flip(bits, op.loc), Int(readbit(bits, op.loc))
    elseif op.name == :CX
        c, t = op.loc
        return flip(bits, t, readbit(bits, c)), 1
    elseif op.name == :CZ
        b1, b2 = readbit(bits, minmax(op.loc...))
        return bits, 2 * (b1 ^ b2) - 1
    else
        error("Operator not specified yet!")
    end
end

# Apply a sequence of operators to a bitstring
function apply(coef::T, ops::Vector{O}, bits::UInt32) where {T <: Number, O <: AbstractOp}
    element = coef
    newbits = bits

    for op in ops
        tmp = act(op, newbits)
        newbits = tmp[1]
        element *= tmp[2]
    end
    return newbits, element
end

# Build operator matrix in given basis
# Outer function for flexibility
function op2mat(coeff::Number, ops::Vector{<:AbstractOp}, basis::AbstractBasis; sparsed::Bool=true)
    _op2mat(coeff, ops, basis, sparsed)
end

# Inner function for type stability
@inline function _op2mat(coeff::T, ops::Vector{O}, basis::B, sparsed::Bool) where {T, O, B}
    dim = length(basis.bitsvec)
    opmat = sparsed ? spzeros(T, dim, dim) : zeros(T, dim, dim)
    @inbounds for (j, bits) in enumerate(basis.bitsvec)
        newbits, element = apply(coeff, ops, bits)
        i = findindex(basis, newbits)
        (i > dim || iszero(element)) && continue
        opmat[i, j] += element
    end
    return opmat
end

# Apply operator(s) to a state and return new state
function apply(ops::Vector{O}, psi::QState, coeff::Number=1.0) where O <: AbstractOp
    opmat = op2mat(coeff, ops, psi.basis)
    vector = opmat * psi.vector
    return QState(psi.basis, vector)
end

# In-place apply operator(s) to a state
function apply!(ops::Vector{O}, psi::QState, coeff::Number=1.0) where O <: AbstractOp
    opmat = op2mat(coeff, ops, psi.basis)
    lmul!(opmat, psi.vector)
end

# Compute expectation value of operator(s) in a state
function expected(ops::Vector{O}, psi::QState, coeff::Number=1.0) where O <: AbstractOp
    opmat = op2mat(coeff, ops, psi.basis)
    v = psi.vector
    return real(v' * opmat * v)
end

# Compute ⟨x|O|y⟩ for two states and operator(s)
function LinearAlgebra.dot(x::QState, ops::Tuple{T, Vector{O}}, y::QState) where {T <: Number, O <: AbstractOp}
    length(x.vector) == length(y.vector) || error("wrong dimension of two states!")
    opmat = op2mat(ops[1], ops[2], y.basis)
    return dot(x.vector, opmat, y.vector)
end

"""
Construct the hamiltonian matrix from OpSum type with assigned basis.
Return either dense or sparse matrix controled by sparsed, default to be dense
because `eigen` in LinearAlgebra does not support sparse matrix.
"""
function makeHamiltonian(opsum::OpSum{T}, basis::B; sparsed::Bool=false) where {T <: Number, B <: AbstractBasis}
    dim = length(basis.bitsvec)
    opnum = length(opsum.covec)
    covec = opsum.covec
    opvec = opsum.opvec

    hmat = sparsed ? spzeros(T, dim, dim) : zeros(T, dim, dim) 
    @inbounds for (j, bits) in enumerate(basis.bitsvec)
        for s in 1:opnum
            newbits, element = apply(covec[s], opvec[s], bits)
            i = findindex(basis, newbits)
            (i > dim || iszero(element)) && continue
            hmat[i, j] += element
        end
    end
    return hmat
end

function apply(opsum::OpSum, psi::QState)
    hmat = makeHamiltonian(opsum, psi.basis; sparsed=true)
    vector = hmat * psi.vector
    return QState(psi.basis, vector)
end

function apply!(opsum::OpSum, psi::QState)
    hmat = makeHamiltonian(opsum, psi.basis; sparsed=true)
    lmul!(hmat, psi.vector)
end

function expected(opsum::OpSum, psi::QState)
    hmat = makeHamiltonian(opsum, psi.basis; sparsed=true)
    return real(dot(psi.vector, opsum, psi.vector))
end

function LinearAlgebra.dot(x::QState, opsum::OpSum, y::QState)
    hmat = makeHamiltonian(opsum, y.basis; sparsed=true)
    return dot(x.vector, hmat, y.vector)
end
