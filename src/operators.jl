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

@enum OpName OP_Z OP_X OP_iY OP_σp OP_σm OP_CX OP_CZ OP_UNKNOWN
function symbol2opname(s::Symbol)
    s === :Z  ? OP_Z  :
    s === :X  ? OP_X  :
    s === :iY ? OP_iY :
    s === :σp ? OP_σp :
    s === :σm ? OP_σm :
    s === :CX ? OP_CX :
    s === :CZ ? OP_CZ : OP_UNKNOWN
end

# Spin operator (e.g., X, iY, Z, CX, CZ)
struct SpinOp <: AbstractOp
    name::OpName
    loc1::UInt8
    loc2::UInt8

    SpinOp(name::Symbol, loc::Int) = new(symbol2opname(name), loc % UInt8, 0x00)
    SpinOp(name::Symbol, loc::Tuple{Int, Int}) = new(symbol2opname(name), loc[1] % UInt8, loc[2] % UInt8)
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
OpSum() = OpSum(Float64)

# Convert tuple to list of SpinOps
function os2ops(os::Tuple, optype::DataType)
    len = length(os)
    ops = optype[]
    sizehint!(ops, len ÷ 2)
    for s in 2:2:len
        loc = os[s + 1]
        push!(ops, optype(os[s], loc))
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

function Base.:+(opsum::OpSum{<:Number, O}, os::Tuple) where  O <: AbstractOp
    push!(opsum.covec, os[1])
    push!(opsum.opvec, os2ops(os, O))
    opsum
end

function Base.:+(ops1::OpSum{T, O}, ops2::OpSum{S, O}) where {T <: Number, S <: Number, O <: AbstractOp}
    R = promote_type(T, S)
    covec = Vector{R}(vcat(ops1.covec, ops2.covec))
    opvec = vcat(ops1.opvec, ops2.opvec)
    return OpSum{R, O}(covec, opvec)
end


"""
act a single qubit operator on the state `bits`=|1001011⟩ for bits=(1001011)₂
|1⟩ = (1, 0)ᵀ = |↑⟩, |0⟩ = (0, 1)ᵀ = |↓⟩
I do not specify the Y operator (has complex element) to keep type stability, but use iY instead.
"""
# Wait for later development on Fermion Operators
@inline function act(op::SpinOp, bits::UInt32)::Tuple{UInt32, Int}
    if op.name == OP_Z
        return bits, 2 * readbit(bits, op.loc1) - 1
    elseif op.name == OP_X
        return flip(bits, op.loc1), 1
    elseif op.name == OP_iY # means simplectic matrix [0 1 ; -1 0] = iY
        return flip(bits, op.loc1), 1 - 2 * readbit(bits, op.loc1)
    elseif op.name == OP_σp
        return flip(bits, op.loc1), Int(! readbit(bits, op.loc1))
    elseif op.name == OP_σm
        return flip(bits, op.loc1), Int(readbit(bits, op.loc1))
    elseif op.name == OP_CX
        c, t = op.loc1, op.loc2
        return flip(bits, t, readbit(bits, c)), 1
    elseif op.name == OP_CZ
        b1, b2 = readbit(bits, minmax(op.loc1, op.loc2))
        return bits, 2 * (b1 ^ b2) - 1
    else
        error("Operator not specified yet!")
    end
end

# Apply a sequence of operators to a bitstring
function apply(coef::Number, ops::Vector{<:AbstractOp}, bits::UInt32)
    element = coef
    newbits = bits

    @inbounds for op in ops
        newbits, ele_mul = act(op, newbits)
        element *= ele_mul
        iszero(element) && break
    end
    return newbits, element
end

# Build operator matrix in given basis
function op2mat(coeff::T, ops::Vector{<:AbstractOp}, basis::SpinBasis{N, Nothing}; 
    sparsed::Bool=true, dtype::DataType=Float64) where {T <: Number, N}
    dim = length(basis.bitsvec)
    eltype = promote_type(T, dtype)
    opmat = sparsed ? spzeros(eltype, dim, dim) : zeros(eltype, dim, dim)

    @inbounds for (j, bits) in enumerate(basis.bitsvec)
        newbits, element = apply(coeff, ops, bits)
        i = findindex(basis, newbits)
        (i > dim || iszero(element)) && continue
        opmat[i, j] += element
    end
    return opmat
end

function op2mat(coeff::T, ops::Vector{<:AbstractOp}, basis::SpinBasis{Nothing, Int}; 
    sparsed::Bool=true, dtype::DataType=Float64) where {T <: Number}
    dim = length(basis.bitsvec)
    
    if basis.kint == 0 || basis.kint == basis.lsize / 2
        eltype = promote_type(T, dtype)
    else
        eltype = ComplexF64
    end
    opmat = sparsed ? spzeros(eltype, dim, dim) : zeros(eltype, dim, dim)
    ks = 2π * basis.kint / basis.lsize
    orbits = basis.orbsize

    @inbounds for (j, bits) in enumerate(basis.bitsvec)
        newbits, element = apply(coeff, ops, bits)
        i, d = findindex(basis, newbits)
        (i > dim || iszero(element)) && continue

        norm_factor = sqrt(orbits[j] / orbits[i])
        opmat[i, j] += element * cis(-ks * d) * norm_factor
    end
    return opmat
end


# Apply operator(s) to a state and return new state
function apply(ops::Vector{<:AbstractOp}, psi::QState, coeff::Number=1.0)
    opmat = op2mat(coeff, ops, psi.basis)
    vector = opmat * psi.vector
    return QState(psi.basis, vector)
end

# In-place apply operator(s) to a state
function apply!(ops::Vector{<:AbstractOp}, psi::QState, coeff::Number=1.0)
    opmat = op2mat(coeff, ops, psi.basis)
    lmul!(opmat, psi.vector)
end

# Compute expectation value of operator(s) in a state
function expected(ops::Vector{<:AbstractOp}, psi::QState, coeff::Number=1.0)
    opmat = op2mat(coeff, ops, psi.basis)
    v = psi.vector
    return real(dot(v, opmat, v))
end

# Compute ⟨x|O|y⟩ for two states and operator(s)
function LinearAlgebra.dot(x::QState, ops::Tuple{Number, Vector{<:AbstractOp}}, y::QState)
    length(x.vector) == length(y.vector) || error("wrong dimension of two states!")
    opmat = op2mat(ops[1], ops[2], y.basis)
    return dot(x.vector, opmat, y.vector)
end

"""
Construct the hamiltonian matrix from OpSum type with assigned basis.
Return either dense or sparse matrix controled by sparsed, default to be dense
because `eigen` in LinearAlgebra does not support sparse matrix.
"""
function makeHamiltonian(opsum::OpSum{T}, basis::SpinBasis{N, Nothing}; 
    sparsed::Bool=false, dtype::DataType=Float64) where {T <: Number, N}
    dim = length(basis.bitsvec)
    opnum = length(opsum.covec)
    covec = opsum.covec
    opvec = opsum.opvec

    eltype = promote_type(T, dtype)
    hmat = sparsed ? spzeros(eltype, dim, dim) : zeros(eltype, dim, dim) 
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

function makeHamiltonian(opsum::OpSum{T}, basis::SpinBasis{Nothing, Int}; 
    sparsed::Bool=false, dtype::DataType=Float64) where T <: Number
    dim = length(basis.bitsvec)
    opnum = length(opsum.covec)
    covec = opsum.covec
    opvec = opsum.opvec
    ks = 2π * basis.kint / basis.lsize
    orbits = basis.orbsize

    if basis.kint == 0 || basis.kint == basis.lsize / 2
        eltype = promote_type(T, dtype)
    else
        eltype = ComplexF64
    end
    hmat = sparsed ? spzeros(eltype, dim, dim) : zeros(eltype, dim, dim)
    @inbounds for (j, bits) in enumerate(basis.bitsvec)
        for s in 1:opnum
            newbits, element = apply(covec[s], opvec[s], bits)
            i, d = findindex(basis, newbits)
            (i > dim || iszero(element)) && continue
            
            norm_factor = sqrt(orbits[j] / orbits[i])
            hmat[i, j] += element * cis(-ks * d) * norm_factor
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
    return real(dot(psi.vector, hmat, psi.vector))
end

function LinearAlgebra.dot(x::QState, opsum::OpSum, y::QState)
    hmat = makeHamiltonian(opsum, y.basis; sparsed=true)
    return dot(x.vector, hmat, y.vector)
end
