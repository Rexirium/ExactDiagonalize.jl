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
abstract type SpinOp <: AbstractOp end

@enum SpinOp1Name OP_Z OP_X OP_iY OP_σp OP_σm OP_Pup OP_Pdn OP_UNKNOWN
@enum SpinOp2Name OP_CX OP_CZ

const opDict = Dict{Symbol, Enum}(
    :Z  => OP_Z,
    :X  => OP_X,
    :iY => OP_iY,
    :σp => OP_σp,
    :σm => OP_σm,
    :Pup => OP_Pup,
    :Pdn => OP_Pdn,
    :CX => OP_CX,
    :CZ => OP_CZ
)

struct SpinOp1 <: SpinOp
    name::SpinOp1Name
    loc::UInt8
end

struct SpinOp2 <: SpinOp
    name::SpinOp2Name
    loc1::UInt8
    loc2::UInt8
end

Op(name::SpinOp1Name, loc::Int) = SpinOp1(name, loc % UInt8)
Op(name::SpinOp2Name, locs::Tuple{Int, Int}) = SpinOp2(name, locs[1] % UInt8, locs[2] % UInt8)
Op(name::Symbol, loc::Union{Int, Tuple{Int,Int}}) = Op(opDict[name], loc)

# Decide which type of operator to take
get_optype(::Val{:Spin}) = SpinOp

# Sum of operators with coefficients
mutable struct OpSum{T <: Number, O <: AbstractOp}
    covec::Vector{T}
    opvec::Vector{Vector{O}}
end
OpSum(ELT::Type{<:Number}, OT::Type{<:AbstractOp}) = OpSum{ELT, OT}(ELT[], Vector{OT}[])
OpSum(ELT::Type{<:Number}) = OpSum(ELT, get_optype(_systype[]))
OpSum() = OpSum(Float64, get_optype(_systype[]))

# Convert tuple to list of SpinOps
function os2ops(os::Tuple, optype::Type{<:AbstractOp})
    len = length(os)
    ops = optype[]
    sizehint!(ops, len ÷ 2)
    for s in 2:2:len
        loc = os[s + 1]
        push!(ops, Op(os[s], loc))
    end
    return ops
end

# Construct OpSum from tuples and element type
OpSum(osvec::Vector{<:Tuple}) = OpSum(Float64, osvec)
function OpSum(ELT::Type{<:Number}, osvec::Vector{<:Tuple})
    covec = Vector{ELT}()
    optype = get_optype(_systype[])
    opvec = Vector{optype}[]

    for os in osvec
        ops = os2ops(os, optype)
        push!(covec, os[1])
        push!(opvec, ops)
    end
    return OpSum{ELT, optype}(covec, opvec)
end

function Base.:+(opsum::OpSum{<:Number, OT}, os::Tuple) where  OT <: AbstractOp
    push!(opsum.covec, os[1])
    push!(opsum.opvec, os2ops(os, OT))
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
@inline function act(op::SpinOp1, bits::UInt32)::Tuple{UInt32, Int}
    if op.name == OP_Z
        return bits, 2 * readbit(bits, op.loc) - 1
    elseif op.name == OP_X
        return flip(bits, op.loc), 1
    elseif op.name == OP_iY # means simplectic matrix [0 1 ; -1 0] = iY
        return flip(bits, op.loc), 1 - 2 * readbit(bits, op.loc)
    elseif op.name == OP_σp
        return flip(bits, op.loc), Int(! readbit(bits, op.loc))
    elseif op.name == OP_σm
        return flip(bits, op.loc), Int(readbit(bits, op.loc))
    elseif op.name == OP_Pup
        return bits, Int(readbit(bits, op.loc))
    elseif op.name == OP_Pdn
        return bits, Int(! readbit(bits, op.loc))
    else
        error("Operator not specified yet!")
    end
end

@inline function act(op::SpinOp2, bits::UInt32)::Tuple{UInt32, Int}
    if op.name == OP_CX
        c, t = op.loc1, op.loc2
        return flip(bits, t, readbit(bits, c)), 1
    elseif op.name == OP_CZ
        b1, b2 = readbit(bits, minmax(op.loc1, op.loc2))
        return bits, 2 * (b1 & b2) - 1 
    else
        error("Operator not specified yet!")
    end
end

# Apply a sequence of operators to a bitstring
function act_seq(coeff::T, ops::Vector{<:AbstractOp}, bits::UInt32) where T <: Number
    element = coeff
    newbits = bits

    for op in ops
        newbits, ele_mul = act(op, newbits)
        element *= ele_mul
        iszero(element) && break
    end
    return newbits, element
end

# Multiple dispatch the acting on different basis
@inline function basis_element(basis::AbstractBasis{N, Nothing}, newbits, j) where {N <: NVInt}
    i, _ = findindex(basis, newbits)
    return i, 1.0
end
# basis with spatial translation invariance
@inline function basis_element(basis::AbstractBasis{N, Int}, newbits, j) where {N <: NVInt}
    i, d = findindex(basis, newbits)
    i > basis.dim && return i, 0.0im

    k = 2π * basis.kint / basis.lsize
    factor = cis(-k * d) * sqrt(basis.orbsize[j] / basis.orbsize[i])

    return i, factor
end

# Determine the element type of the matrix form of oerators
operator_eltype(::AbstractBasis{N, Nothing}, CT::Type{<:Number}, DT::Type{<:Number}) where {N} =
    promote_type(CT, DT, Float64)

operator_eltype(basis::AbstractBasis{N, Int}, CT::Type{<:Number}, DT::Type{<:Number}) where {N} =
    basis.kint == 0 || abs(basis.kint) == basis.lsize / 2 ? promote_type(CT, DT, Float64) : ComplexF64

# Build operator matrix in given basis
function matrixform(ops::Vector{<:AbstractOp}, basis::AbstractBasis, coeff::Number=1; 
    sparsed::Bool=true, dtype::Type{<:Number}=Float64)
    dim = basis.dim
    ELT = operator_eltype(basis, typeof(coeff), dtype)
    opmat = sparsed ? spzeros(ELT, dim, dim) : zeros(ELT, dim, dim)

    @inbounds for (j, bits) in enumerate(basis.bitsvec)
        newbits, element = act_seq(coeff, ops, bits)
        iszero(element) && continue

        i, factor = basis_element(basis, newbits, j)
        (i > dim) && continue

        opmat[i, j] += factor * element
    end
    return opmat
end


# Matrix-free action of operator(s) on a state vector
function _apply!(output::AbstractVector, ops::Vector{<:AbstractOp},
    basis::AbstractBasis, input::AbstractVector, coeff::Number)
    fill!(output, zero(eltype(output)))
    dim = basis.dim

    @inbounds for (j, bits) in enumerate(basis.bitsvec)
        newbits, element = act_seq(coeff, ops, bits)
        iszero(element) && continue

        i, factor = basis_element(basis, newbits, j)
        (i > dim) && continue

        output[i] += factor * element * input[j]
    end
end

# Apply operator(s) to a state and return new state
function apply(ops::Vector{<:AbstractOp}, psi::QState{T}, coeff::Number=1.0) where T <: Number
    output = zeros(T, psi.basis.dim)
    _apply!(output, ops, psi.basis, psi.vector, coeff)
    return QState{T}(psi.basis, output)
end
 
function apply(ops::Vector{<:AbstractOp}, basis::AbstractBasis, psi::AbstractVector, coeff::Number=1.0)
    psi_new = similar(psi)
    _apply!(psi_new, ops, basis, psi, coeff)
    return psi_new
end

# In-place apply operator(s) to a state
function apply!(ops::Vector{<:AbstractOp}, psi::QState, coeff::Number=1.0)
    input = copy(psi.vector)
    _apply!(psi.vector, ops, psi.basis, input, coeff)
end

function apply!(ops::Vector{<:AbstractOp}, basis::AbstractBasis, psi::AbstractVector, coeff::Number=1.0)
    input = copy(psi)
    _apply(psi, ops, basis, input, coeff)
end

# Compute expectation value of operator(s) in a state
function expected(ops::Vector{<:AbstractOp}, psi::QState, coeff::Number=1.0)
    opmat = matrixform(ops, psi.basis, coeff)
    v = psi.vector
    return real(dot(v, opmat, v))
end

# Compute ⟨x|O|y⟩ for two states and operator(s)
function LinearAlgebra.dot(x::QState, ops::Vector{<:AbstractOp}, y::QState)
    length(x.vector) == length(y.vector) || error("wrong dimension of two states!")
    opmat = matrixform(ops, y.basis)
    return dot(x.vector, opmat, y.vector)
end

"""
Construct the hamiltonian matrix from OpSum type with assigned basis.
Return either dense or sparse matrix controled by sparsed, default to be dense
because `eigen` in LinearAlgebra does not support sparse matrix.
"""
function makeHamiltonian(opsum::OpSum{T}, basis::AbstractBasis; 
    sparsed::Bool=false, dtype::Type{<:Number}=Float64) where {T <: Number}
    dim = basis.dim
    
    covec = opsum.covec
    opvec = opsum.opvec

    ELT = operator_eltype(basis, T, dtype)
    hmat = sparsed ? spzeros(ELT, dim, dim) : zeros(ELT, dim, dim) 
    @inbounds for (j, bits) in enumerate(basis.bitsvec)
        for s in eachindex(opsum.covec)
            newbits, element = act_seq(covec[s], opvec[s], bits)
            iszero(element) && continue

            i, factor = basis_element(basis, newbits, j)
            (i > dim) && continue

            hmat[i, j] += factor * element
        end
    end
    return hmat
end


# Matrix-free action of an operator sum on a state vector
function _apply!(output::AbstractVector, opsum::OpSum, basis::AbstractBasis,
    input::AbstractVector)
    fill!(output, zero(eltype(output)))
    dim = basis.dim
    covec = opsum.covec
    opvec = opsum.opvec

    @inbounds for (j, bits) in enumerate(basis.bitsvec)
        for s in eachindex(covec)
            newbits, element = act_seq(covec[s], opvec[s], bits)
            iszero(element) && continue

            i, factor = basis_element(basis, newbits, j)
            (i > dim) && continue

            output[i] += factor * element * input[j]
        end
    end
end

function apply(opsum::OpSum, psi::QState{T}) where {T <: Number}
    output = zeros(T, psi.basis.dim)
    _apply!(output, opsum, psi.basis, psi.vector)
    return QState{T}(psi.basis, output)
end

function apply(opsum::OpSum, basis::AbstractBasis, psi::AbstractVector)
    psi_new = similar(psi)
    _apply!(psi_new, opsum, basis, psi)
    return psi_new
end

function apply!(opsum::OpSum, psi::QState)
    input = copy(psi.vector)
    _apply!(psi.vector, opsum, psi.basis, input)
end

function apply!(opsum::OpSum, basis::AbstractBasis, psi::AbstractVector)
    input = copy(psi)
    _apply(psi, opsum, basis, input, coeff)
end

function expected(opsum::OpSum, psi::QState)
    hmat = makeHamiltonian(opsum, psi.basis; sparsed=true)
    return real(dot(psi.vector, hmat, psi.vector))
end

function LinearAlgebra.dot(x::QState, opsum::OpSum, y::QState)
    hmat = makeHamiltonian(opsum, y.basis; sparsed=true)
    return dot(x.vector, hmat, y.vector)
end
