# ============================================================================
# STATE AND BASIS REPRESENTATIONS FOR QUANTUM MANY-BODY SYSTEMS
# ============================================================================
# This module defines the core data structures for representing quantum states
# and bases in exact diagonalization calculations. It supports both full and
# number-conserving (fixed particle number) bases for spin or fermionic systems.

# Abstract base types for basis and state representations
const NInt = Union{Int, Nothing}
abstract type AbstractBasis{N <: NInt, K <: NInt} end

# ============================================================================
# SYMBOL-TO-BIT MAPPING
# ============================================================================
# Dictionary mapping spin or occupation symbols to bit values used internally
# for efficient bitstring representation of quantum states.
# Symbols: :Up/Dn (spins), :Occ/Emp (occupation)
const bitDict = Dict{Symbol, Bool}(
    :Up => true,  # Spin up or occupied state
    :Dn => false,  # Spin down or empty state
    :Occ => true,  # Occupied state (fermionic)
    :Emp => false  # Empty state (fermionic)
)

# ============================================================================
# BASIS DEFINITIONS AND CONSTRUCTORS
# ============================================================================
# Consstruct basis for 1D spin system
struct SpinBasis{N <: NInt, K <: NInt} <: AbstractBasis{N, K}
    lsize::Int
    dim::Int
    num::N # total spin up numbers
    kint::K # momentum sector label, i.e. `m` in  k = 2πm/L
    bitsvec::AbstractVector{UInt32}
    orbsize::Vector{UInt32} 
end

function SpinBasis(lsize::Int; num = nothing, kint = nothing)
    if isnothing(num) && isnothing(kint)
        # full basis use UnitRange to save memory
        bitsvec = 0x00000 : ((0x00001 << lsize) - 0x00001)
        return SpinBasis(lsize, 1 << lsize, num, kint, bitsvec, UInt32[])
    elseif !isnothing(num) && isnothing(kint)
        # Generate bits vector with fixed `1` s
        bitsvec = numbitbasis(lsize, num)
        return SpinBasis(lsize, length(bitsvec), num, kint, bitsvec, UInt32[])
    elseif isnothing(num) && !isnothing(kint)
        bitsvec, orbsize = momentbitbasis(lsize, kint)
        return SpinBasis(lsize, length(bitsvec), num, kint, bitsvec, orbsize)
    else
        error("Invalid basis type, wait for later development.")
    end
end

Base.:(==)(b1::SpinBasis, b2::SpinBasis) = 
    (b1.lsize == b2.lsize) && (b1.num == b2.num) && (b1.kint == b2.kint)

# find index of a product state in general basis
function findindex(basis::SpinBasis{Nothing, Nothing}, bits::UInt32)::Int
    return Int(bits) + 1
end

# find index of a product state in number conserving basis
function findindex(basis::SpinBasis{Int, Nothing}, bits::UInt32)::Int
    count_ones(bits) == basis.num || return basis.dim + 1  # if number of ones doesn't match, return out of bounds index
    return searchsortedfirst(basis.bitsvec, bits)
end

function findindex(basis::SpinBasis{Nothing, Int}, bits::UInt32)
    # For momentum sector basis, we need to check both the bitstring and its momentum label
    lsize = basis.lsize
    resbits = bits
    tmpbits = bits
    bs64 = (UInt64(bits) << lsize) | bits
    mask = typemax(UInt32) >> (32 - lsize)

    dist = 0
    for i in 1 : lsize - 1
        tmpbits = ((bs64 >> (lsize - i)) % UInt32) & mask

        is_less = tmpbits < resbits
        resbits = ifelse(is_less, tmpbits, resbits)
        dist = ifelse(is_less, i, dist)
    end
    return searchsortedfirst(basis.bitsvec, resbits), dist
end

# find index of a product state in other basis
function findindex(basis::SpinBasis{Int, Int}, bits::UInt32)::Int
    return searchsortedfirst(basis.bitsvec, bits)
end

function Base.print(basis::SpinBasis; bitstyle::String="bin")
    lsize = basis.lsize
    println("Basis size: $(basis.dim)")
    println("Index\tState\tInteger")
    if bitstyle == "bin"
        for (i, bits) in enumerate(basis.bitsvec)
            bstr = bitstring(bits)[33 - lsize : end]
            bstr = "| " * join(bstr, ' ') * " ⟩"
            println("$(i).\t$bstr\t$bits")
        end
    elseif bitstyle == "sym" || bitstyle == "arrow"
        for (i, bits) in enumerate(basis.bitsvec)
            bstr = bitstring(bits)[33 - lsize : end]
            sstr = replace(bstr, '1' => "↑", '0' => "↓")
            sstr = '|' * join(sstr, ' ') * " ⟩"
            println("$(i).\t$sstr\t$bits")
        end
    end
end

# ============================================================================
# QUANTUM STATE CONSTRUCTORS
# ============================================================================

# Constructors for product states and random states in a given basis
product_state(basis::AbstractBasis, bits::UInt32) = product_state(ComplexF64, basis, bits)
product_state(basis::AbstractBasis, func::Function) = product_state(ComplexF64, basis, func)
product_state(basis::AbstractBasis, symvec::Vector{Symbol}) = product_state(ComplexF64, basis, symvec)
product_state(basis::AbstractBasis, statestr::String) = product_state(ComplexF64, basis, statestr)
random_state(basis::AbstractBasis) = random_state(ComplexF64, basis)

function product_state(ELT::Type{<:Number}, basis::AbstractBasis, bits::UInt32)
    vector = zeros(ELT, basis.dim)
    idx = first(findindex(basis, bits))
    idx > basis.dim && error("bitstring not in basis!")
    vector[idx] = one(ELT)
    return vector
end

function product_state(ELT::Type{<:Number}, basis::AbstractBasis, func::Function)
    bits = 0x00000
    for j in 1 : basis.lsize
        bits |= bitDict[func(j)]  # Set bit for current symbol
        bits <<= 0x01        # Shift left for next symbol
    end
    bits >>= 0x01
    return product_state(ELT, basis, bits)
end

function product_state(ELT::Type{<:Number}, basis::AbstractBasis, symvec::Vector{Symbol})
    length(symvec) == basis.lsize || error("The length of symbol vector incompactible with the basis!")

    bits = 0x00000
    for s in symvec
        bits |= bitDict[s]  # Set bit for current symbol
        bits <<= 0x01        # Shift left for next symbol
    end
    bits >>= 0x01
    return product_state(ELT, basis, bits)
end

function product_state(ELT::Type{<:Number}, basis::AbstractBasis, statestr::String)
    length(statestr) == basis.lsize || error("The length of state string incompactible with the basis!")
    return product_state(ELT, basis, parse(UInt32, statestr; base=2))
end

function random_state(ELT::Type{<:Number}, basis::AbstractBasis)
    vector = randn(ELT, basis.dim)
    LinearAlgebra.normalize!(vector)
    return vector
end

# Construct the Quantum state with basis and coefficient vector
struct QState{T <: Number}
    basis::AbstractBasis
    vector::Vector{T}
end

# Convenience constructors for QState
ProductState(ELT::Type{<:Number}, basis::AbstractBasis, args...) = 
    QState(basis, product_state(ELT, basis, args...))
ProductState(basis::AbstractBasis, args...) = 
    QState(basis, product_state(ComplexF64, basis, args...))

RandomState(ELT::Type{<:Number}, basis::AbstractBasis) = QState(basis, random_state(ELT, basis))
RandomState(basis::AbstractBasis) = QState(basis, random_state(ComplexF64, basis))


# ============================================================================
# Some basic linear algebra operations for quantum states
# ============================================================================
function LinearAlgebra.normalize!(psi::QState)
    LinearAlgebra.normalize!(psi.vector)
end
LinearAlgebra.norm(psi::QState) = LinearAlgebra.norm(psi.vector)

function LinearAlgebra.dot(x::QState, y::QState)
    x.basis == y.basis || error("Basis mismatch! Cannot perform inner product on different bases.")
    return dot(x.vector, y.vector)
end

function Base.:+(x::QState, y::QState)
     x.basis == y.basis || error("Basis mismatch! Cannot add different bases.")
    return QState(y.basis, x.vector + y.vector)
end

