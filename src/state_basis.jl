# ============================================================================
# STATE AND BASIS REPRESENTATIONS FOR QUANTUM MANY-BODY SYSTEMS
# ============================================================================
# This module defines the core data structures for representing quantum states
# and bases in exact diagonalization calculations. It supports both full and
# number-conserving (fixed particle number) bases for spin or fermionic systems.

# Abstract base types for basis and state representations
abstract type AbstractBasis end

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

# Convert a vector of symbols (e.g. [:Up, :Dn]) to an integer bitstring
function symbol2bits(strvec::Vector{Symbol})::UInt32
    length(strvec) <= 32 || error("System size too large for UInt32 representation")
    bits = 0x00000
    for s in strvec
        bits |= bitDict[s]  # Set bit for current symbol
        bits <<= 0x01        # Shift left for next symbol
    end
    return bits >> 0x01      # Remove final unnecessary shift
end

# ============================================================================
# BASIS DEFINITIONS AND CONSTRUCTORS
# ============================================================================
# Consstruct basis for 1D spin system
struct SpinBasis{N, K, V <: AbstractVector{UInt32}} <: AbstractBasis
    lsize::Int
    num::N # total spin up numbers
    kint::K # momentum sector label, i.e. `m` in  k = 2πm/L
    bitsvec::V
end

function SpinBasis(lsize::Int; num = nothing, kint = nothing)
    if isnothing(num) && isnothing(kint)
        # full basis use UnitRange to save memory
        bitsvec = 0x00000 : ((0x00001 << lsize) - 0x00001)
    elseif !isnothing(num) && isnothing(kint)
        # Generate bits vector with fixed `1` s
        bitsvec = numbitbasis(lsize, num)
    else
        error("Invalid basis type, wait for later development.")
    end
    
    return SpinBasis(lsize, num, kint, bitsvec)
end

Base.:(==)(b1::SpinBasis, b2::SpinBasis) = 
    (b1.lsize == b2.lsize) && (b1.num == b2.num) && (b1.kint == b2.kint)
# find index of a product state in general basis
function findindex(basis::SpinBasis{Nothing, Nothing}, bits::UInt32)::Int
    return Int(bits) + 1
end
# find index of a product state in number conserving basis
function findindex(basis::SpinBasis{Int, Nothing}, bits::UInt32)::Int
    count_ones(bits) == basis.num || return length(basis.bitsvec) + 1  # if number of ones doesn't match, return out of bounds index
    return searchsortedfirst(basis.bitsvec, bits)
end
# find index of a product state in other basis
function findindex(basis::SpinBasis{Int, Int}, bits::UInt32)::Int
    return searchsortedfirst(basis.bitsvec, bits)
end

# ============================================================================
# QUANTUM STATE CONSTRUCTORS
# ============================================================================
# Construct the Quantum state with basis and coefficient vector
struct QState{T <: Number, B <: AbstractBasis}
    basis::B
    vector::Vector{T}
end

function QState(lsize::Int, bits::UInt32; num = nothing, kint = nothing, type::DataType = ComplexF64)
    basis = SpinBasis(lsize; num = num, kint = kint)
    vector = zeros(type, length(basis.bitsvec))
    idx = findindex(basis, bits) # find the index of the assigned state
    idx > length(basis.bitsvec) && error("bitstring not in basis!")
    vector[idx] = one(type) # nonzero coefficient only for the assigned state
    QState{type, SpinBasis}(basis, vector)
end

QState(statestr::String; num = nothing, kint = nothing, type::DataType = ComplexF64) = 
    QState(length(statestr), parse(UInt32, statestr; base=2); num = num, kint = kint, type = type)

QState(strvec::Vector{Symbol}; num = nothing, kint = nothing, type::DataType = ComplexF64) = 
    QState(length(strvec), symbol2bits(strvec); num = num, kint = kint, type = type)


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
