
# Abstract base types for basis and state representations
abstract type AbstractBasis end
abstract type AbstractState end

# Dictionary mapping spin or occupation symbols to bit values
global bitDict = Dict{Symbol, Bool}(
    :Up => true, 
    :Dn => false,
    :Occ => true,
    :Emp => false
)

# Convert a vector of symbols (e.g. [:Up, :Dn]) to an integer bitstring
function translate(strvec::Vector{Symbol})::Int
    bits = 0
    for s in strvec
        bits |= bitDict[s]
        bits <<= 1
    end
    return bits >> 1
end
#===============Basis constructors=================#
# Basis for states with fixed particle number (e.g. number of up spins)
struct NumBasis <: AbstractBasis
    num::Int                # number of particles (ones)
    bitsvec::Vector{<:Int}  # all basis states as bitstrings

    NumBasis(lsize::Int, num::Int) = new(num, numbitbasis(lsize, num))
end

# Basis for all possible states (no conservation)
struct FullBasis <: AbstractBasis
    lsize::Int                  # system size
    bitsvec::UnitRange{<:Int}   # all bitstrings from 0 to 2^lsize-1

    FullBasis(lsize::Int) = new(lsize, 0 : (1 << lsize - 1))
end

# Find index of a bitstring in NumBasis (returns 0 if not in basis)
function findindex(basis::NumBasis, bits::Int)
    count_ones(bits) == basis.num || return 0
    return searchsortedfirst(basis.bitsvec, bits)
end

# Find index of a bitstring in FullBasis
findindex(basis::FullBasis, bits::Int) = bits + 1


#===========Particle number conserved state  ===========#

# State vector for a system with fixed particle number
mutable struct NumState{T <: Number} <: AbstractState
    basis::NumBasis      # basis object
    vector::Vector{T}   # state vector (amplitudes)
end


# Create a NumState with a single basis state set to 1
function NumState(lsize::Int, bits::Int; type::DataType=ComplexF64)
    basis = NumBasis(lsize, count_ones(bits))
    vector = zeros(type, length(basis.bitsvec))
    idx = searchsortedfirst(basis.bitsvec, bits)
    vector[idx] = one(type)
    NumState{type}(basis, vector)
end


# Create NumState from a binary string (e.g. "1010")
NumState(statestr::String; type::DataType=ComplexF64) = 
    NumState(length(statestr), parse(Int, statestr; base=2); type=type)


# Create NumState from a vector of symbols (e.g. [:Up, :Dn])
NumState(strvec::Vector{Symbol}; type::DataType=ComplexF64) = 
    NumState(length(strvec), translate(strvec); type=type)

#===============Generic and state================#
mutable struct FullState{T <: Number} <: AbstractState
    basis::FullBasis
    vector::Vector{T}
end

function FullState(lsize::Int, bits::Int; type::DataType=ComplexF64)
    basis = FullBasis(lsize)
    vector = zeros(type, 1 << lsize)
    vector[bits + 1] = one(type)
    FullState{type}(basis, vector)
end

FullState(statestr::String; type::DataType=ComplexF64) = 
    FullState(length(statestr), parse(Int, statestr; base=2); type=type)

FullState(strvec::Vector{:Symbol}; type::DataType=ComplexF64) = 
    FullState(length(strvec), translate(strvec); type=type)

State(basis::NumBasis, vector::Vector{T}) where T <: Number = NumState{T}(basis, vector)
State(basis::FullBasis, vector::Vector{T}) where T <: Number = FullState{T}(basis, vector)

#============Convertion between two basis=============#
function FullState(state::NumState{T}, lsize::Int) where T <: Number
    lsize < state.basis.num && error("system size too small!")
    basis = FullBasis(lsize)
    vector = zeros(T, 1 << lsize)
    inds = state.basis.bitsvec
    vector[inds] .= state.vector

    return FullState{T}(basis, vector)
end

function NumState(state::FullState{T}, num::Int) where T <: Number
    num > state.basis.lsize && error("too many particles")
    basis = NumBasis(state.basis.lsize, num)
    inds = basis.bitsvec
    vector = state.vector[inds]

    return NumState{T}(basis, vector)
end

function LinearAlgebra.normalize!(psi::AbstractState)
    LinearAlgebra.normalize!(psi.vector)
end

function inner(x::S, y::S) where S <: AbstractState
    length(x.vector) == length(y.vector) || error("wrong dimension of two states!")
    return x.vector' * y.vector
end

function Base.:+(x::S, y::S) where S <: AbstractState
    length(x.vector) == length(y.vector) || error("wrong dimension of two states!")
    return State(y.basis, x.vector + y.vector)
end
