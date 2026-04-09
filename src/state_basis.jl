# Abstract base types for basis and state representations
abstract type AbstractBasis end

# Dictionary mapping spin or occupation symbols to bit values
global bitDict = Dict{Symbol, Bool}(
    :Up => true, 
    :Dn => false,
    :Occ => true,
    :Emp => false
)

# Convert a vector of symbols (e.g. [:Up, :Dn]) to an integer bitstring
function symbol2bits(strvec::Vector{Symbol})::UInt32
    bits = 0x00000
    for s in strvec
        bits |= bitDict[s]
        bits <<= 0x01
    end
    return bits >> 0x01
end

#===============Basis constructors=================#
# Basis for states with fixed particle number (e.g. number of up spins)
struct NumBasis <: AbstractBasis
    num::Int                # number of particles (ones)
    bitsvec::Vector{UInt32}  # all basis states as bitstrings

    NumBasis(lsize::Int, num::Int) = new(num, numbitbasis(lsize, num))
end

# Basis for all possible states (no conservation)
struct FullBasis <: AbstractBasis
    lsize::Int                  # system size
    bitsvec::UnitRange{UInt32}   # all bitstrings from 0 to 2^lsize-1

    FullBasis(lsize::Int) = new(lsize, 0x00000 : (0x00001 << lsize - 0x00001))
end

SpinBasis(lsize::Int) = FullBasis(lsize)
SpinBasis(lsize::Int, num::Int) = NumBasis(lsize, num)

# Find index of a bitstring in NumBasis (returns 0 if not in basis)
function findindex(basis::NumBasis, bits::UInt32)::Int
    count_ones(bits) == basis.num || return 0
    return searchsortedfirst(basis.bitsvec, bits)
end

# Find index of a bitstring in FullBasis
findindex(basis::FullBasis, bits::UInt32)::Int = bits + 1


#================== Quantum State ===================#
mutable struct QState{T <: Number}
    basis::AbstractBasis
    vector::Vector{T}
end

function QState(lsize::Int, bits::UInt32; type::DataType=ComplexF64)
    basis = FullBasis(lsize)
    vector = zeros(type, length(basis.bitsvec))
    vector[bits + 1] = one(type)
    QState{type}(basis, vector)
end

function QState(lsize::Int, num::Int, bits::UInt32; type::DataType=ComplexF64)
    basis = NumBasis(lsize, num)
    vector = zeros(type, length(basis.bitsvec))
    idx = findindex(basis, bits)
    idx == 0 && error("bitstring not in basis!")
    vector[idx] = one(type)  # default to first basis state
    QState{type}(basis, vector)
end

QState(statestr::String; type::DataType=ComplexF64) = 
    QState(length(statestr), parse(UInt32, statestr; base=2); type=type)
QState(strvec::Vector{Symbol}; type::DataType=ComplexF64) = 
    QState(length(strvec), symbol2bits(strvec); type=type)

QState(statestr::String, num::Int; type::DataType=ComplexF64) = 
    QState(length(statestr), num, parse(UInt32, statestr; base=2); type=type)
QState(strvec::Vector{Symbol}, num::Int; type::DataType=ComplexF64) = 
    QState(length(strvec), num, symbol2bits(strvec); type=type)


function LinearAlgebra.normalize!(psi::QState)
    LinearAlgebra.normalize!(psi.vector)
end
LinearAlgebra.norm(psi::QState) = LinearAlgebra.norm(psi.vector)

function inner(x::QState, y::QState)
    length(x.vector) == length(y.vector) || error("wrong dimension of two states!")
    return x.vector' * y.vector
end

function Base.:+(x::QState, y::QState)
    length(x.vector) == length(y.vector) || error("wrong dimension of two states!")
    return QState(y.basis, x.vector + y.vector)
end
