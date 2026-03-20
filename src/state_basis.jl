abstract type AbstractBasis end
abstract type AbstractState end

global bitDict = Dict{Symbol, Bool}(
    :Up => true, 
    :Dn => false,
    :Occ => true,
    :Emp => false
)

function translate(strvec::Vector{Symbol})::Int
    bits = 0
    for s in strvec
        bits |= bitDict[s]
        bits <<= 1
    end
    return bits
end
#===============Basis constructors=================#
struct NumBasis <: AbstractBasis
    num::Int
    bitsvec::Vector{<:Int}

    NumBasis(lsize::Int, num::Int) = new(num, numbitbasis(lsize, num))
end

struct TotalBasis <: AbstractBasis
    lsize::Int
    bitsvec::UnitRange{<:Int}

    TotalBasis(lsize::Int) = new(lsize, 0 : (1 << lsize - 1))
end

function findindex(basis::NumBasis, bits::Int)
    count_ones(bits) == basis.num || return 0
    return searchsortedfirst(basis.bitsvec, bits)
end

findindex(basis::TotalBasis, bits::Int) = bits + 1


#===========Particle number conserved state  ===========#
mutable struct NumState{T <: Number} <: AbstractState
    basis::NumBasis
    vector::Vector{T}
end

function NumState(lsize::Int, bits::Int; type::DataType=ComplexF64)
    basis = NumBasis(lsize, count_ones(bits))
    vector = zeros(type, length(basis.bitsvec))
    idx = searchsortedfirst(basis.bitsvec, bits)
    vector[idx] = one(type)
    NumState{type}(basis, vector)
end

NumState(statestr::String; type::DataType=ComplexF64) = 
    NumState(length(statestr), parse(Int, statestr; base=2); type=type)

NumState(strvec::Vector{Symbol}; type::DataType=ComplexF64) = 
    NumState(length(strvec), translate(strvec); type=type)

#===============Generic and state================#
mutable struct TotalState{T <: Number} <: AbstractState
    basis::TotalBasis
    vector::Vector{T}
end

function TotalState(lsize::Int, bits::Int; type::DataType=ComplexF64)
    basis = TotalBasis(lsize)
    vector = zeros(type, 1 << lsize)
    vector[bits + 1] = one(type)
    TotalState{type}(basis, vector)
end

TotalState(statestr::String; type::DataType=ComplexF64) = 
    TotalState(length(statestr), parse(Int, statestr; base=2); type=type)

TotalState(strvec::Vector{:Symbol}; type::DataType=ComplexF64) = 
    TotalState(length(strvec), translate(strvec); type=type)

State(basis::NumBasis, vector::Vector{T}) where T <: Number = NumState{T}(basis, vector)
State(basis::TotalBasis, vector::Vector{T}) where T <: Number = TotalState{T}(basis, vector)

#============Convertion between two basis=============#
function TotalState(state::NumState{T}, lsize::Int) where T <: Number
    lsize < state.basis.num && error("system size too small!")
    basis = TotalBasis(lsize)
    vector = zeros(T, 1 << lsize)
    inds = state.basis.bitsvec
    vector[inds] .= state.vector

    return TotalState{T}(basis, vector)
end

function NumState(state::TotalState{T}, num::Int) where T <: Number
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
