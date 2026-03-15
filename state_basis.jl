abstract type AbstractBasis end
abstract type AbstractState end

global bitDict = Dict{Symbol, Bool}(
    :Up => true, 
    :Dn => false,
    :Occ => true,
    :Emp => false
)

struct NumBasis <: AbstractBasis
    num::Int
    bitsvec::Vector{<:Int}

    NumBasis(lsize::Int, num::Int) = new(num, numbitbasis(lsize, num))
end

function translate(strvec::Vector{Symbol})::Int
    bits = 0
    for s in strvec
        bits |= bitDict[s]
        bits <<= 1
    end
    return bits
end

mutable struct NumState{T <: Number} <: AbstractState
    basis::NumBasis
    statevec::Vector{T}
end

function NumState(lsize::Int, bits::Int; type::DataType=ComplexF64)
    basis = NumBasis(lsize, count_ones(bits))
    statevec = zeros(type, length(basis.bitsvec))
    idx = searchsortedfirst(basis.bitsvec, bits)
    statevec[idx] = one(type)
    NumState{type}(basis, statevec)
end

NumState(statestr::String; type::DataType=ComplexF64) = 
    NumState(length(statestr), parse(Int, statestr; base=2); type=type)

NumState(strvec::Vector{Symbol}; type::DataType=ComplexF64) = 
    NumState(length(strvec), translate(strvec); type=type)

struct TotalBasis <: AbstractBasis
    bitsvec::UnitRange{<:Int}

    TotalBasis(lsize::Int) = new(0 : (1 << lsize - 1))
end

mutable struct TotalState{T <: Number} <: AbstractState
    basis::TotalBasis
    statevec::Vector{T}
end

function TotalState(lsize::Int, bits::Int; type::DataType=ComplexF64)
    basis = TotalBasis(lsize)
    statevec = zeros(type, length(basis.bitsvec))
    statevec[bits + 1] = one(type)
    NumState{type}(basis, statevec)
end

TotalState(statestr::String; type::DataType=ComplexF64) = 
    TotalState(length(statestr), parse(Int, statestr; base=2); type=type)

TotalState(strvec::Vector{:Symbol}; type::DataType=ComplexF64) = 
    TotalState(length(strvec), translate(strvec); type=type)

State(basis::NumBasis, statevec::AbstractVector) = NumState(basis, statevec)
State(basis::TotalBasis, statevec::AbstractVector) = TotalState(basis, statevec)

function findindex(basis::NumBasis, bits::Int)
    count_ones(bits) == basis.num || return -1
    return searchsortedfirst(basis.bitsvec, bits)
end

findindex(basis::TotalBasis, bits::Int) = bits + 1