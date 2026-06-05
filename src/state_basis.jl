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

# ============================================================================
# BASIS DEFINITIONS AND CONSTRUCTORS
# ============================================================================
# Consstruct basis for 1D spin system
struct SpinBasis{N, K, V <: AbstractVector{UInt32}} <: AbstractBasis
    lsize::Int
    dim::Int
    num::N # total spin up numbers
    kint::K # momentum sector label, i.e. `m` in  k = 2πm/L
    bitsvec::V
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
# Construct the Quantum state with basis and coefficient vector
struct QState{T <: Number, B <: AbstractBasis}
    basis::B
    vector::Vector{T}
end

statevec(basis::AbstractBasis, bits::UInt32) = statevec(ComplexF64, basis, bits)
statevec(basis::AbstractBasis, lsize::Int, func::Function) = statevec(ComplexF64, basis, lsize, func)
statevec(basis::AbstractBasis, symvec::Vector{Symbol}) = statevec(ComplexF64, basis, symvec)
statevec(basis::AbstractBasis, statestr::String) = statevec(ComplexF64, basis, statestr)

function statevec(ELT::Type{<:Number}, basis::AbstractBasis, bits::UInt32)
    vector = zeros(ELT, basis.dim)
    idx = first(findindex(basis, bits))
    idx > basis.dim && error("bitstring not in basis!")
    vector[idx] = one(ELT)
    return vector
end

function statevec(ELT::Type{<:Number}, basis::AbstractBasis, lsize::Int, func::Function)
    lsize <= 32 || error("System size too large for UInt32 representation")
    bits = 0x00000
    for j in 1 : lsize
        bits |= bitDict[func(j)]  # Set bit for current symbol
        bits <<= 0x01        # Shift left for next symbol
    end
    bits >>= 0x01
    return statevec(ELT, basis, bits)
end

function statevec(ELT::Type{<:Number}, basis::AbstractBasis, symvec::Vector{Symbol})
    length(symvec) <= 32 || error("System size too large for UInt32 representation")
    bits = 0x00000
    for s in symvec
        bits |= bitDict[s]  # Set bit for current symbol
        bits <<= 0x01        # Shift left for next symbol
    end
    bits >>= 0x01
    return statevec(ELT, basis, bits)
end

statevec(ELT::Type{<:Number}, basis::AbstractBasis, statestr::String) = 
    statevec(ELT, basis, parse(UInt32, statestr; base=2))


QState(basis::AbstractBasis, bits::UInt32) = QState(ComplexF64, basis, bits)
QState(basis::AbstractBasis, statestr::String) = QState(ComplexF64, basis, statestr)
QState(basis::AbstractBasis, symvec::Vector{Symbol}) = QState(ComplexF64, basis, symvec)
QState(basis::AbstractBasis, lsize::Int, func::Function) = QState(ComplexF64, basis, lsize, func)

function QState(ELT::Type{<:Number}, basis::B, bits::UInt32) where B <: AbstractBasis
    vector = statevec(ELT, basis, bits)
    QState{ELT, B}(basis, vector)
end

function QState(ELT::Type{<:Number}, basis::B, statestr::String) where B <: AbstractBasis
    vector = statevec(ELT, basis, statestr)
    QState{ELT, B}(basis, vector)
end

function QState(ELT::Type{<:Number}, basis::B, symvec::Vector{Symbol}) where B <: AbstractBasis
    vector = statevec(ELT, basis, symvec)
    QState{ELT, B}(basis, vector)
end

function QState(ELT::Type{<:Number}, basis::B, lsize::Int, func::Function) where B <: AbstractBasis
    vector = statevec(ELT, basis, lsize, func)
    QState{ELT, B}(basis, vector)
end


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

function matrixize(psi::QState, b::Int)
    basis = psi.basis
    shift = basis.lsize - b

    left_parts = basis.bitsvec .>> shift
    right_parts = basis.bitsvec .& ((0x00001 << shift) - 0x00001)

    lbits = unique(left_parts)
    rbits = unique(right_parts)
    M, N = length(lbits), length(rbits)

    ldict = Dict(v => i for (i, v) in enumerate(lbits))
    rdict = Dict(v => i for (i, v) in enumerate(rbits))

    mat = zeros(eltype(psi.vector), M, N)
    @inbounds for i in 1 : basis.dim
        lidx = ldict[left_parts[i]]
        ridx = rdict[right_parts[i]]
        mat[lidx, ridx] = psi.vector[i]
    end
    return mat
end

function ent_entropy(psi::QState, b::Int=psi.basis.lsize ÷ 2)
    (b <= 0 || b >= psi.basis.lsize) && return 0.0
    mat = matrixize(psi, b)
    Σ = svdvals!(mat)
    SvN = 0.0

    @inbounds for s in Σ
        p = s*s
        if p > 1e-300
            SvN -= p * log(p)
        end
    end
    return SvN
end
