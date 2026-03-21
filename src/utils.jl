"""
Utility functions for bit manipulation and basis generation.

This module provides helper functions for reading, flipping, and splitting bits, as well as generating integer bases with a fixed number of set bits.
"""
function readbit(bits::Int, pos::Unsigned)::Bool
    # Return the value of the bit at position `pos` (1-based) in the integer `bits`.
    return (bits >> (pos - 0x01)) & true
end

function readbit(bits::Int, pos::Tuple{<:Unsigned, <:Unsigned})::Tuple{Bool, Bool}
    # Return the values of the bits at positions `pos[1]` and `pos[2]` in `bits` as a tuple.
    shifted = bits >> (pos[1] - 0x01)
    b1 = shifted & true
    shifted >>= (pos[2] - pos[1])
    b2 = shifted & true
    return b1, b2
end

"""
    signbetween(bits::Int, i::Unsigned, j::Unsigned)

Count the number of ones between bit positions `i` and `j` (excluding `i` and `j`, with `i < j`),
and return the sign (-1 or 1) according to whether the count is odd or even.
"""
function signbetween(bits::Int, i::Unsigned, j::Unsigned)
    mask = 1 << (j - i - 0x01) - 1
    segbits = (bits >> i) & mask
    return (-1)^count_ones(segbits)
end

function flip(bits::Int, pos::Unsigned)::Int
    # Flip (toggle) the bit at position `pos` in `bits`.
    return bits ⊻ (1 << (pos - 0x01))
end

function flip(bits::Int, pos::Unsigned, b::Bool)::Int
    # Flip (toggle) the bit at position `pos` in `bits` if `b` is true; otherwise, leave unchanged.
    return bits ⊻ (b << (pos - 0x01))
end

function splitbasis(bits::Int, b::Unsigned)::Tuple{Int, Int}
    # Split the integer `bits` into two parts at bit position `b`.
    # Returns a tuple (right, left), where `right` contains the lower `b` bits and `left` contains the remaining higher bits.
    b >= 0x00 || return 0, bits
    left = bits >> b
    right = bits & ((1 << b) - 1)
    return right, left
end

function numbitbasis(len::Int, num::Int)
    """
    numbitbasis(len::Int, num::Int)
    Generate all `len`-bit integers with exactly `num` bits set to 1.
    """
    num > len && error("more ones than total bits")
    num == 0 && return Int[0]
    basis = Int[]
    sizehint!(basis, binomial(len, num))
    maxind = (1 << len) - 1
    ind = (1 << num) - 1
    while ind <= maxind
        push!(basis, ind)
        u = ind & (-ind)
        v = ind + u
        next = v + ((v ⊻ ind) ÷ u) >> 2
        next > maxind && break
        ind = next
    end
    return basis
end