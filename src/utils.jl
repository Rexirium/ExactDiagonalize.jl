"""
Utility functions for bit manipulation and basis generation.

This module provides helper functions for reading, flipping, and splitting bits, as well as generating integer bases with a fixed number of set bits.
"""
function readbit(bits::UInt32, pos::Unsigned)::Bool
    # Return the value of the bit at position `pos` (1-based) in the integer `bits`.
    return (bits >> (pos - 0x01)) & true
end

function readbit(bits::UInt32, pos::Tuple{<:Unsigned, <:Unsigned})::Tuple{Bool, Bool}
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
function signbetween(bits::UInt32, i::Unsigned, j::Unsigned)
    mask = 0x00001 << (j - i - 0x01) - 0x00001
    segbits = (bits >> i) & mask
    return (-1) ^ count_ones(segbits)
end

function flip(bits::UInt32, pos::Unsigned)::UInt32
    # Flip (toggle) the bit at position `pos` in `bits`.
    return bits ⊻ (0x00001 << (pos - 0x01))
end

function flip(bits::UInt32, pos::Unsigned, b::Bool)::UInt32
    # Flip (toggle) the bit at position `pos` in `bits` if `b` is true; otherwise, leave unchanged.
    return bits ⊻ (b << (pos - 0x01))
end

function cshift(bits::UInt32, len::Int, shift::Int)::UInt32
    # Circularly shift the bits in `bits` to the left by `shift` positions, within a total length of `len` bits.
    shift_mod = mod(shift, len)
    mask = (0x00001 << len) - 0x00001
    return ((bits << shift_mod) | (bits >> (len - shift_mod))) & mask
end

function cshift(bits::UInt32, len::Int)::UInt32
    mask = (0x00001 << len) - 0x00001
    return ((bits << 1) | (bits >> (len - 1))) & mask
end

function splitbasis(bits::UInt32, b::Unsigned)::Tuple{UInt32, UInt32}
    # Split the integer `bits` into two parts at bit position `b`.
    # Returns a tuple (right, left), where `right` contains the lower `b` bits and `left` contains the remaining higher bits.
    b >= 0x00000 || return 0x00000, bits
    left = bits >> b
    right = bits & ((0x00001 << b) - 0x00001)
    return right, left
end

function numbitbasis(len::Int, num::Int)::Vector{UInt32}
    """
    numbitbasis(len::Int, num::Int)
    Generate all `len`-bit integers with exactly `num` bits set to 1.
    """
    num > len && error("more ones than total bits")
    num == 0 && return UInt32[0x00000]
    basis = UInt32[]
    sizehint!(basis, binomial(len, num))

    len_u = len % UInt8
    num_u = num % UInt8
    maxind = (0x00001 << len_u) - 0x00001
    ind = (0x00001 << num_u) - 0x00001
    while ind <= maxind
        push!(basis, ind)
        u = ind & (-ind)
        v = ind + u
        next = v + ((v ⊻ ind) >> 0x02) ÷ u 
        next > maxind && break
        ind = next
    end
    return basis
end

function momentbitbasis(len::Int, kint::Int, a::Int=1)
    u = zeros(Bool, len + 1)
    basis = UInt32[]
    orbit_sizes = UInt32[]

    function fkm_necklace(t, p)
        if t > len
            if len % p == 0
                if (kint * p) % len == 0
                    val = 0x00000
                    for i in 1:len
                        val = (val << 0x01) | u[i + 1]
                    end
                    push!(basis, val)
                    push!(orbit_sizes, p % UInt32)
                end
            end
        else
            u[t + 1] = u[t - p + 1]
            fkm_necklace(t + 1, p)
            
            if u[t - p + 1] == false
                u[t + 1] = true
                fkm_necklace(t + 1, t)
            end
        end
    end
    fkm_necklace(1, 1)
    return basis, orbit_sizes
end