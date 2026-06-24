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

function cshift(bits::UInt32, len::Int, shift::Int=1)::UInt32
    # Circularly shift the bits in `bits` to the right by `shift` positions, within a total length of `len` bits.
    shift = mod(shift, len)
    mask = typemax(UInt32) >> (32 - len)
    return ((bits >> shift) | (bits << (len - shift))) & mask
end

function find_repr(bits::UInt32, len::Int, a::Int)
    resbits = bits
    tmpbits = bits

    bs64 = (UInt64(bits) << len) | bits
    mask = typemax(UInt32) >> (32 - len)

    dist = 0
    for d in a : a : len - a
        tmpbits = UInt32(bs64 >> d) & mask

        if tmpbits < resbits
            resbits = tmpbits
            dist = d
        end
    end
    return resbits, dist
end

function splitbasis(bits::UInt32, shift::Int)::Tuple{UInt32, UInt32}
    # Split the integer `bits` into two parts at bit position `b`.
    # Returns a tuple (right, left), where `right` contains the lower `b` bits and `left` contains the remaining higher bits.
    shift >= 0x00000 || return 0x00000, bits
    left = bits >> shift
    right = bits & ((0x00001 << shift) - 0x00001)
    return right, left
end

function splitbasis(bitsvec::Vector{UInt32}, shift::Int)
    mask = (0x00001 << shift) - 0x00001
    lefts = bitsvec .>> shift
    rights = bitsvec .& mask
    return lefts, rights
end

function numbitbasis(len::Int, num::Int)::Vector{UInt32}
    """
    numbitbasis(len::Int, num::Int)
    Generate all `len`-bit integers with exactly `num` bits set to 1 with Gosper algorithm
    """
    num > len && error("more ones than total bits!")
    num == 0 && return UInt32[0x00000]

    basis = Vector{UInt32}(undef, binomial(len, num))
    idx = 0

    maxind = (0x00001 << len) - 0x00001
    ind = (0x00001 << num) - 0x00001

    while ind <= maxind
        idx += 1
        basis[idx] = ind

        #Gosper generating engine
        u = ind & (-ind)
        v = ind + u
        next_ind = v + ((v ⊻ ind) >> 0x02) ÷ u 
        next_ind > maxind && break
        ind = next_ind
    end
    return basis
end

function numbitbasis(len::Int, nums::Vector{Int})::Vector{UInt32}
    """
    numbitbasis(len::Int, nums::Vector{Int})
    Generate all `len`-bit integers with number of `1`s in a vector of integers `nums` with Gosper algorithm
    """
    total_dim = sum(n -> binomial(len, n), nums; init=0)

    total_basis = UInt32[]
    sizehint!(total_basis, total_dim)

    # Gosper engine for every particle number `num`
    for num in nums
        basis = numbitbasis(len, num)
        append!(total_basis, basis)
    end

    sort!(total_basis) # globally ordered to use biparte search
    return total_basis
end

function momentbitbasis(len::Int, kint::Int, a::Int=1)
    """
    momentbitbasis(len::Int, kint::Int, a::Int=1)
    Generate all `len`-bit integers that is distinct under translation by lattice spacing `a`
    """
    n = len ÷ a
    k = 0x001 << a
    u = zeros(UInt16, len + 1)

    basis = UInt32[]
    orbit_sizes = UInt32[]

    function fkm_necklace(t, p)
        if t > n
            if n % p == 0
                if (kint * p * a) % len == 0
                    val = 0x00000
                    for i in 1:n
                        val = (val << a) | u[i + 1]
                    end
                    push!(basis, val)
                    push!(orbit_sizes, p % UInt32)
                end
            end
        
        else
            u[t + 1] = u[t - p + 1]
            fkm_necklace(t + 1, p)
            
            for j in (u[t - p + 1] + 0x001) : (k - 0x001)
                u[t + 1] = j
                fkm_necklace(t + 1, t)
            end
        end
    end

    fkm_necklace(1, 1)
    return basis, orbit_sizes
end

function num_moment_bitbasis(len::Int, num::Int, kint::Int, a::Int=1)
    basis = UInt32[]
    orbsizes = UInt32[]

    if num == 0
        if (kint * 1) % len == 0
            push!(basis, 0x00000)
            push!(orbsizes, 0x00001)
        end
        return basis, orbsizes
    end

    n = len ÷ a
    mask = typemax(UInt32) >> (32 - len)

    maxind = (0x00001 << len) - 0x00001
    ind = (0x00001 << num) - 0x00001

    while ind <= maxind
        # ===================================================
        # First part：combining smallest element for orbits check in momentbitbasis
        # ===================================================
        cur = ind
        is_rep = true
        p = n
        
        for step in 1:(n - 1)
            # cyclic right shift by `a`
            cur = ((cur >> a) | (cur << (len - a))) & mask
            
            if cur < ind
                # smaller integer than `ind` indicating `ind` is not representative element
                is_rep = false
                break
            elseif cur == ind
                # return to itself for the first time, record the period
                p = step
                break
            end
        end
        # if ind is representative, check the momentum selection rule.
        if is_rep
            # CAUTION: rotating `p` blocks is translating `p*a` sites
            if (kint * p * a) % len == 0
                push!(basis, ind)
                push!(orbsizes, p % UInt32)
            end
        end
        
        # ===================================================
        # Second part: Gosper generating engine
        # ===================================================
        u = ind & (-ind)
        v = ind + u
        next_ind = v + ((v ⊻ ind) >> 0x02) ÷ u 
        next_ind > maxind && break
        ind = next_ind
    end
    
    return basis, orbsizes
end

function num_moment_bitbasis(len::Int, nums::Vector{Int}, kint::Int, a::Int)
    total_basis = UInt32[]
    total_orbsizes = UInt32[]

    for num in nums
        basis, orbsizes = num_moment_bitbasis(len, num, kint, a)
        append!(total_basis, basis)
        append!(total_orbsizes, orbsizes)
    end

    idx = sortperm(total_basis)
    return total_basis[idx], total_orbsizes[idx]
end