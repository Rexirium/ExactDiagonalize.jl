function readbit(bits::Int, pos::Int)::Bool
    return (bits >> (pos - 1)) & true == true
end

function readbit(bits::Int, pos::Tuple{Int, Int})::Tuple{Bool, Bool}
    shifted = bits >> (pos[1] - 1)
    b1 = shifted & true == true
    shifted >>= (pos[2] - pos[1])
    b2 = shifted & true == true
    return b1, b2
end
# count number of ones between bit i and j (excluding i and j), i<j 
# and return the sign according to the odd or even of the number 
function signbetween(bits::Int, i::Int, j::Int)
    mask = 1<<(j-i-1) - 1
    segbits = (bits>>i) & mask
    return (-1)^count_ones(segbits)
end

function flip(bits::Int, pos::Int)::Int
    return bits ⊻ (1 << (pos - 1))
end

function flip(bits::Int, pos::Int, b::Bool)::Int
    return bits ⊻ (b << (pos - 1))
end

function splitbasis(bits::Int, b::Int)
    b >=0 || return 0, bits
    left = bits >> b
    right = bits & ((1<<b) - 1)
    return right, left
end

function numbitbasis(len::Int, num::Int)
    """
    generating all the len-bit integers with num bits read 1.
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