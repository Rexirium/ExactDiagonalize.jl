using MKL
using LinearAlgebra
using CairoMakie

makeTBHamiltonian(onsite::Vector, hopping::Vector{T}) where T <: Number = Tridiagonal{T}(conj(hopping), onsite, hopping)

function makeTBHamiltonian(v::Number, w::Number, lsize::Int)
    hopping = repeat([v, w], lsize ÷ 2)
    if iseven(lsize)
        deleteat!(hopping, lsize)
    end
    makeTBHamiltonian(zeros(lsize), hopping)
end

function updateTBHamiltonian!(H::Tridiagonal{S}, onsite::Vector{T}, hopping::Vector{T}) where {S <: Number, T <: S}
    H.du .= hopping
    H.d .= onsite
    H.dl .= conj(hopping)
end

function updateTBHamiltonian!(H::Tridiagonal{S}, v::T, w::T) where {S <: Number, T <: S}
    upinds = diagind(H, 1)
    loinds = diagind(H, -1)
    H[upinds[1:2:end]] .= v
    H[loinds[1:2:end]] .= v'
    H[upinds[2:2:end]] .= w
    H[loinds[2:2:end]] .= w'
end

let 
    H = makeTBHamiltonian(1.0 + 0im, 0.5+ 0im, 10)
    updateTBHamiltonian!(H, 0.5 + 0.1im, 0.8- 0.2im)
    H
end