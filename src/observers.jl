#=============Obsrever system to record quantities during evolution=============#
abstract type AbstractObserver end

mutable struct OperatorObserver{T <: Number} <: AbstractObserver
    opmat::SpMatrix{T}
    data::Vector{Float64}

    OperatorObserver(os::Tuple, basis::AbstractBasis; sparsed::Bool=true) = new{typeof(os[1])}(
        op2mat(os[1], os2ops(os, get_optype(_systype[])), basis; sparsed=sparsed), Vector{Float64}()
    )
end

function record!(obs::OperatorObserver, psi::Vector, step::Int)
    val = real(dot(psi, obs.opmat, psi))
    push!(obs.data, val)
end

mutable struct OpSumObserver{T <: Number} <: AbstractObserver
    opsmat::SpMatrix{T}
    data::Vector{Float64}

    OpSumObserver(ops::OpSum{T}, basis::AbstractBasis; sparsed::Bool=true) where T <: Number = new{T}(
        makeHamiltonian(ops, basis; sparsed=sparsed), Vector{Float64}()
    )
end

function record!(obs::OpSumObserver, psi::Vector, step::Int)
    val = real(dot(psi, obs.opsmat, psi))
    push!(obs.data, val)
end

mutable struct ZObserver <: AbstractObserver
    phases::Vector{Int}
    data::Vector{Float64}

    ZObserver(loc::Int, basis::SpinBasis) = new(
        2 .* readbit.(basis.bitsvec, loc % UInt8) .- 1, 
        Vector{Float64}()
    )
end

function record!(obs::ZObserver, psi::Vector, step::Int)
    phi = psi .* obs.phases
    val = real(dot(psi, phi))
    push!(obs.data, val)
end

mutable struct XObserver <: AbstractObserver
    idx0::Vector{Int}
    idx1::Vector{Int}
    data::Vector{Float64}

    function XObserver(loc::Int, basis::SpinBasis)
        mask = readbit.(basis.bitsvec, loc % UInt8)
        new(basis.bitsvec[.! mask] .+ 1, basis.bitsvec[mask] .+ 1, Vector{Float64}())
    end
end

function record!(obs::XObserver, psi::Vector, step::Int)
    phi = similar(psi)
    phi[obs.idx0], phi[obs.idx1] = psi[obs.idx1], psi[obs.idx0]
    val = real(dot(psi, phi))
    push!(obs.data, val)
end