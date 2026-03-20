
function updating!(psi::Vector{T}, hmat::SpMatrix, dt::Real, order::Int) where T <: Number
    fac = one(T)
    tmp = copy(psi)
    Hpsi = Vector{T}(undef, length(psi))
    for k in 1:order
        fac *= -im * dt / k
        mul!(Hpsi, hmat, tmp)
        psi .+= fac * Hpsi
        tmp .= Hpsi
    end
end

function timeEvolve_spmat(ops::OpSum, init::AbstractState, ts::AbstractVector, obs::AbstractObserver; order::Int=4)
    hmat = makeHamiltonian(ops, init.basis; sparsed=true)
    psi = ComplexF64.(init.vector)

    for (i, t) in enumerate(ts)
        record!(obs, psi)
        if i == length(ts)
            break
        end
        dt = ts[i + 1] - t
        updating!(psi, hmat, dt, order)
    end
    return State(init.basis, psi)
end