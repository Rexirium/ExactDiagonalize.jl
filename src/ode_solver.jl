rk4() = Val(:rk4)

function timeEvolve(ops::OpSum, init::AbstractState, ts::AbstractVector, obs::AbstractObserver, ::Val{:rk4})
    ihmat = -im * makeHamiltonian(ops, init.basis; sparsed=true)
    psi = ComplexF64.(init.vector)
    dim = length(psi)
    
    k1 = Vector{ComplexF64}(undef, dim)
    k2 = Vector{ComplexF64}(undef, dim)
    k3 = Vector{ComplexF64}(undef, dim)
    k4 = Vector{ComplexF64}(undef, dim)
    tmp = Vector{ComplexF64}(undef, dim)

    for (i, t) in enumerate(ts)
        record!(obs, psi)
        if i == length(ts)
            break
        end
        h = ts[i+1] - t
        h_2 = h / 2

        mul!(k1, ihmat, psi)
        tmp = psi + h_2 * k1
        mul!(k2, ihmat, tmp)
        tmp = psi + h_2 * k2
        mul!(k3, ihmat, tmp)
        tmp = psi + h * k3
        mul!(k4, ihmat, tmp)

        psi += (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    end
    return State(init.basis, psi)

end
