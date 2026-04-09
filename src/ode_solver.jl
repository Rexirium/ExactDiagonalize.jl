"""
Runge-Kutta 4th order (RK4) time evolution solver
"""
# Tag for RK4 method
rk4() = Val(:rk4)

# Evolve state using RK4 method for time steps ts
function timeEvolve(ops::OpSum, init::QState, ts::AbstractRange, obs::AbstractObserver, ::Val{:rk4})
    ihmat = -im * makeHamiltonian(ops, init.basis; sparsed=true)
    psi = ComplexF64.(init.vector)
    dim = length(psi)
    
    k1 = Vector{ComplexF64}(undef, dim)
    k2 = Vector{ComplexF64}(undef, dim)
    k3 = Vector{ComplexF64}(undef, dim)
    k4 = Vector{ComplexF64}(undef, dim)
    tmp = Vector{ComplexF64}(undef, dim)

    h = step(ts)
    h_2 = h / 2

    for i in 1:length(ts)
        record!(obs, psi, i)  # record state at current time
        if i == length(ts)
            break
        end

        # RK4 steps
        mul!(k1, ihmat, psi)
        tmp = psi + h_2 * k1
        mul!(k2, ihmat, tmp)
        tmp = psi + h_2 * k2
        mul!(k3, ihmat, tmp)
        tmp = psi + h * k3
        mul!(k4, ihmat, tmp)

        psi += (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    end
    return QState(init.basis, psi)
end
