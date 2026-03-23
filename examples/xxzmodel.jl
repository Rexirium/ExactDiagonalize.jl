using Revise
using ExactDiagonalize
# For plotting
using CairoMakie

let 
    set_systype(:Spin)  # Set system type to spin
    L, N = 12, 6     # System size and particle number
    Δ = 1.0            # Interaction parameter

    # Initial state: Neel state
    init = NumState([isodd(j) ? :Up : :Dn for j in 1:L])
    
    # Build Hamiltonian terms for XXZ model
    opsum = OpSum(Float64)
    for j in 1:L
        nj = mod1(j + 1, L)
        opsum += Δ, :Z, j, :Z, nj
        opsum += 1.0, :X, j, :X, nj
        opsum += -1.0, :iY, j, :iY, nj
    end

    # opsum2: sum of Z operators (not used here)
    os2 = [(1.0, :Z, j) for j in 1:L]
    opsum2 = OpSum(os2, Float64)

    # Observable: Z at last site
    obs = OperatorObserver((1.0, :Z, L), init.basis)
    
    # Time points for evolution
    ts = 0.0:0.05:10.0
    @time timeEvolve(opsum, init, ts, obs, exact())

    # Plot observable vs time
    fig = Figure()
    ax = Axis(fig[1,1],

    )
    lines!(ax, ts, obs.data)
    fig
end
