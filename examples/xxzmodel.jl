using Revise
using ExactDiagonalize
# For plotting
using CairoMakie

let 
    set_systype(:Spin)  # Set system type to spin
    L, N = 10, 1    # System size and particle number
    Δ = 1.0            # Interaction parameter

    # Initial state: Neel state
    basis = SpinBasis(L; num = N)
    init = ProductState(basis, "1000000000")
    
    # Build Hamiltonian terms for XXZ model
    opsum = OpSum(Float64)
    for j in 1:L
        nj = mod1(j + 1, L)
        opsum += Δ, :Z, j, :Z, nj
        opsum += 1.0, :X, j, :X, nj
        opsum += -1.0, :iY, j, :iY, nj
    end

    # Observable: Z at last site
    obs = ZObserver(L, basis)
    
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
