include("src/operators.jl")
include("src/exactdiagonalize.jl")
include("src/ode_solver.jl")
include("src/sparsemat.jl")

using CairoMakie

let 
    set_systype(:Spin)
    L, N = 10, 1
    Δ = 0.5

    init = NumState("1000000000")
    
    os = Tuple[]
    for j in 1:L
        nj = mod1(j + 1, L)
        push!(os, (Δ, :Z, j, :Z, nj))
        push!(os, (1.0, :X, j, :X, nj))
        push!(os, (-1.0, :iY, j, :iY, nj))
    end
    # push!(os, (1.0, :X, L))
    opsum = OpSum(os, Float64)

    os2 = [(1.0, :Z, j) for j in 1:L]
    opsum2 = OpSum(os2, Float64)

    obs = OperatorObserver((1.0, :Z, L), init.basis)
    
    #@show makeHamiltonian(ops2, init.basis)
    
    ts = 0.0:0.05:10.0
    @time timeEvolve_exact(opsum, init, ts, obs)

    fig = Figure()
    ax = Axis(fig[1,1],

    )
    lines!(ax, ts, obs.data)
    fig
    
end
