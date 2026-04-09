using ExactDiagonalize
# For plotting
using CairoMakie

let 
    set_systype(:Spin)  # Set system type to spin
    L, N = 10, 1        # System size and particle number
    Δ = 1.0         # Interaction parameter

    # Initial state: single up spin at site 1
    init = QState("1000000000", N)
    
    # Build Hamiltonian terms for XY model
    opsum = OpSum(Float64)
    for j in 1:L
        nj = mod1(j + 1, L)
        opsum += (Δ, :Z, j, :Z, nj)
        opsum += (1.0, :X, j, :X, nj)
        opsum += (-1.0, :iY, j, :iY, nj)
    end

    obs_exa = ZObserver(L, init.basis)
    obs_ode = ZObserver(L, init.basis)
    obs_spm = ZObserver(L, init.basis)

    ts_exa = 0.0:0.02:10.0
    ts_ode = 0.0:0.05:10.0
    ts_spm = 0.0:0.05:10.0

    timeEvolve(opsum, init, ts_exa, obs_exa, exact())
    timeEvolve(opsum, init, ts_ode, obs_ode, rk4())
    timeEvolve(opsum, init, ts_spm, obs_spm, spmat())

    steps_ode = length(ts_ode)
    steps_spm = length(ts_spm)

    inter = 3
    ode_idx = 1 : inter : steps_ode
    spm_idx = 2 : inter : steps_spm

    set_theme!(Axis = (
        xtickalign = 1, 
        ytickalign = 1, 
        xlabelsize = 18, 
        ylabelsize = 18
    ))

    fig = Figure()
    ax = Axis(fig[1, 1], 
        xlabel = L"t",
        ylabel = L"\langle Z_1 \rangle(t)", 
        xticks = 0.0:2.0:10.0
    )

    cg = cgrad(:darkrainbow, 3, categorical=true)

    lines!(ax, ts_exa, obs_exa.data, label="ED", color = cg[1])
    scatter!(ax, ts_ode[ode_idx], obs_ode.data[ode_idx], label="ODE", 
        color=cg[2], markersize=12)
    scatter!(ax, ts_spm[spm_idx], obs_spm.data[spm_idx], label="SPM", 
        color=cg[3], marker=:x, markersize=12)
    axislegend(ax, L"Δ = %$Δ")
    fig
    # save("homeworks/hw1/hw1_2a.png", fig)

end

