using Revise
using ExactDiagonalize
using CairoMakie

let 
    set_systype(:Spin)
    L = 12
    B = 0.5

    init = QState(fill(:Dn, L))

    opsum = OpSum(Float64)
    for j in 1:L-1
        opsum += -1.0, :Z, j, :Z, j + 1
        opsum += -B, :X, j
    end
    opsum += -B, :X, L

    obs = ZObserver(L ÷ 2, init.basis)

    ts = 0.0:0.02:10.0
    @time timeEvolve(opsum, init, ts, obs, exact())

    # Plot observable vs time
    fig = Figure()
    ax = Axis(fig[1,1],

    )
    lines!(ax, ts, obs.data)
    fig
end