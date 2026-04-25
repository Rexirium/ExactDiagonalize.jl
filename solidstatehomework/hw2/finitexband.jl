using CairoMakie

include("haldane.jl")

function plot_Haldane_spectrum(Lx::Int, nky::Int, t1::Real, t2::Number; m2::Real=0.0, start::Char='B')
    kys = start == 'B' ? range(0, 2π, nky) : range(-π, π, nky)
    kyc = start == 'B' ? π : 0.0
    spectra = Matrix{Float64}(undef, Lx, nky)

    H = zeros(Lx, Lx)
    for (i, ky) in enumerate(kys)
        updateHaldaneHamiltonian!(H, Lx, ky, t1, t2; m2 = m2, start=start)
        spectra[:, i] = eigvals(H)
    end

    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], title="Haldane Spectrum", 
        xlabel=L"k_y a_y", ylabel=L"E", 
        xticks= start == 'B' ? 
            (0 : π/2 : 2π, [L"0", L"π/2", L"π", L"3π/2", L"2π"]) : 
            (-π : π/2 : π, [L"-π", L"-π/2", L"0", L"π/2", L"π"])
    )

    for j in 1:Lx
        lines!(ax, kys, spectra[j, :], color=:black, linewidth=1)
    end
    vlines!(ax, [kyc - π/2], color=:red, linestyle=:dash)
    vlines!(ax, [kyc + π/2], color=:red, linestyle=:dash)
    fig
end

function plot_Haldane_edgestates(Lx::Int, kys::Vector{Float64}, t1::Real, t2::Number; m2=0.0, start::Char='B') 
    fig = Figure()

    for (row, kyc) in enumerate(kys)
        H = makeHaldaneHamiltonian(Lx, kyc * π, t1, t2; m2=m2, start=start)
        edgestates = get_edgestates(H; zeromode=false)

        ax = Axis(fig[row, 1], 
            title="Haldane Edge States at k_y a_y = $(kyc)π", 
            xlabel=L"j", ylabel=L"|ψ_j|^2", 
            xticks = 0 : 10 : Lx
        )
        deg = size(edgestates, 2)
        for i in 1:deg
            barplot!(ax, 1 : Lx, abs2.(edgestates[:, i]), label="Edge State $i")
        end
        axislegend(ax; position=:ct)
    end
    fig   
end

let 
    set_theme!(Axis=(
        xtickalign = 1,
        ytickalign = 1,
        xticklabelsize = 16,
        yticklabelsize = 16,
        xlabelsize = 18,
        ylabelsize = 18,
    ))

    fig = plot_Haldane_edgestates(50, [0.5], 1.0, 0.2im; m2=0, start='B')
    #save("solidstatehomework/hw2/edgestate.png", fig)
end