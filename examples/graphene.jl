using MKL
using LinearAlgebra
using CairoMakie

makeTBHamiltonian(onsite::Vector, hopping::Vector{T}) where T <: Number = Tridiagonal{T}(conj(hopping), onsite, hopping)

function makeSSHHamiltonian(v::Number, w::Number, lsize::Int; eletype::DataType = Base.promote_typeof(v, w))
    hopping = repeat(eletype.([v, w]), lsize ÷ 2)
    if iseven(lsize)
        deleteat!(hopping, lsize)
    end
    Tridiagonal{eletype}(conj(hopping), zeros(eletype, lsize), hopping)
end

function updateTBHamiltonian!(H::Tridiagonal{S}, onsite::Vector{T}, hopping::Vector{T}) where {S <: Number, T <: S}
    H.du .= hopping
    H.d .= onsite
    H.dl .= conj(hopping)
end

function updateSSHHamiltonian!(H::Tridiagonal{S}, v::T, w::T) where {S <: Number, T <: S}
    upinds = diagind(H, 1)
    loinds = diagind(H, -1)
    H[upinds[1:2:end]] .= v
    H[loinds[1:2:end]] .= v'
    H[upinds[2:2:end]] .= w
    H[loinds[2:2:end]] .= w'
end

function graphene_site(t::Real, k::Real, sitetype::String)
    if sitetype == "A"
        v = t * exp(im * k) + t
        w = complex(t)
    elseif sitetype == "B"
        v = complex(t)
        w = t * exp(im * k) + t
    else
        error("Invalid sitetype: $sitetype. Must be 'A' or 'B'.")
    end
    return v, w
end

function makeGrapheneHamiltonian(t::Real, k::Real, lsize::Int; sitetype::String="A")
    v, w = graphene_site(t, k, sitetype)
    makeSSHHamiltonian(v, w, lsize)
end

function updateGrapheneHamiltonian!(H::Tridiagonal, t::Real, k::Real; sitetype::String="A") 
    v, w = graphene_site(t, k, sitetype)
    updateSSHHamiltonian!(H, v, w)
end

function solvedegenerate(degstates::Matrix)
    X = range(1, size(degstates, 1))
    mat = degstates' * (X .* degstates)
    _, evecs = eigen(Hermitian(mat))
    return degstates * evecs
end

function get_edgestates(H::Tridiagonal{S}) where S <: Number
    evals, evecs = eigen(Hermitian(H))
    deginds = findall(x -> isapprox(x, 0.0, atol=1e-5), evals)
    if !isempty(deginds)
        println("Degenerate states found, solving...")
        degstates = evecs[:, deginds]
        return solvedegenerate(degstates)
    else
        println("No degenerate states found.")
        return zeros(S, size(H, 1), 2)
    end 
end
#=
let 
    Ly = 50
    kxs = [1.0, 1.1, 1.2]

    set_theme!(Axis=(
        xtickalign = 1,
        ytickalign = 1,
        xlabelsize = 18,
        ylabelsize = 18,
    ))
    fig = Figure(size=(800, 900))

    for (row, kxc) in enumerate(kxs)
        H = makeGrapheneHamiltonian(1.0, kxc * π, Ly)
        edgestates = get_edgestates(H)
        ax = Axis(fig[row, 1], 
            title="Graphene Edge States at kₓaₓ = $(kxc)π", 
            xlabel=L"j", ylabel=L"|ψ_j|^2", 
            xticks = 0:10:Ly
        )
        deg = size(edgestates, 2)
        for i in 1:deg
            barplot!(ax, 1:Ly, abs2.(edgestates[:, i]), label="Edge State $i")
        end
        axislegend(ax; position=:ct)
    end
    fig
    
end
=#
let 
    Ly = 49
    nkx = 500
    kxs = range(0, 2π, nkx)
    spectra = Matrix{Float64}(undef, Ly, nkx)
    H = makeGrapheneHamiltonian(1.0, 0.0, Ly; sitetype="B")
    for (i, kx) in enumerate(kxs)
        updateGrapheneHamiltonian!(H, 1.0, kx; sitetype="B")
        spectra[:, i] = eigvals(H)
    end

    set_theme!(Axis=(
        xtickalign = 1,
        ytickalign = 1,
        xlabelsize = 18,
        ylabelsize = 18,
    ))
    fig = Figure()
    ax = Axis(fig[1, 1], title="Graphene Spectrum", 
        xlabel=L"k_x a_x", ylabel=L"E", 
        xticks=(0 : π/2 : 2π, [L"0", L"π/2", L"π", L"3π/2", L"2π"])
    )

    for j in 1:Ly
        lines!(ax, kxs, spectra[j, :], color=:black)
    end
    vlines!(ax, [2π / 3], color=:red, linestyle=:dash)
    vlines!(ax, [4π / 3], color=:red, linestyle=:dash)
    fig

end