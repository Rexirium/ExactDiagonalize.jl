using Revise
using LinearAlgebra
using ExactDiagonalize
# For plotting
using CairoMakie

let
    set_systype(:Spin)
    Ls = 12
    g = -0.1

    basis = SpinBasis(Ls)

    initvec = product_state(basis, n -> isodd(n) ? :Up : :Dn)

    opsum = OpSum(Float64)
    for j in 1 : Ls
        jpprev = mod1(j - 2, Ls)
        jprev = mod1(j - 1, Ls)
        jpost = mod1(j + 1, Ls)
        jppost = mod1(j + 2, Ls)
        opsum += 1.0, :Pdn, jprev, :X, j, :Pdn, jpost
        # opsum += g, :Pdn, jprev, :X, j, :Pdn, jpost, :Z, jppost
        # opsum += g, :Z, jpprev, :Pdn, jprev, :X, j, :Pdn, jpost
    end

    eigenergies, eigstates = spectrum(opsum, basis; retvecs=true)

    entropies = zeros(basis.dim)

    for n in 1 : basis.dim
        entropy= ent_entropy(basis, eigstates[:, n], Ls ÷ 2)
        entropies[n] = entropy
        
    end

    
    overlaps = abs2.(transpose(eigstates) * initvec)
    marksizes = [overlaps[n] > 1e-2 ? 20 : 5 for n in 1 : basis.dim]

    fig = Figure()
    ax = Axis(fig[1, 1], title="Entropy vs energy", yscale=log10, 
        xlabel=L"E_n", ylabel=L"S(L/2)",  
        limits=(nothing, (2e-1, 3)))

    scatter!(ax, eigenergies, entropies, color = overlaps, colormap=:viridis, markersize=marksizes)
    fig
    
end