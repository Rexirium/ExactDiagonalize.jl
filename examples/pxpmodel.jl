using Revise
using LinearAlgebra
using ExactDiagonalize
# For plotting
using CairoMakie

let
    set_systype(:Spin)
    Ls = 12
    g = - 0.1

    basis = SpinBasis(Ls)

    initvec = statevec(basis, Ls, n -> isodd(n) ? :Up : :Dn)

    opsum = OpSum(Float64)
    for j in 1 : Ls
        jprev = mod1(j - 1, Ls)
        jpost = mod1(j + 1, Ls)
        opsum += 1.0, :Pdn, jprev, :X, j, :Pdn, jpost
        opsum += g, :Z, j
        opsum += g, :X, j, :X, jpost
        opsum += -g, :iY, j, :iY, jpost
    end

    eigenergies, eigstates = spectrum(opsum, basis; retvecs=true)

    entropies = zeros(basis.dim)
    for n in 1 : basis.dim
        entropies[n] = ent_entropy(basis, eigstates[:, n], Ls ÷ 2)
    end

    overlaps = abs2.(transpose(eigstates) * initvec)
    # mask = entropies .> 0
    
    fig = Figure()
    ax = Axis(fig[1, 1], title="Entropy vs energy", yscale=log10, 
        xlabel=L"E_n", ylabel=L"S(L/2)",  
        limits=(nothing, (1e-2, 1e1)))

    scatter!(ax, eigenergies, entropies)
    fig
    
end