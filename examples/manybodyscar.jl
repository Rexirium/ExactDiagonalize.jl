using Revise
using LinearAlgebra
using ExactDiagonalize
# For plotting
using CairoMakie

let
    set_systype(:Spin)
    Ls = 14
    g = 0.0

    basis = SpinBasis(Ls)

    initvec = statevec(basis, Ls, n -> isodd(n) ? :Up : :Dn)

    opsum = OpSum(Float64)
    for j in 1 : Ls
        jprev = mod1(j - 1, Ls)
        jpost = mod1(j + 1, Ls)
        opsum += 1.0, :Pdn, jprev, :X, j, :Pdn, jpost
        opsum += g * (-1.0)^j, :Z, j
    end

    eigenergies, eigstates = spectrum(opsum, basis; retvecs=true)

    overlaps = abs2.(transpose(eigstates) * initvec)
    mask = overlaps .> 0
    
    fig = Figure()
    ax = Axis(fig[1, 1], title="Overlaps", yscale=log10, 
        xlabel=L"E_n", ylabel=L"|\langle \mathbb{Z}_2 | n \rangle |^2", 
        limits=(nothing, (1e-10, 1e1)))

    scatter!(ax, eigenergies[mask], overlaps[mask])
    fig
    
end