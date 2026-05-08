using CairoMakie
using Makie.GeometryBasics
using HDF5

function bandplot!(subfig, kx::Matrix, ky::Matrix, lowerband::Matrix, upperband::Matrix)
    ax1 = Axis3(subfig, title="Band Structure in BZ", 
        xlabel=L"k_x", ylabel=L"k_y", zlabel=L"E", aspect=(1, √3/2, 2/3),
        xticks=(-4π/3 : 2π/3 : 2π/3, [L"-4π/3", L"-2π/3", L"0"]),
        yticks=(0 : π/2 : π, [L"0", L"π/2", L"π"]),
        azimuth=-0.3π, elevation=0.125π)
    
    surface!(ax1, kx, ky, lowerband, alpha=0.5, shininess = 50)
    surface!(ax1, kx, ky, upperband, alpha=0.5, shininess = 50)
end

function curvatureplot!(subfig, kx::Matrix, ky::Matrix, curvature::Matrix, chernnum::Real)
    nx, ny = size(curvature)

    polys = vec([
        Polygon(Point2f[
            (kx[i, j],   ky[i, j]),
            (kx[i+1, j], ky[i+1, j]),
            (kx[i+1, j+1], ky[i+1, j+1]),
            (kx[i, j+1], ky[i, j+1])
        ])
        for i in 1:nx, j in 1:ny
    ])
    
    ax1 = Axis(subfig[1, 1], title="Berry Curvature with Chern number $(round(chernnum; digits=3))", 
        xlabel=L"k_x", ylabel=L"k_y", aspect=DataAspect(), 
        xticks=(-4π/3 : 2π/3 : 2π/3, [L"-4π/3", L"-2π/3", L"0"]),
        yticks=(0 : π/2 : π, [L"0", L"π/2", L"π"]),)
    pp = poly!(ax1, polys, color=vec(curvature),
        colormap=:balance, strokewidth=0, stroke_depth_shift=0)
    Colorbar(subfig[1, 2], pp)
end

function eigenstateplot!(subfig, kx::Matrix, ky::Matrix, lowervecs::Array{ComplexF64})
    nx, ny = size(kx)
    us = Matrix{Float64}(undef, nx, ny)
    vs = Matrix{Float64}(undef, nx, ny)
    ls = Matrix{Float64}(undef, nx, ny)

    for j in 1:ny
        for i in 1:nx
            c = lowervecs[2, i, j] * conj(lowervecs[1, i, j]) / abs(lowervecs[1, i, j])
            us[i, j] = real(c)
            vs[i, j] = imag(c)
            ls[i, j] = abs(c)
        end
    end

    ax1 = Axis(subfig[1, 1], title="Eigenstate Amplitude in BZ", 
        xlabel=L"k_x", ylabel=L"k_y", 
        xticks=(-4π/3 : 2π/3 : 2π/3, [L"-4π/3", L"-2π/3", L"0"]),
        yticks=(0 : π/2 : π, [L"0", L"π/2", L"π"]),
        aspect=DataAspect())

    ap = arrows2d!(ax1, vec(kx), vec(ky), vec(us), vec(vs), color=vec(ls), align=:center, 
        lengthscale=0.4, colormap=:plasma, tipwidth=12, tiplength=6)
    Colorbar(subfig[1, 2], ap)
end

let 
    m2 =0.0
    file = h5open("solidstatehomework/hw2/haldane_data_1.h5", "r")
    kx_grid = read(file, "kx_grid")
    ky_grid = read(file, "ky_grid")
    lowerband = read(file, "lowerband")
    upperband = read(file, "upperband")
    lowervecs = read(file, "lowervecs")
    curvature = read(file, "curvature")
    chernnum = read(file, "chernnum")
    close(file)

    kx_list = kx_grid[1:11:end, 1:11:end]
    ky_list = ky_grid[1:11:end, 1:11:end]
    lowervecs_list = lowervecs[:, 1:11:end, 1:11:end]

    set_theme!(Axis=(
        xtickalign = 1,
        ytickalign = 1,
        xticklabelsize = 16,
        yticklabelsize = 16,
        xlabelsize = 18,
        ylabelsize = 18,
    ))

    fig0 = Figure()
    bandplot!(fig0[1,1], kx_grid, ky_grid, lowerband, upperband)
    save("solidstatehomework/hw2/band_1.png", fig0)

    fig = Figure(size=(600, 750))
    
    ga = fig[1,1] = GridLayout()
    gb = fig[2,1] = GridLayout()
    curvatureplot!(ga, kx_grid, ky_grid, curvature, chernnum)
    eigenstateplot!(gb, kx_list, ky_list, lowervecs_list)
    save("solidstatehomework/hw2/curvature_1.png", fig)
end