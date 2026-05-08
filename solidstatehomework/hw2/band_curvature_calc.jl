using HDF5
include("haldane.jl")

let 
    t1 = 1.0
    t2 = 0.0
    m2 = 3√3 * 0.2 * t1
    num = 120
    kxs = range(-4π/3, 0, num + 1)
    kys = range(0, 4π/3, num + 1)
    dkx, dky = step(kxs), step(kys)
    kx_grid = kxs .+ 1/2 * kys'
    ky_grid = zeros(num+1) .+ √3/2 * kys'

    lowerband, upperband, lowervecs = eigenHaldane(kx_grid, ky_grid, t1, t2; m2=m2)
    curvature = computeBerryCurvature(lowervecs, dkx * dky)
    chernnum = sum(curvature) * dkx * dky / (2π)
    println("Chern number: ", chernnum)

    h5open("solidstatehomework/hw2/haldane_data_2.h5", "w") do file
        write(file, "kx_grid", kx_grid)
        write(file, "ky_grid", ky_grid)
        write(file, "lowerband", lowerband)
        write(file, "upperband", upperband)
        write(file, "lowervecs", lowervecs)
        write(file, "curvature", curvature)
        write(file, "chernnum", chernnum)
    end
end