using MKL, LinearAlgebra

const Am = [ 1.0 0.0 ;
    -1/2 √3/2 ;
    -1/2 -√3/2]
const Bm = [0.0 √3 ;
    -3/2 -√3/2 ;
    3/2 -√3/2]

function makeHaldaneHamiltonian(kv::Vector{Float64}, t1::Real, t2::Number; m2::Real=0.0)
    hmat = Matrix{ComplexF64}(undef, 2, 2)
    hmat[1, 1] = 2 * real(t2) * sum(cos.(Bm * kv)) + m2
    hmat[2, 2] = - 2 * imag(t2) * sum(sin.(Bm * kv)) - m2
    hmat[2, 1] = t1 * sum(cis.( Am * kv))
    hmat[1, 2] = conj(hmat[2, 1])
    return hmat
end

function makeHaldaneHamiltonian(Lx::Int, ky::Float64, t1::Real, t2::Number; m2::Real=0.0, eletype::DataType=Float64, start::Char='B')
    t2a, phi = abs(t2), angle(t2)

    if start == 'B'
        dd = repeat(eletype[2t2a * cos(ky - phi) - m2, 2t2a * cos(ky + phi) + m2], Lx ÷ 2)
        d1 = repeat(eletype[2t1 * cos(ky / 2), t1], Lx ÷ 2)
        d2 = repeat(eletype[2t2a * cos(ky/2 + phi), 2t2a * cos(ky/2 - phi)], Lx ÷ 2 - 1)
    else
        dd = repeat(eletype[2t2a * cos(ky + phi) + m2, 2t2a * cos(ky - phi) - m2], Lx ÷ 2)
        d1 = repeat(eletype[t1, 2t1 * cos(ky / 2)], Lx ÷ 2)
        d2 = repeat(eletype[2t2a * cos(ky/2 - phi), 2t2a * cos(ky/2 + phi)], Lx ÷ 2 - 1)
    end

    if iseven(Lx)
        pop!(d1)
    else
        push!(dd, 2t2a * cos(ky - phi) + m2)
        push!(d2, 2t2a * cos(ky/2 + phi))
    end

    return diagm(
        -2 => conj(d2), -1 => conj(d1), 
        0 => dd, 1 => d1, 2 => d2
    )
end

function updateHaldaneHamiltonian!(hmat::Matrix{ComplexF64}, kv::Vector{Float64}, t1::Real, t2::Number; m2::Real=0.0)
    hmat[1, 1] = 2 * real(t2) * sum(cos.(Bm * kv)) + m2
    hmat[2, 2] = - 2 * imag(t2) * sum(sin.(Bm * kv)) - m2
    hmat[2, 1] = t1 * sum(cis.( Am * kv))
    hmat[1, 2] = conj(hmat[2, 1])
end

function updateHaldaneHamiltonian!(hmat::Matrix{T}, Lx::Int, ky::Float64, t1::Real, t2::Number; m2::Real=0.0, start::Char='B') where T <: Number
    t2a, phi = abs(t2), angle(t2)

    if start == 'B'
        dd_o, dd_e = convert(T, 2t2a * cos(ky - phi) - m2), convert(T, 2t2a * cos(ky + phi) + m2)
        d1_o, d1_e = convert(T, 2t1 * cos(ky / 2)), convert(T, t1)
        d2_o, d2_e = convert(T, 2t2a * cos(ky/2 + phi)), convert(T, 2t2a * cos(ky/2 - phi))
    else
        dd_o, dd_e = convert(T, 2t2a * cos(ky + phi) + m2), convert(T, 2t2a * cos(ky - phi) - m2)
        d1_o, d1_e = convert(T, t1), convert(T, 2t1 * cos(ky / 2))
        d2_o, d2_e = convert(T, 2t2a * cos(ky/2 - phi)), convert(T, 2t2a * cos(ky/2 + phi))
    end

    @inbounds for j in 1 : Lx
        is_odd = isodd(j)

        hmat[j, j] = ifelse(is_odd, dd_o, dd_e)

        if j <= Lx - 1
            v1 = ifelse(is_odd, d1_o, d1_e)
            hmat[j, j+1] = v1
            hmat[j+1, j] = conj(v1)
        end

        if j <= Lx - 2
            v2 = ifelse(is_odd, d2_o, d2_e)
            hmat[j, j+2] = v2
            hmat[j+2, j] = conj(v2)
        end
    end
end

function eigenHaldane(kx::Matrix, ky::Matrix, t1::Real, t2::Number; m2::Real=0.0)
    hmat = Matrix{ComplexF64}(undef, 2, 2)
    lowerband = similar(kx)
    upperband = similar(kx)
    lowervecs = Array{ComplexF64}(undef, 2, size(kx)...)

    for (idx, kv) in enumerate(zip(kx, ky))
        updateHaldaneHamiltonian!(hmat, collect(kv), t1, t2; m2=m2)
        eigs, eigvs = eigen(hmat)
        lowerband[idx] = real(eigs[1])
        upperband[idx] = real(eigs[2])
        lowervecs[2idx-1 : 2idx] = eigvs[:, 1]
    end
    return lowerband, upperband, lowervecs
end

function computeBerryCurvature(lowervecs::Array{ComplexF64}, area::Real)
    nx, ny = size(lowervecs, 2) - 1, size(lowervecs, 3) - 1
    curvature = Matrix{Float64}(undef, nx, ny)
    U = Matrix{ComplexF64}(undef, 2, 2)
    for j in 1:ny
        for i in 1:nx
            ψ1 = lowervecs[:, i, j]
            ψ2 = lowervecs[:, i+1, j]
            ψ3 = lowervecs[:, i+1, j+1]
            ψ4 = lowervecs[:, i, j+1]

            U[1, 1] = dot(ψ1, ψ2) / abs(dot(ψ1, ψ2))
            U[2, 1] = dot(ψ2, ψ3) / abs(dot(ψ2, ψ3))
            U[2, 2] = dot(ψ3, ψ4) / abs(dot(ψ3, ψ4))
            U[1, 2] = dot(ψ4, ψ1) / abs(dot(ψ4, ψ1))

            curvature[i, j] = angle(prod(U)) / area
        end
    end
    return curvature
end

function solvedegenerate(degstates::Matrix)
    X = range(1, size(degstates, 1))
    mat = degstates' * (X .* degstates)
    _, evecs = eigen(Hermitian(mat))
    return degstates * evecs
end

function get_edgestates(H::Matrix{S}; zeromode::Bool=true) where S <: Number
    evals, evecs = eigen(Hermitian(H))
    if zeromode
        deginds = findall(x -> isapprox(x, 0.0, atol=1e-5), evals)
    else
        deginds = [length(evals) ÷ 2, length(evals) ÷ 2 + 1]
    end
    
    if !isempty(deginds)
        println("Degenerate states found, solving...")
        degstates = evecs[:, deginds]
        return solvedegenerate(degstates)
    else
        println("No degenerate states found.")
        return zeros(S, size(H, 1), 2)
    end 
end
