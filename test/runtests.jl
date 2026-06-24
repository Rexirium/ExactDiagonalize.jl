using Test
using ExactDiagonalize
using LinearAlgebra
using SparseArrays

using ExactDiagonalize: readbit, flip, signbetween, splitbasis

const TEST_ATOL = 1e-10

function transverse_ising_opsum(; J=1.0, h=0.5)
    opsum = OpSum(Float64)
    opsum += J, :Z, 1, :Z, 2
    opsum += h, :X, 1
    opsum += h, :X, 2
    return opsum
end

function zz_opsum()
    opsum = OpSum(ComplexF64)
    opsum += 1.0, :Z, 1, :Z, 2
    return opsum
end

@testset "ExactDiagonalize.jl" begin
    set_systype(:Spin)

    @testset "Basis construction and lookup" begin
        full = SpinBasis(3)
        fixed = SpinBasis(4; num=2)
        mixed_nums = SpinBasis(4; num=[1, 3])

        @test full.lsize == 3
        @test full.dim == 8
        @test collect(full.bitsvec) == UInt32.(0:7)

        @test fixed.num == 2
        @test fixed.dim == binomial(4, 2)
        @test count_ones.(fixed.bitsvec) == fill(2, fixed.dim)

        @test mixed_nums.dim == 8
        @test all(count_ones(bits) in (1, 3) for bits in mixed_nums.bitsvec)

        @test findindex(full, 0x00005) == (6, 0)
        @test findindex(fixed, 0x00003) == (1, 0)
        @test findindex(fixed, 0x00001) == (fixed.dim + 1, 0)
    end

    @testset "State constructors" begin
        basis = SpinBasis(4; num=2)
        full = SpinBasis(3)

        from_bits = ProductState(basis, 0x0000a)
        from_string = ProductState(basis, "1010")
        from_symbols = ProductState(full, [:Up, :Dn, :Up])
        from_function = ProductState(full, site -> isodd(site) ? :Up : :Dn)
        complex_state = ProductState(ComplexF64, full, 0x00006)
        random_qstate = RandomState(full)

        @test from_bits.vector == from_string.vector
        @test norm(from_bits) ≈ 1.0
        @test from_symbols.vector == product_state(full, 0x00005)
        @test from_function.vector == from_symbols.vector
        @test eltype(complex_state.vector) == ComplexF64
        @test norm(random_qstate) ≈ 1.0 atol=TEST_ATOL
        @test norm(random_state(full)) ≈ 1.0 atol=TEST_ATOL
    end

    @testset "Bit utilities" begin
        bits = 0x0000a

        @test readbit(bits, 0x01) == false
        @test readbit(bits, 0x02) == true
        @test readbit(bits, (0x01, 0x04)) == (false, true)

        @test flip(bits, 0x01) == 0x0000b
        @test flip(bits, 0x02, false) == bits
        @test flip(bits, 0x02, true) == 0x00008

        @test signbetween(0x0000e, 0x01, 0x04) == 1
        @test splitbasis(0x00001a, 0x02) == (0x00002, 0x00006)
    end

    @testset "System type configuration" begin
        set_systype(:Spin)
        @test get_systype() == Val(:Spin)

        set_systype(:Fermion)
        @test get_systype() == Val(:Fermion)

        set_systype(:Spin)
    end

    @testset "Operator construction and bit actions" begin
        op_z = Op(:Z, 1)
        op_cx = Op(:CX, (1, 2))

        @test op_z.name == ExactDiagonalize.OP_Z
        @test op_z.loc == 0x01
        @test op_cx.name == ExactDiagonalize.OP_CX
        @test op_cx.loc1 == 0x01
        @test op_cx.loc2 == 0x02

        opsum = OpSum(ComplexF64, [(1.0, :Z, 1, :Z, 2), (0.5, :X, 1)])
        opsum += (0.25, :X, 2)

        @test opsum.covec == ComplexF64[1.0, 0.5, 0.25]
        @test length.(opsum.opvec) == [2, 1, 1]
        @test opsum.opvec[3][1].name == ExactDiagonalize.OP_X

        bits = 0x00005
        @test act(Op(:Z, 1), bits) == (bits, 1)
        @test act(Op(:X, 1), bits) == (0x00004, 1)
        @test act(Op(:iY, 1), bits) == (0x00004, -1)
        @test act(Op(:σp, 1), bits)[2] == 0
        @test act(Op(:σm, 1), bits) == (0x00004, 1)
        @test act(Op(:CX, (1, 2)), 0x00003) == (0x00001, 1)
        @test act(Op(:CZ, (1, 2)), 0x00003) == (0x00003, 1)
        @test act_seq(2.0, [Op(:X, 1), Op(:X, 1)], bits) == (bits, 2.0)
    end

    @testset "Hamiltonians, spectra, and matrix forms" begin
        basis = SpinBasis(2)
        fixed = SpinBasis(3; num=1)
        opsum = transverse_ising_opsum()

        h_dense = makeHamiltonian(opsum, basis)
        h_sparse = makeHamiltonian(opsum, basis; sparsed=true)
        x1x2 = matrixform([Op(:X, 1), Op(:X, 2)], basis)
        identity = Matrix{Float64}(I, basis.dim, basis.dim)

        @test size(h_dense) == (4, 4)
        @test h_sparse isa SparseMatrixCSC
        @test Matrix(h_sparse) == h_dense
        @test ishermitian(h_dense)
        @test size(makeHamiltonian(opsum, fixed)) == (fixed.dim, fixed.dim)
        @test Matrix(x1x2)^2 ≈ identity

        evals = spectrum(opsum, basis)
        eig = spectrum(opsum, basis; retvecs=true)
        conserving_opsum = zz_opsum()
        sector_evals = spectrum(conserving_opsum, 2)
        conserving_evals = spectrum(conserving_opsum, basis)

        @test length(evals) == basis.dim
        @test issorted(evals)
        @test eig.values ≈ evals
        @test eig.vectors' * eig.vectors ≈ identity atol=TEST_ATOL
        @test sort(sector_evals) ≈ conserving_evals
    end

    @testset "Observers and expectation values" begin
        basis = SpinBasis(2)
        state = ProductState(basis, "00")
        plus_state = QState(basis, fill(0.5 + 0.0im, basis.dim))
        opsum = zz_opsum()

        z_obs = ZObserver(1, basis)
        x_obs = XObserver(1, basis)
        energy_obs = OpSumObserver(opsum, basis)
        corr_obs = OperatorObserver((1.0, :Z, 1, :Z, 2), basis)

        record!(z_obs, state.vector, 1)
        record!(x_obs, state.vector, 1)
        record!(energy_obs, state.vector, 1)
        record!(corr_obs, state.vector, 1)

        @test z_obs.data == [-1.0]
        @test x_obs.data == [0.0]
        @test energy_obs.data == [1.0]
        @test corr_obs.data == [1.0]

        @test expected(opsum, state) == 1.0
        @test expected([Op(:Z, 1)], state) == -1.0
        @test expected([Op(:X, 1)], plus_state) ≈ 1.0
        @test dot(state, [Op(:X, 1)], ProductState(basis, "01")) ≈ 1.0
        @test dot(state, opsum, state) ≈ 1.0
    end

    @testset "Operator application" begin
        basis = SpinBasis(2)
        state = ProductState(basis, "00")
        x1 = [Op(:X, 1)]

        flipped = apply(x1, state)
        inplace = ProductState(basis, "00")
        apply!(x1, inplace)

        @test flipped.vector == ProductState(basis, "01").vector
        @test inplace.vector == flipped.vector

        h_state = apply(zz_opsum(), state)
        apply!(zz_opsum(), state)

        @test h_state.vector == ProductState(basis, "00").vector
        @test state.vector == h_state.vector
    end

    @testset "Time evolution" begin
        basis = SpinBasis(2)
        init = ProductState(basis, "00")
        ts = 0.0:0.01:0.02

        for (name, alg, atol) in [
            ("exact", exact(), 1e-14),
            ("rk4", rk4(), 1e-10),
            ("sparse matrix", spmat(; order=6), 1e-8),
        ]
            @testset "$name algorithm" begin
                obs = ZObserver(1, basis)
                final = timeEvolve(zz_opsum(), init, ts, obs, alg)

                @test final isa QState
                @test final.basis == basis
                @test norm(final) ≈ 1.0 atol=atol
                @test length(obs.data) == length(ts)
                @test all(obs.data .≈ -1.0)
            end
        end
    end

    @testset "Entanglement entropy and reduced density matrices" begin
        basis = SpinBasis(4)
        product = ProductState(basis, "0000")

        ghz_vec = zeros(ComplexF64, basis.dim)
        ghz_vec[1] = inv(sqrt(2))
        ghz_vec[end] = inv(sqrt(2))
        ghz = QState(basis, ghz_vec)

        @test ent_entropy(product) ≈ 0.0 atol=TEST_ATOL
        @test ent_entropy(ghz) ≈ log(2) atol=TEST_ATOL
        @test ent_entropy(ghz, 3) ≈ log(2) atol=TEST_ATOL

        ρ_A = reduced_density_matrix(ghz; subsys='A')
        ρ_B = reduced_density_matrix(ghz; subsys='B')

        @test size(ρ_A) == (4, 4)
        @test size(ρ_B) == (4, 4)
        @test ishermitian(ρ_A)
        @test ishermitian(ρ_B)
        @test tr(ρ_A) ≈ 1.0
        @test tr(ρ_B) ≈ 1.0
    end
end
