using Test
using Revise
using ExactDiagonalize
using LinearAlgebra
using SparseArrays

@testset "ExactDiagonalize.jl Tests" begin

    @testset "Basis Construction" begin
        # Test NumBasis for fixed particle number
        basis_num = NumBasis(4, 2)
        @test basis_num.num == 2
        @test length(basis_num.bitsvec) == 6  # C(4,2) = 6
        
        # Test FullBasis for all states
        basis_full = FullBasis(3)
        @test basis_full.lsize == 3
        @test length(basis_full.bitsvec) == 8  # 2^3 = 8
    end

    @testset "State Creation" begin
        # Test NumState creation from integer bits
        bits = 10
        state_num = NumState(4, bits)
        @test state_num.basis.num == count_ones(bits)
        @test length(state_num.vector) == 6  # Hilbert space dimension for 2 particles
        @test sum(abs.(state_num.vector)) ≈ 1.0  # Normalized
        
        # Test FullState creation
        state_full = FullState(3, 0b101)
        @test state_full.basis.lsize == 3
        @test length(state_full.vector) == 8
        @test state_full.vector[0b101 + 1] ≈ 1.0
        
        # Test NumState creation from binary string
        state_str = NumState("1010")
        @test state_str.basis.num == 2
        
        # Test FullState creation from binary string
        state_full_str = FullState("101")
        @test state_full_str.basis.lsize == 3
        
        # Test state with different element types
        state_complex = NumState(3, 0b110; type=ComplexF64)
        @test eltype(state_complex.vector) == ComplexF64
    end

    @testset "Bit Manipulation Utilities" begin
        using ExactDiagonalize: readbit, flip, signbetween, splitbasis
        
        bits = 0b1010
        
        # Test readbit for single position
        @test readbit(bits, 1) == false  # bit 0 (1-based: position 1)
        @test readbit(bits, 2) == true   # bit 1
        @test readbit(bits, 3) == false  # bit 2
        @test readbit(bits, 4) == true   # bit 3
        
        # Test readbit for tuple of positions
        b1, b2 = readbit(bits, (1, 3))
        @test b1 == false
        @test b2 == false
        
        # Test flip single bit
        flipped = flip(bits, 1)
        @test readbit(flipped, 1) == true
        
        # Test flip with boolean
        flipped_cond = flip(bits, 2, false)
        @test readbit(flipped_cond, 2) == false
        
        # Test sign between bits
        sign_val = signbetween(0b1110, 1, 4)  # ones between positions 1 and 4
        @test sign_val isa Int
        
        # Test splitbasis
        right, left = splitbasis(0b11010, 2)
        @test right == 0b10
        @test left == 0b111
    end

    @testset "Operator Construction" begin
        set_systype(:Spin)
        
        # Test single qubit operator
        op_z = SpinOp(:Z, 1)
        @test op_z.name == :Z
        @test op_z.loc == 1
        
        # Test two-qubit operator
        op_cx = SpinOp(:CX, (1, 2))
        @test op_cx.name == :CX
        @test op_cx.loc == (1, 2)
        
        # Test OpSum construction
        operators = [
            (1.0, :Z, 1, :Z, 2),
            (0.5, :X, 1)
        ]
        ops_sum = OpSum(operators, ComplexF64)
        @test length(ops_sum.covec) == 2
        @test ops_sum.covec[1] ≈ 1.0
        @test ops_sum.covec[2] ≈ 0.5
    end

    @testset "System Type Configuration" begin
        # Test system type get/set
        set_systype(:Spin)
        @test get_systype() == Val(:Spin)
        
        set_systype(:Fermion)
        @test get_systype() == Val(:Fermion)
    end

    @testset "Hamiltonian Construction" begin
        set_systype(:Spin)
        
        # Create simple Ising model: H = Z_0 Z_1 + 0.5 X_1
        operators = [
            (1.0, :Z, 1, :Z, 2),
            (0.5, :X, 2)
        ]
        ops_sum = OpSum(operators, ComplexF64)
        
        # Test with FullBasis
        basis = FullBasis(2)
        hmat = makeHamiltonian(ops_sum, basis)
        @test size(hmat) == (4, 4)
        @test ishermitian(hmat)
        
        # Test with NumBasis
        basis_num = NumBasis(3, 1)
        hmat_num = makeHamiltonian(ops_sum, basis_num)
        @test size(hmat_num)[1] == length(basis_num.bitsvec)
    end

    @testset "Spectrum Computation" begin
        set_systype(:Spin)
        
        # Create simple transverse Ising model
        operators = [
            (1.0, :Z, 1, :Z, 2),
            (0.5, :X, 1),
            (0.5, :X, 2)
        ]
        ops_sum = OpSum(operators, ComplexF64)
        
        # Test spectrum with FullBasis
        basis = FullBasis(2)
        eigs = spectrum(ops_sum, basis)
        @test length(eigs) == 4
        @test all(isreal, eigs)  # Eigenvalues of Hermitian matrix are real
        @test issorted(eigs)  # Should be sorted
        
        # Test spectrum with eigenvectors
        eigs_vals, eigs_vecs = spectrum(ops_sum, basis; retvecs=true)
        @test size(eigs_vecs) == (4, 4)
        
        # Test spectrum for fixed particle number
        eigs_fixed = spectrum(ops_sum, 2)  # System size 2
        @test length(eigs_fixed) == 4
    end

    @testset "Observables and Recording" begin
        # Create some test data
        psi = zeros(ComplexF64, 4)
        psi[1] = 1.0
        
        # Test OperatorObserver
        ops_list = [OpSum([(1.0, :Z, 1)], ComplexF64)]
        obs = OperatorObserver(ops_list)
        
        @test length(obs.measured) == 1
        
        # Record a measurement
        record!(obs, psi, 1)
        @test length(obs.times) == 1
        @test obs.times[1] == 1
    end

    @testset "Time Evolution: Exact Diagonalization" begin
        set_systype(:Spin)
        
        # Create simple Hamiltonian
        operators = [
            (1.0, :X, 1, :X, 2)
        ]
        ops_sum = OpSum(operators, ComplexF64)
        
        # Initial state
        init_state = FullState(2, 0b00)
        
        # Evolve to short time
        tf = 0.1
        final_state = timeEvolve(ops_sum, init_state, tf)
        
        @test length(final_state.vector) == length(init_state.vector)
        @test norm(final_state.vector) ≈ norm(init_state.vector)  # Norm preserved
    end

    @testset "Time Evolution: RK4 Method" begin
        set_systype(:Spin)
        
        operators = [
            (1.0, :Z, 1)
        ]
        ops_sum = OpSum(operators, ComplexF64)
        
        init_state = FullState(2, 0b00)
        times = [0.0, 0.1, 0.2]
        obs = OperatorObserver([ops_sum])
        
        final_state = timeEvolve(ops_sum, init_state, times, obs, rk4())
        
        @test length(final_state.vector) == 4
        @test norm(final_state.vector) ≈ 1.0 atol=1e-10
        @test length(obs.times) == 3
    end

    @testset "Time Evolution: Sparse Matrix Method" begin
        set_systype(:Spin)
        
        operators = [
            (1.0, :Z, 1)
        ]
        ops_sum = OpSum(operators, ComplexF64)
        
        init_state = FullState(2, 0b01)
        times = [0.0, 0.1, 0.2]
        obs = OperatorObserver([ops_sum])
        
        final_state = timeEvolve(ops_sum, init_state, times, obs, spmat())
        
        @test length(final_state.vector) == 4
        @test norm(final_state.vector) ≈ 1.0 atol=1e-10
        @test length(obs.times) == 3
    end

    @testset "Spin Operators Acting on Basis" begin
        set_systype(:Spin)
        using ExactDiagonalize: act
        
        # Test Z operator
        bits = 0b101
        new_bits, element = act(SpinOp(:Z, 1), bits, ComplexF64)
        @test new_bits == bits  # Z doesn't change the state
        @test element ∈ (1.0, -1.0)  # Z eigenvalues
        
        # Test X operator
        bits = 0b101
        new_bits, element = act(SpinOp(:X, 1), bits, ComplexF64)
        @test new_bits == flip(bits, 1)  # X flips the bit
        @test element == 1.0
        
        # Test iY operator  
        bits = 0b101
        new_bits, element = act(SpinOp(:iY, 1), bits, ComplexF64)
        @test new_bits == flip(bits, 1)
        
        # Test sigma plus operator
        bits = 0b101
        new_bits, element = act(SpinOp(:σp, 1), bits, ComplexF64)
        @test new_bits == flip(bits, 1)
        
        # Test sigma minus operator
        bits = 0b101
        new_bits, element = act(SpinOp(:σm, 1), bits, ComplexF64)
        @test new_bits == flip(bits, 1)
        
        # Test CX operator
        bits = 0b11
        new_bits, element = act(SpinOp(:CX, (1, 2)), bits, ComplexF64)
        @test element == 1.0
        
        # Test CZ operator
        bits = 0b11
        new_bits, element = act(SpinOp(:CZ, (1, 2)), bits, ComplexF64)
        @test new_bits == bits
        @test element ∈ (1.0, -1.0)
    end

    @testset "State Index Lookup" begin
        using ExactDiagonalize: findindex
        
        # Test findindex for NumBasis
        basis = NumBasis(4, 2)
        idx = findindex(basis, 0b0011)
        @test idx > 0
        
        # Test with invalid state (wrong particle number)
        idx_invalid = findindex(basis, 0b0001)
        @test idx_invalid == 0
        
        # Test findindex for FullBasis
        basis_full = FullBasis(3)
        idx_full = findindex(basis_full, 0b101)
        @test idx_full == 0b101 + 1
    end

end