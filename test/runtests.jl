using Test
using ExactDiagonalize
using LinearAlgebra
using SparseArrays

@testset "ExactDiagonalize.jl Tests" begin

    @testset "Basis Construction" begin
        # Test SpinBasis for fixed particle number
        basis_num = SpinBasis(4; num=2)
        @test basis_num.num == 2
        @test length(basis_num.bitsvec) == 6  # C(4,2) = 6
        
        # Test Full dimension SpinBasis for all states
        basis_full = SpinBasis(3)
        @test basis_full.lsize == 3
        @test length(basis_full.bitsvec) == 8  # 2^3 = 8
    end

    @testset "State Creation" begin
        # Test ProductState creation from integer bits
        bits = 0x0000a
        basis_num = SpinBasis(4; num=2)
        state_num = ProductState(basis_num, bits)
        @test state_num.basis.num == count_ones(bits)
        @test length(state_num.vector) == 6  # Hilbert space dimension for 2 particles
        @test norm(state_num) ≈ 1.0  # Normalized
        
        # Test full dimension State creation
        basis_full = SpinBasis(3)
        state_full = ProductState(basis_full, 0x00005)
        @test state_full.basis.lsize == 3
        @test length(state_full.vector) == 8
        @test state_full.vector[5 + 1] ≈ 1.0
        
        # Test ProductState creation from binary string
        state_str = ProductState(basis_num, "1010")
        @test state_str.basis.num == 2
        
        # Test full dimension State creation from binary string
        state_full_str = ProductState(basis_full, "101")
        @test state_full_str.basis.lsize == 3
        
        # Test state with different element types
        state_complex = ProductState(ComplexF64, basis_full, 0x00006)
        @test eltype(state_complex.vector) == ComplexF64
        
        # Test RandomState creation
        random_st = RandomState(basis_full)
        @test norm(random_st) ≈ 1.0  # Should be normalized
        
        # Test low-level product_state function
        vec_state = product_state(basis_full, 0x00005)
        @test norm(vec_state) ≈ 1.0
        @test vec_state[6] ≈ 1.0
        
        # Test low-level random_state function
        vec_random = random_state(basis_full)
        @test norm(vec_random) ≈ 1.0
    end

    @testset "Bit Manipulation Utilities" begin
        using ExactDiagonalize: readbit, flip, signbetween, splitbasis
        
        bits = 0x0000a
        
        # Test readbit for single position
        @test readbit(bits, 0x01) == false  # bit 0 (1-based: position 1)
        @test readbit(bits, 0x02) == true   # bit 1
        @test readbit(bits, 0x03) == false  # bit 2
        @test readbit(bits, 0x04) == true   # bit 3
        
        # Test readbit for tuple of positions
        b1, b2 = readbit(bits, (0x01, 0x03))
        @test b1 == false
        @test b2 == false
        
        # Test flip single bit
        flipped = flip(bits, 0x01)
        @test readbit(flipped, 0x01) == true
        
        # Test flip with boolean
        flipped_cond = flip(bits, 0x02, false)
        @test readbit(flipped_cond, 0x02) == true
        
        # Test sign between bits
        sign_val = signbetween(0x0000e, 0x01, 0x04)  # ones between positions 1 and 4
        @test sign_val isa Int
        
        # Test splitbasis
        right, left = splitbasis(0x00001a, 0x02)
        @test right == 0b010
        @test left == 0b110
    end

    @testset "Operator Construction" begin
        set_systype(:Spin)
        
        # Test single qubit operator
        op_z = Op(:Z, 1)
        @test op_z.name == ExactDiagonalize.OP_Z
        @test op_z.loc == 0x01
        
        # Test two-qubit operator
        op_cx = Op(:CX, (1, 2))
        @test op_cx.name == ExactDiagonalize.OP_CX
        @test op_cx.loc1 == 0x01
        @test op_cx.loc2 == 0x02
        
        # Test OpSum construction
        operators = [
            (1.0, :Z, 1, :Z, 2),
            (0.5, :X, 1)
        ]
        ops_sum = OpSum(ComplexF64, operators)
        @test length(ops_sum.covec) == 2
        @test ops_sum.covec[1] ≈ 1.0
        @test ops_sum.covec[2] ≈ 0.5

        ops_sum += (0.5, :X, 2)
        @test ops_sum.opvec[3][1].name == ExactDiagonalize.OP_X
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
        ops_sum = OpSum(Float64)
        ops_sum += 1.0, :Z, 1, :Z, 2
        ops_sum += 0.5, :X, 1
        ops_sum += 0.5, :X, 2
        
        # Test with Full dimension SpinBasis
        basis = SpinBasis(2)
        hmat = makeHamiltonian(ops_sum, basis)
        @test size(hmat) == (4, 4)
        @test ishermitian(hmat)
        
        # Test with fixed number SpinBasis
        basis_num = SpinBasis(3; num=1)
        hmat_num = makeHamiltonian(ops_sum, basis_num)
        @test size(hmat_num)[1] == length(basis_num.bitsvec)
        
        # Test matrixform for single operator
        ops_vec = [Op(:X, 1), Op(:X, 2)]
        mat_single = matrixform(ops_vec, basis)
        @test size(mat_single) == (4, 4)
    end

    @testset "Spectrum Computation" begin
        set_systype(:Spin)
        
        # Create simple transverse Ising model
        ops_sum = OpSum(ComplexF64)
        ops_sum += 1.0, :Z, 1, :Z, 2
        ops_sum += 0.5, :X, 1
        ops_sum += 0.5, :X, 2
        
        # Test spectrum with Full dimension SpinBasis
        basis = SpinBasis(2)
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
        ops_list = (1.0, :Z, 1)
        obs = OperatorObserver(ops_list, SpinBasis(2))
        
        @test length(obs.data) == 0
        
        # Record a measurement
        record!(obs, psi, 1)
        @test length(obs.data) == 1
        @test obs.data[1] == -1
        
        # Test OpSumObserver for energy tracking
        set_systype(:Spin)
        ops_sum = OpSum(ComplexF64)
        ops_sum += 1.0, :Z, 1, :Z, 2
        basis = SpinBasis(2)
        obs_energy = OpSumObserver(ops_sum, basis)
        @test length(obs_energy.data) == 0
        
        record!(obs_energy, psi, 1)
        @test length(obs_energy.data) == 1
        
        # Test ZObserver directly
        obs_z = ZObserver(1, basis)
        @test length(obs_z.data) == 0
        record!(obs_z, psi, 1)
        @test length(obs_z.data) == 1
        
        # Test XObserver directly
        obs_x = XObserver(1, basis)
        @test length(obs_x.data) == 0
        record!(obs_x, psi, 1)
        @test length(obs_x.data) == 1
    end

    @testset "Time Evolution: Exact Diagonalization" begin
        set_systype(:Spin)
        
        # Create simple Hamiltonian
        operators = [
            (1.0, :X, 1, :X, 2)
        ]
        ops_sum = OpSum(ComplexF64, operators)
        
        # Initial state
        basis = SpinBasis(2)
        init_state = ProductState(basis, 0x00000)
        
        # Evolve with time array and observer
        ts = 0.0 : 0.05 : 0.1
        obs = XObserver(1, basis)
        final_state = timeEvolve(ops_sum, init_state, ts, obs, exact())
        
        @test length(final_state.vector) == length(init_state.vector)
        @test norm(final_state) ≈ 1.0 atol = 1e-14 # Norm preserved
        @test length(obs.data) == length(ts)  # Observer recorded at each time point
    end

    @testset "Time Evolution: RK4 Method" begin
        set_systype(:Spin)
        
        operators = [
            (1.0, :Z, 1, :Z, 2)
        ]
        ops_sum = OpSum(ComplexF64, operators)
        
        basis = SpinBasis(2)
        init_state = ProductState(basis, 0x00000)
        ts = 0.00 : 0.01 : 0.02
        obs = ZObserver(1, basis)
        
        final_state = timeEvolve(ops_sum, init_state, ts, obs, rk4())
        
        @test length(final_state.vector) == 4
        @test norm(final_state) ≈ 1.0 atol = 1e-10
        @test length(obs.data) == 3
    end

    @testset "Time Evolution: Sparse Matrix Method" begin
        set_systype(:Spin)
        
        operators = [
            (1.0, :Z, 1, :Z, 2)
        ]
        ops_sum = OpSum(ComplexF64, operators)
        
        basis = SpinBasis(2)
        init_state = ProductState(basis, 0x00001)
        ts = 0.00 : 0.01 : 0.02
        obs = XObserver(1, basis)
        
        final_state = timeEvolve(ops_sum, init_state, ts, obs, spmat())
        
        @test length(final_state.vector) == 4
        @test norm(final_state) ≈ 1.0 atol=1e-10
        @test length(obs.data) == 3
    end

    @testset "Spin Operators Acting on Basis" begin
        set_systype(:Spin)
        
        # Test Z operator
        bits = 0x00005
        new_bits, element = act(Op(:Z, 1), bits)
        @test new_bits == bits  # Z doesn't change the state
        @test element ∈ (1, -1)  # Z eigenvalues
        
        # Test X operator
        bits = 0x00005
        new_bits, element = act(Op(:X, 1), bits)
        @test new_bits == flip(bits, 0x01)  # X flips the bit
        @test element == 1
        
        # Test iY operator  
        bits = 0x00005
        new_bits, element = act(Op(:iY, 1), bits)
        @test new_bits == flip(bits, 0x01)
        @test element == -1
        
        # Test sigma plus operator
        bits = 0x00005
        new_bits, element = act(Op(:σp, 1), bits)
        @test element == 0
        
        # Test sigma minus operator
        bits = 0x00005
        new_bits, element = act(Op(:σm, 1), bits)
        @test new_bits == flip(bits, 0x01)
        
        # Test CX operator
        bits = 0x00003
        new_bits, element = act(Op(:CX, (1, 2)), bits)
        @test element == 1
        
        # Test CZ operator
        bits = 0x00003
        new_bits, element = act(Op(:CZ, (1, 2)), bits)
        @test new_bits == bits
        @test element ∈ (1, -1)
        
        # Test act_seq - applying multiple operators
        ops_seq = [Op(:X, 1), Op(:X, 1)]  # Apply X twice (identity)
        bits_input = 0x00005
        new_bits_seq, elem_seq = act_seq(1.0, ops_seq, bits_input)
        @test new_bits_seq == bits_input  # X*X = I
        @test elem_seq == 1
    end

    @testset "Operator Application and Expectation Values" begin
        set_systype(:Spin)
        
        # Create a simple Hamiltonian and initial state
        basis = SpinBasis(2)
        init_state = ProductState(basis, 0x00000)  # |00⟩ state
        
        # Test apply function
        op_x1 = [Op(:X, 1)]
        new_state = apply(op_x1, init_state)
        @test norm(new_state) ≈ 1.0
        
        # Test apply! function (in-place)
        test_state = ProductState(basis, 0x00000)
        apply!(op_x1, test_state)
        @test norm(test_state) ≈ 1.0
        
        # Test expected value with OpSum (energy)
        ops_sum = OpSum(ComplexF64)
        ops_sum += 1.0, :Z, 1, :Z, 2
        ops_sum += 0.5, :X, 1
        energy = expected(ops_sum, init_state)
        @test isa(energy, Real)
        
        # Test expected value with operator list
        ops_list = [Op(:Z, 1)]
        exp_val = expected(ops_list, init_state)
        @test isa(exp_val, Real)
        
        # Test dot product with operators: ⟨ψ|O|ψ⟩
        bra_state = ProductState(basis, 0x00000)
        ket_state = ProductState(basis, 0x00001)
        
        # ⟨ψ₁|O|ψ₂⟩ with OpSum
        dot_prod_opsum = dot(bra_state, ops_sum, ket_state)
        @test isa(dot_prod_opsum, ComplexF64)
    end
    
    @testset "Entanglement Entropy and Reduced Density Matrix" begin
        set_systype(:Spin)
        
        # Create a basis and a state
        basis = SpinBasis(4)
        
        # Create a superposition state that has entanglement
        state_vec = zeros(ComplexF64, length(basis.bitsvec))
        state_vec[1] = 1/sqrt(2)  # |0000⟩
        state_vec[end] = 1/sqrt(2)  # |1111⟩
        state = QState(basis, state_vec)
        
        # Test entanglement entropy
        S = ent_entropy(state)
        @test S > 0  # Should have entanglement for this state
        @test S ≤ log(2) * 2  # Upper bound for bipartition of 4-site system
        
        # Test with different bipartition
        S_cut3 = ent_entropy(state, 3)
        @test isa(S_cut3, Float64)
        
        # Test reduced density matrix
        rho_A = reduced_density_matrix(state; subsys='A')
        @test size(rho_A) == (4, 4)
        @test ishermitian(rho_A)
        
        rho_B = reduced_density_matrix(state; subsys='B')
        @test size(rho_B) == (4, 4)
        @test ishermitian(rho_B)
    end

    @testset "State Index Lookup" begin
        using ExactDiagonalize: findindex
        
        # Test findindex for fixed number SpinBasis
        basis = SpinBasis(4; num=2)
        idx, dist = findindex(basis, 0x00003)
        @test idx == 1
        @test dist == 0
        
        # Test with invalid state (wrong particle number)
        idx_invalid, dist_invalid = findindex(basis, 0x00001)
        @test idx_invalid == basis.dim + 1
        @test dist_invalid == 0
        
        # Test findindex for Full dimension SpinBasis
        basis_full = SpinBasis(3)
        idx_full, dist_full = findindex(basis_full, 0x00005)
        @test idx_full == 0x00005 + 1
        @test dist_full == 0
    end

end