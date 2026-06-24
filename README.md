# ExactDiagonalize.jl

A Julia package for exact diagonalization of quantum many-body systems. This package provides tools for constructing Hamiltonians, computing eigenspectra, and performing quantum dynamics simulations via time evolution.

## Overview

ExactDiagonalize enables computational studies of small to moderate-sized quantum systems through exact numerical methods. It supports:

- **Hamiltonian Construction**: Build quantum Hamiltonians from sum of operator products, i.e. almost **handwriting form**
- **Spectrum Computation**: Diagonalize Hamiltonians to obtain eigenvalues and eigenvectors
- **Time Evolution**: Simulate quantum dynamics using exact diagonalized results, RK4 based ODE solver or sparse matrix multiplication method
- **Observable Tracking**: Record **highly customizable** time-dependent expectation values during evolution
- **Sparse Matrix Support**: Efficient sparse matrix representations of quantum operators

## Installation

Add ExactDiagonalize to your Julia project (maybe later, for I have not registered the package yet):

```julia
using Pkg
Pkg.add("ExactDiagonalize")
```

Or add directly from the repository (**use this method for now**):

```julia
using Pkg
Pkg.add(url="https://github.com/Rexirium/ExactDiagonalize.jl")
```

### Requirements

- Julia 1.12+
- Standard library: `LinearAlgebra`, `SparseArrays`
- Optional: `CairoMakie` for visualization

## Quick Start

### Basic XXZ Model Example

Simulate time evolution of an XXZ spin chain:

$$
    H = J \sum_{i = 1}^L \left(X_i X_{i+1} + Y_i Y_{i+1} + \Delta Z_i Z_{i+1} \right). 
$$

```julia
using ExactDiagonalize

# Set system type to spin
set_systype(:Spin)
L, N = 10, 1  # System size and particle number
Δ = 1.0       # Interaction strength

# Initial state: single excitation at site 1
basis = SpinBasis(L; num = N)
init = ProductState(basis, "1000000000")

# Build XXZ Hamiltonian
opsum = OpSum(Float64)
for j in 1:L
    nj = mod1(j + 1, L) # PBC
    opsum += Δ, :Z, j, :Z, nj
    opsum += 1.0, :X, j, :X, nj
    opsum += -1.0, :iY, j, :iY, nj
end

# Define observer for tracking Z expectation on site 1 and time points
obs = ZObserver(1, basis)  # Track Z expectation at site 1
ts = 0.0:0.05:10.0

# Run time evolution with exact diagonalization
timeEvolve(opsum, init, ts, obs, exact())

@show obs.data # the data recorded
```

## Core API

### State Representation

- **`QState`**: Quantum state defined with basis and coefficient vector
- **`SpinBasis(lsize; a, num, kint)`**: Basis for spin one-half system (only one dimension so far)
  - The system has `lsize` sites with either spin-up $\uparrow$ represented by `1` or spin-down $\downarrow$ represented by `0`, i.e. on-site Hilbert subspace has dimension $d=2$.
  - `a`: lattice spacing parameter (default: 1). Must divide `lsize` evenly.
  - If only `num` is passed, the method will generate basis in subsector with conserved total particle number (total spin-z) `num`.
  - If only `kint` is passed, the method will generate basis in subsector with conserved quasi-momentum $k = 2\pi m/(L/a),\; m = 0, 1, \ldots L/a-1$, with $m$ labeled by `kint`.
  - If none of the above is passed, the method will generate all $2^L$ full basis.
- **`findindex(basis, bits)`**: Find the index of a product state (bitstring) in the basis
  - Returns tuple `(index, distance)` where `index` is the basis index and `distance` is for momentum conservation
  - Returns `(basis.dim + 1, 0)` if state is not in basis
- **`ProductState(basis, args...)`** / **`product_state(basis, args...)`**: Create product states (computational basis states)
  - Supports multiple input formats: bitstring `UInt32`, function `func(site) -> Symbol`, symbol vector, or binary string
  - Examples: `ProductState(basis, 0x0001)`, `ProductState(basis, "1000000000")`, `ProductState(basis, [:Up, :Dn, :Up, ...])`
- **`RandomState(basis)`** / **`random_state(basis)`**: Create random normalized quantum states

### Operators

- **`SpinOp`**: Base type for spin operators
  - `SpinOp1`: Single-site spin operators ($X$, $Z$, $iY$, $\sigma^+$, $\sigma^-$, $P_\uparrow$, $P_\downarrow$)
  - `SpinOp2`: Two-site spin operators ($CX$, $CZ$)
- **`Op(name, loc)`**: Constructor for individual local operator, automatically determines operator type
  - `name`: Symbol specifying the operator (`:X`, `:Z`, `:iY`, `:σp`, `:σm`, `:Pup`, `:Pdn`, `:CX`, `:CZ`)
  - `loc`: Site location (integer for single-site, tuple for two-site)
- **`OpSum{T, O}`**: Linear combination of operator sequences (represents Hamiltonians)
  - Can be built iteratively: `opsum = OpSum(Float64)` then `opsum += coeff, :op1, site1, :op2, site2, ...`
  - Or from vector of tuples: `OpSum(Float64, [(coeff1, :op1, site1, ...), ...])`

### Core Functions

#### Spectrum Computation
- **`spectrum(opsum, basis; retvecs=false)`**: Compute eigenvalues and optionally eigenvectors of an `OpSum` in given basis
  - Returns eigenvalues if `retvecs=false` (default)
  - Returns `Eigen` object (eigenvalues and eigenvectors) if `retvecs=true`
- **`spectrum(opsum, lsize)`**: Compute full spectrum by summing over all particle number sectors (for `SpinBasis` with `num` conservation)

#### Time Evolution
- **`timeEvolve(opsum, init, ts, obs, alg)`**: Time evolution with observable recording
  - `init`: Initial state as `QState` or basis with state vector
  - `ts`: Array of time points
  - `obs`: Observer for recording observables (see below)
  - `alg`: Algorithm selector - `exact()`, `rk4()`, or `spmat(; order=4)` (default: `exact()`)
  - Returns final state as `QState` or vector

#### Observable Computation
- **`record!(observer, state, step)`**: Record observable at time step (called internally by `timeEvolve`)
- **`expected(ops, psi)`** / **`expected(opsum, psi)`**: Compute expected value $\langle \psi | O | \psi \rangle$
  - First form: `ops` is a vector of operators, `psi` is `QState`
  - Second form: `opsum` is `OpSum`, `psi` is `QState` (useful for computing total energy, etc.)
- **`dot(x, ops, y)`**: Compute inner product $\langle x | O | y \rangle$ with operator `O`

#### Operator Application
- **`apply(ops, psi)`**: Apply operator(s) to state, returning new state
- **`apply!(ops, psi)`**: In-place operator application to state (saves memory)
- **`act(op, bits)`**: Apply single operator to bitstring, returns new bitstring and phase factor
- **`act_seq(coeff, ops_list, bits)`**: Apply sequence of operators to bitstring

#### Hamiltonian Construction
- **`makeHamiltonian(opsum, basis; sparsed=false, dtype=ComplexF64)`**: Convert `OpSum` to matrix representation
  - Returns dense matrix by default (required for exact diagonalization with `LinearAlgebra.eigen`)
  - Returns sparse matrix if `sparsed=true` (for RK4 and sparse matrix methods)
  - Used internally by evolution algorithms
- **`matrixform(ops, basis[, coeff]; sparsed=true, dtype=Float64)`**: Build matrix representation of single operator sequence

#### Entanglement Properties
- **`ent_entropy(psi, b=lsize÷2)`**: Von Neumann entanglement entropy of subsystem with `b` sites
  - Works with `QState` or basis with state vector
- **`reduced_density_matrix(psi, b=lsize÷2; subsys='A')`**: Reduced density matrix for subsystem A or B

### Observer Types

- **`ZObserver(loc, basis)`**: Track Z-expectation value $\langle Z_i \rangle$ at site `loc`
- **`XObserver(loc, basis)`**: Track X-expectation value $\langle X_i \rangle$ at site `loc`
- **`OpSumObserver(opsum, basis)`**: Track expectation value of entire `OpSum` (e.g., total energy)
- **`OperatorObserver(os, basis)`**: Track expectation value of operator sequence `os` specified as tuple

All observers have `.data` field containing the recorded time-dependent expectation values.

### System Configuration

- **`set_systype(type)`**: Set system type (`:Spin` supported; `:Fermion` under development)
- **`get_systype()`**: Query current system type

## Examples

See the `examples/` directory for complete working examples:

- `xxzmodel.jl`: XXZ spin chain with time evolution and observable tracking ($U(1)$ symmetry used)
- `tfimodel.jl`: Transverse field Ising model time evolution and observable tracking (using full state)

### Advanced Usage Examples

#### Using Different Time Evolution Algorithms

```julia
# Exact diagonalization (default)
timeEvolve(opsum, init, ts, obs, exact())

# RK4 ODE solver (for intermediate-sized systems)
timeEvolve(opsum, init, ts, obs, rk4())

# Sparse matrix exponential (good for larger systems)
timeEvolve(opsum, init, ts, obs, spmat(; order=6))
```

#### Tracking Multiple Observables

```julia
# Track energy (OpSumObserver tracks full Hamiltonian)
obs_energy = OpSumObserver(opsum, basis)

# Track local expectation values
obs_z1 = ZObserver(1, basis)
obs_x2 = XObserver(2, basis)

# Run evolution once, all observers record
timeEvolve(opsum, init, ts, obs_energy)
timeEvolve(opsum, init, ts, obs_z1)
timeEvolve(opsum, init, ts, obs_x2)
```

#### Computing Entanglement

```julia
# Get final state from evolution
final_state = timeEvolve(opsum, init, ts, obs)

# Compute entanglement entropy
S_vn = ent_entropy(final_state)  # Default: bipartition at L/2

# Compute for different cut position
S_vn_cut = ent_entropy(final_state, b=3)  # Bipartition after 3 sites

# Get reduced density matrix
ρ_A = reduced_density_matrix(final_state; subsys='A')
ρ_B = reduced_density_matrix(final_state; subsys='B')
```

#### Computing Spectrum

```julia
# Compute eigenvalues in fixed particle number sector
evals = spectrum(opsum, basis)

# Compute eigenvalues and eigenvectors
eig_result = spectrum(opsum, basis; retvecs=true)
evals = eig_result.values
evecs = eig_result.vectors

# Compute full spectrum across all particle sectors
evals_full = spectrum(opsum, L)
```

#### Working with Operators and Expectation Values

```julia
# Compute expectation value of energy
E = expected(opsum, final_state)

# Apply operator to state
op_vec = Op(:Z, 1)  # Z operator at site 1
new_state = apply([op_vec], final_state)

# Compute correlation function ⟨ψ|O₁O₂|ψ⟩
ops = [Op(:Z, 1), Op(:Z, 3)]
corr = expected(ops, final_state)

# Compute matrix representation of Hamiltonian (for advanced use)
H_dense = makeHamiltonian(opsum, basis; sparsed=false)
H_sparse = makeHamiltonian(opsum, basis; sparsed=true)
```

## Key Features

- **Efficient Sparse Representation**: Leverages sparse matrix formats for memory efficiency
- **Flexible Operator Syntax**: Intuitive specification of Hamiltonians. Just Build it from what you see on papers
- **Observable Recording**: Built-in framework for tracking time-dependent measurements. Highly **flexible, customizable and expandable**
- **Hardware Acceleration**: Optional MKL support for accelerated linear algebra

## Project Structure

```
src/
  ├── ExactDiagonalize.jl   # Main module
  ├── state_basis.jl        # State and basis definitions
  ├── operators.jl          # Operator and Hamiltonian construction
  ├── observers.jl          # Observer system for tracking observables
  ├── exactdiag.jl          # Exact diagonalization time evolution
  ├── ode_solver.jl         # Time evolution by RK4 ODE solver
  ├── sparsemat.jl          # Sparse matrix exponential time evolution
  ├── entanglement.jl       # Entanglement entropy and reduced density matrices
  └── utils.jl              # Helper utilities for bitstring operations
```

## Acknowlegement

- The motivation to develop this package comes from the homework of the course *Nonequilibrium Dynamics in Closed Quantum System* taught by HongZheng Zhao in the spring semester of 2026 at the School of Physics, Peking University.
- The ideas of `OpSum` and `AbstractObserver` are inspired by the [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl) project by Matthew Fishman (@mtfishman) and other developers.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

See LICENSE file for details.

## Citation

If you use ExactDiagonalize in your research, please cite this repository.
