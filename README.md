# ExactDiagonalize.jl

ExactDiagonalize.jl is a Julia package for exact diagonalization studies of
small to moderate quantum many-body systems. It focuses on a compact operator
syntax for building Hamiltonians, computing spectra, and simulating real-time
dynamics with observable recording.

[![Julia](https://img.shields.io/badge/Julia-1.12%2B-9558B2.svg)](https://julialang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Highlights

- **Paper-like Hamiltonian construction** with `OpSum`
- **Spin-1/2 basis support** for full Hilbert spaces, fixed particle-number sectors, and momentum sectors
- **Spectrum computation** with dense exact diagonalization
- **Time evolution** via exact diagonalization, RK4 integration, or sparse matrix propagation
- **Observable recording** through a lightweight observer interface
- **Entanglement tools** for von Neumann entropy and reduced density matrices

## Installation

The package is not registered yet, so install it directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/Rexirium/ExactDiagonalize.jl")
```

Requirements:

- Julia 1.12+
- Standard libraries: `LinearAlgebra`, `SparseArrays`
- Optional plotting: `CairoMakie`
- Optional acceleration: `MKL`

## Quick Start

The following example evolves a single excitation in an XXZ spin chain with
periodic boundary conditions:

```julia
using ExactDiagonalize

set_systype(:Spin)

L = 10          # system size
N = 1           # fixed particle number
Δ = 1.0         # interaction strength
ts = 0.0:0.05:10.0

basis = SpinBasis(L; num=N)
init = ProductState(basis, "1000000000")

opsum = OpSum(Float64)
for j in 1:L
    nj = mod1(j + 1, L)
    opsum += Δ, :Z, j, :Z, nj
    opsum += 1.0, :X, j, :X, nj
    opsum += -1.0, :iY, j, :iY, nj
end

obs = ZObserver(1, basis)
final_state = timeEvolve(opsum, init, ts, obs, exact())

@show obs.data
```

This corresponds to

$$
H = \sum_{i=1}^{L}
\left(X_i X_{i+1} + Y_i Y_{i+1} + \Delta Z_i Z_{i+1}\right).
$$

## Core Concepts

### Basis and States

`SpinBasis` describes the computational basis for a one-dimensional spin-1/2
system. Spin-up is represented by `1`, spin-down by `0`, and the local Hilbert
space dimension is `d = 2`.

```julia
basis_full = SpinBasis(8)          # full 2^L basis
basis_num = SpinBasis(8; num=2)    # fixed particle-number sector
basis_k = SpinBasis(8; kint=1)     # fixed momentum sector
```

Useful state constructors:

```julia
ProductState(basis, "1010")
ProductState(basis, 0x0005)
ProductState(basis, [:Up, :Dn, :Up, :Dn])
RandomState(basis)
```

### Operators and Hamiltonians

Local operators are created with `Op`, while Hamiltonians are usually assembled
with `OpSum`:

```julia
op = Op(:Z, 1)

opsum = OpSum(Float64)
opsum += 1.0, :Z, 1, :Z, 2
opsum += 0.5, :X, 1
```

Supported spin operators include:

| Symbol | Meaning |
| --- | --- |
| `:X`, `:Z`, `:iY` | single-site Pauli operators |
| `:σp`, `:σm` | spin raising and lowering operators |
| `:Pup`, `:Pdn` | local projectors |
| `:CX`, `:CZ` | two-site controlled operators |

### Spectra

```julia
evals = spectrum(opsum, basis)

evals, evecs = spectrum(opsum, basis; retvecs=true)

evals_full = spectrum(opsum, L)
```

### Time Evolution

```julia
timeEvolve(opsum, init, ts, obs, exact())          # exact diagonalization
timeEvolve(opsum, init, ts, obs, rk4())            # RK4 ODE solver
timeEvolve(opsum, init, ts, obs, spmat(; order=6)) # sparse matrix method
```

All methods return the final state. Observers store measurements in their
`.data` field.

### Observables

```julia
obs_z = ZObserver(1, basis)
obs_x = XObserver(2, basis)
obs_energy = OpSumObserver(opsum, basis)
obs_corr = OperatorObserver((1.0, :Z, 1, :Z, 3), basis)
```

Expectation values and operator actions can also be computed directly:

```julia
E = expected(opsum, final_state)

ops = [Op(:Z, 1), Op(:Z, 3)]
corr = expected(ops, final_state)

new_state = apply([Op(:X, 1)], final_state)
```

### Entanglement

```julia
S = ent_entropy(final_state)             # default bipartition at L ÷ 2
S_cut = ent_entropy(final_state, 3)       # cut after 3 sites

ρ_A = reduced_density_matrix(final_state; subsys='A')
ρ_B = reduced_density_matrix(final_state; subsys='B')
```

## API Reference

### State Representation

- `QState`: quantum state with a basis and coefficient vector
- `SpinBasis(lsize; a, num, kint)`: spin-1/2 basis constructor
- `findindex(basis, bits)`: find a product-state index in a basis
- `ProductState(basis, args...)` / `product_state(basis, args...)`: computational basis states
- `RandomState(basis)` / `random_state(basis)`: random normalized states

### Operators

- `Op(name, loc)`: construct a local operator
- `OpSum{T, O}`: linear combination of operator sequences
- `matrixform(ops, basis[, coeff]; sparsed=true, dtype=Float64)`: matrix representation of an operator sequence
- `makeHamiltonian(opsum, basis; sparsed=false, dtype=ComplexF64)`: matrix representation of a Hamiltonian

### Dynamics and Measurements

- `spectrum(opsum, basis; retvecs=false)`: eigenspectrum in a basis
- `spectrum(opsum, lsize)`: full spectrum over particle-number sectors
- `timeEvolve(opsum, init, ts, obs, alg=exact())`: time evolution with observable recording
- `record!(observer, state, step)`: record one observable value
- `expected(ops, psi)` / `expected(opsum, psi)`: expectation values
- `dot(x, ops, y)`: operator-valued inner product
- `apply(ops, psi)` / `apply!(ops, psi)`: apply operators to a state
- `act(op, bits)` / `act_seq(coeff, ops_list, bits)`: low-level bitstring operator actions

### Observers

- `ZObserver(loc, basis)`: track `<Z_i>`
- `XObserver(loc, basis)`: track `<X_i>`
- `OpSumObserver(opsum, basis)`: track the expectation value of an `OpSum`
- `OperatorObserver(os, basis)`: track a custom operator sequence

### Configuration

- `set_systype(type)`: set the active system type
- `get_systype()`: query the active system type

Currently, `:Spin` is the supported system type. `:Fermion` is under development.

## Examples

Complete examples are available in [`examples/`](examples):

- [`examples/xxzmodel.jl`](examples/xxzmodel.jl): XXZ chain time evolution in a fixed particle-number sector
- [`examples/tfimodel.jl`](examples/tfimodel.jl): transverse-field Ising model time evolution in the full basis

## Project Layout

```text
src/
  ├── ExactDiagonalize.jl   # main module and public exports
  ├── state_basis.jl        # state and basis definitions
  ├── operators.jl          # local operators and Hamiltonian construction
  ├── observers.jl          # observable recording
  ├── exactdiag.jl          # exact diagonalization time evolution
  ├── ode_solver.jl         # RK4 time evolution
  ├── sparsemat.jl          # sparse matrix time evolution
  ├── entanglement.jl       # entanglement entropy and reduced density matrices
  └── utils.jl              # bitstring utilities
```

## Acknowledgement

This package grew out of homework for the course *Nonequilibrium Dynamics in
Closed Quantum System*, taught by HongZheng Zhao in the spring semester of 2026
at the School of Physics, Peking University.

The design of `OpSum` and `AbstractObserver` is inspired by
[ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl), developed by Matthew
Fishman (@mtfishman) and contributors.

## Contributing

Issues and pull requests are welcome.

## License

See [`LICENSE`](LICENSE) for details.

## Citation

If you use ExactDiagonalize.jl in research or coursework, please cite this
repository.
