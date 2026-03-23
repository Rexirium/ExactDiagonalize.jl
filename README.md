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
Pkg.add(url="https://github.com/Rexirium/ExactDiagonalize")
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
init = NumState("1000000000")

# Build XY Hamiltonian
opsum = OpSum(Float64)
for j in 1:L
    nj = mod1(j + 1, L) # PBC
    opsum += Δ, :Z, j, :Z, nj
    opsum += 1.0, :X, j, :X, nj
    opsum += -1.0, :iY, j, :iY, nj
end

# Define observable and time points
obs = OperatorObserver((1.0, :Z, L), init.basis)
ts = 0.0:0.05:10.0

# Run time evolution
timeEvolve(opsum, init, ts, obs)

@show obs.data # the data recorded
```

## Core API

### State Representation

- **`FullState`**: Quantum state defined on full dimension of Hilbert space
- **`NumState`**: Particle number or total spin conserved state (use it when system has $U(1)$ symmetry)
- **`FullBasis`**: Basis defined on full dimension of Hilbert space (avoid using it when possible)
- **`NumBasis`**: Basis constrained with fixed particle number occupation or total spin

### Operators

- **`SpinOp`**: Individual spin operators ($X$, $Y$, $Z$, $\sigma^+$, $\sigma^-$, $iY$, $CX$, $CZ$)
- **`FermionOp`**: Wait for later development
- **`Operator`**: Multi-site operator products, such as $aX_i X_{i+1}$ ,  $b CX_{1,2} Z_3$
- **`OpSum`**: Linear combinations of operators (Hamiltonian)

### Functions

- **`spectrum(opsum, basis; retvecs)`**: Compute eigenvalues (and eigenvectors if retvecs is `true`)
- **`timeEvolve(opsum, init_state, basis, tf)`**: Exact diagonalization time evolution to the final time `tf`
- **`timeEvolve(opsum, init_state, times, ts, observer, alg)`**: Time evolution performed by chosen algorithm `alg`. For now,  `alg`: can take:
  - `exact()` for exact diagonalization results ,
  -  `rk4()` for ODE solver using 4th order Runge-Kutta algorithm, 
  - `spmat()` for sparse matrix multiplication
- **`record!(observer, state, step)`**: Record observable at time step `step`
- **`makeHamiltonian(opsum, basis; sparsed)`**: Convert OpSum to sparse matrix Hamiltonian if `sparsed = true`
  - **Caution!**: For now, the exact diagonalization algorithm only take Hamiltonian as a dense matrix for `eigen` in `LinearAlgebra.jl` do NOT support sparse matrix from `SparseArrays.jl`
- **`expected(ops, psi)`**: compute the expected value of `ops`, an `OpSum` or an array of `Op`s, w.r.t the state `psi`
- **`apply[!](ops, psi)`**: apply `ops`, an `OpSum` or an array of `Op`s to the state `psi`, return the result state. (`!` means inplace version to save memory)
- **`inner(x, ops, y)`**: compute the inner product $\langle x | O|y\rangle$, $O$ can be an `OpSum` or an array of `Op`s

### System Configuration

- **`set_systype(type)`**: Set system type (`:Spin` or `:Fermion`) (`:Fermion` type is yet to be developed)
- **`get_systype()`**: Query current system type

## Examples

See the `examples/` directory for complete working examples:

- `xymodel.jl`: XY spin chain with time evolution and observable tracking

## Key Features

- **Efficient Sparse Representation**: Leverages sparse matrix formats for memory efficiency
- **Flexible Operator Syntax**: Intuitive specification of Hamiltonians. Just Build it from what you see on papers
- **Observable Recording**: Built-in framework for tracking time-dependent measurements. Highly **flexible, customizable and expandable**
- **Hardware Acceleration**: Optional MKL support for accelerated linear algebra

## Project Structure

```
src/
  ├── ExactDiagonalize.jl   # Main module
  ├── exactdiag.jl          # Diagonalization core functions
  ├── operators.jl          # Operator and Hamiltonian construction
  ├── ode_solver.jl         # Time evolution by ODE (RK4)
  ├── sparsemat.jl          # Sparse matrix multiplication method
  ├── state_basis.jl        # State and basis definitions
  └── utils.jl              # Helper utilities for binary operations
```

## Acknowlegement

- The motivation to develop this package comes from the homework of the course *Nonequilibrium Dynamics in Closed Quantum System* taught by HongZheng Zhao in the spring semester of 2026 at the School of Physics, Peking University.
- The ideas of `OpSum` and `AbstractObserver` are inspired by the [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl) project by Matt Fishman (@mtfishman) and other developers.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

See LICENSE file for details.

## Citation

If you use ExactDiagonalize in your research, please cite this repository.
