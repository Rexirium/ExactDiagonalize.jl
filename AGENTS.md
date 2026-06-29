# Repository Guidelines

## Project Structure & Module Organization

`src/ExactDiagonalize.jl` defines the package module, includes implementation
files, and exports the public API. Keep features grouped by responsibility:
bases and states in `src/state_basis.jl`, operators in `src/operators.jl`,
diagonalization and evolution in `src/exactdiag.jl`, `ode_solver.jl`, and
`sparsemat.jl`, and measurements in `observers.jl` and `entanglement.jl`.
Optional XDiag integration belongs in `ext/XDiagExt.jl`.

Tests live in `test/runtests.jl`. Executable demonstrations are under
`examples/`. The `manybodyscars/` and `homeworks/` directories contain research
scripts and course material, including Python utilities; they are not part of
the package's exported Julia API.

## Build, Test, and Development Commands

- `julia --project=. -e 'using Pkg; Pkg.instantiate()'` installs dependencies
  recorded in `Project.toml` and `Manifest.toml`.
- `julia --project=. -e 'using Pkg; Pkg.test()'` runs the complete test suite.
- `julia --project=. examples/tfimodel.jl` runs the transverse-field Ising
  example; substitute `examples/xxzmodel.jl` for the XXZ workflow.
- `python -m pip install -r requirements.txt` installs dependencies needed by
  the auxiliary Python scripts.

Use Julia 1.12 or newer, as documented in the README and compatibility entries.

## Coding Style & Naming Conventions

Follow the existing Julia style: four-space indentation, lowercase
`snake_case` for functions and variables, and `UpperCamelCase` for types such
as `SpinBasis` and `OperatorObserver`. Use `!` for mutating functions
(`apply!`, `record!`). Add public exports centrally in
`src/ExactDiagonalize.jl`; keep internal helpers unexported. Prefer multiple
dispatch and concrete, meaningful method signatures over manual type
branching. No formatter or linter is configured, so match nearby code and keep
changes focused.

## Testing Guidelines

Tests use Julia's standard `Test` framework. Add focused `@testset` blocks with
descriptive names to `test/runtests.jl`, and use `TEST_ATOL` for approximate
floating-point comparisons. Cover normal behavior, symmetry-restricted bases,
and relevant type variants. Run `Pkg.test()` before submitting changes.

## Commit & Pull Request Guidelines

Recent commits use concise subjects such as `matrix-free apply and apply!`.
Write short, present-tense summaries that name the affected behavior; avoid
mixing unrelated changes. Pull requests should explain the motivation and API
impact, list verification commands, and link related issues. Update tests and
README examples when public behavior changes. Attach plots or screenshots only
when output is visual.
