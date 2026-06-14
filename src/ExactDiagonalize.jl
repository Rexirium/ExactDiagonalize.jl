module ExactDiagonalize

using MKL, LinearAlgebra
using SparseArrays: spzeros
using SparseArrays: SparseMatrixCSC as SpMatrix

include("utils.jl")
include("state_basis.jl")
include("entanglement.jl")
include("operators.jl")
include("observers.jl")
include("exactdiag.jl")
include("ode_solver.jl")
include("sparsemat.jl")

# export public API
export AbstractBasis, SpinBasis, QState, product_state, ProductState, random_state, RandomState, ent_entropy
export set_systype, get_systype, AbstractOp, SpinOp, Op, OpSum, matrixform, makeHamiltonian
export act, apply, apply!, expected
export AbstractObserver, OperatorObserver, OpSumObserver, ZObserver, XObserver, record!
export spectrum, exact, rk4, spmat, timeEvolve

end # ExactDiagonalize