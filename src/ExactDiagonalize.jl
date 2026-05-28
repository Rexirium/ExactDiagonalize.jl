module ExactDiagonalize

using MKL, LinearAlgebra
using SparseArrays: spzeros
using SparseArrays: SparseMatrixCSC as SpMatrix

include("operators.jl")
include("observers.jl")
include("exactdiag.jl")
include("ode_solver.jl")
include("sparsemat.jl")

# export public API
export AbstractBasis, SpinBasis, QState, statevec
export set_systype, get_systype, AbstractOp, get_optype, SpinOp, Op, OpSum, makeHamiltonian
export act, apply, apply!, expected
export AbstractObserver, OperatorObserver, OpSumObserver, ZObserver, XObserver, record!
export spectrum, exact, rk4, spmat, timeEvolve

end # ExactDiagonalize