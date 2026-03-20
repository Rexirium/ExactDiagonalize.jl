module ExactDiagonalize

include("operators.jl")
include("exactdiag.jl")
include("ode_solver.jl")
include("sparsemat.jl")
# export public API
export NumBasis, TotalBasis, NumState, TotalState, State, AbstractState
export set_systype, get_systype, AbstractOp, SpinOp, Operator, OpSum, makeHamiltonian
export AbstractObserver, OperatorObserver, OpSumObserver, record!
export spectrum, exact, rk4, spmat, timeEvolve

end