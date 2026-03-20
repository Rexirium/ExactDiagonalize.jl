using Test

if !isdefined(Main, :ExactDiagonalize)
    include("../src/ExactDiagonalize.jl")
    using .ExactDiagonalize
end

