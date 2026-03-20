using Test

if !isdefined(Main, :ExactDiagonalize)
    include("../src/ExactDiagonalize.jl")
    using .ExactDiagonalize
end

@testset "ExactDiagonalize.jl Tests" begin

    @testset "Basic Math Example" begin
        # Example: Replace these with your actual package functions
        # @test MyPackageName.process_data() == expected_result
        
        @test 1 + 1 == 2
        @test 2 * 3 == 6
    end

    @testset "Error Handling" begin
        # You can also test if your code correctly throws errors
        @test_throws DivideError 1 / 0
    end

end