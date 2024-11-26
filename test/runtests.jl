using MX2RKKY
using Test
using Aqua

@testset "MX2RKKY.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(MX2RKKY)
    end
    # Write your tests here.
end
