using MX2RKKY
using Test
using Random: MersenneTwister
using Aqua

@testset "MX2RKKY.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(MX2RKKY)
    end

    @testset "Lattice" begin
        L1, L2 = 3, 4
        @test MX2RKKY.neighbors((1, 1), L1, L2) ==
              [(2, 1), (3, 1), (1, 2), (1, 4), (2, 2), (3, 4)]
    end

    @testset "Configuration" begin
        c1 = Configuration(3, 4)
        c2 = Configuration(3, 4, MersenneTwister(123456))
    end

    @testset "Hamiltonians" begin
        L1 = 10
        L2 = 15

        conf = Configuration(L1, L2, MersenneTwister(123456))

        MX2RKKY.xxz_energy(conf, Jpar=1.6, Jperp=1.5)

        Vpar = 3.0
        Vperp = 2.0
        λ = 0.1

        i = (3, 2)

        @test MX2RKKY.rkky_int(i, Vpar, Vperp, λ) ≈ [
            1.821551601453564 3.1550199223260273 0.;
           -3.1550199223260273 1.821551601453564 0.;
            0. 0. -2.428735468604752
        ]
        @test isapprox(MX2RKKY.renorm_rkky_int(i, Vpar, Vperp, λ,
                                               L1=100, L2=150, maxshells=100, atol=1e-4),
                       [1.8090998587819007	3.140442549655362	0.;
                        -3.140442549655362	1.8090998587819007	0.;
                        0.	0.	-2.4385083545467614],
                       atol=1e-3)

        V = MX2RKKY.make_rkky_int_tensor(L1, L2, Vpar, Vperp, λ, renormalize=false)
        MX2RKKY.rkky_energy_term(conf, i, V)
        MX2RKKY.rkky_energy(conf, V)
    end

end
