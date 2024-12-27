using MX2RKKY
using MX2RKKY.montecarlo

using Test
using LinearAlgebra: dot
using Random: MersenneTwister
using Aqua

@testset "MX2RKKY.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(MX2RKKY)
    end

    @testset "Lattice" begin
        L1, L2 = 3, 4
        @test montecarlo.neighbors((0, 0), L1, L2) ==
              [(1, 0), (0, 3), (2, 3), (2, 0), (0, 1), (1, 1)]
    end

    @testset "Configuration" begin
        c1 = Configuration(3, 4)
        c2 = Configuration(3, 4, MersenneTwister(123456))
    end

    @testset "Hamiltonians" begin
        L1 = 10
        L2 = 15

        conf = Configuration(L1, L2, MersenneTwister(123456))

        Vpar = 3.0
        Vperp = 2.0
        λ = 0.1

        i = (3, 2)

        @test montecarlo.rkky_int(i, Vpar, Vperp, λ) ≈ [
            1.821551601453564 3.1550199223260273 0.;
           -3.1550199223260273 1.821551601453564 0.;
            0. 0. -2.428735468604752
        ]
        @test isapprox(montecarlo.renorm_rkky_int(i, Vpar, Vperp, λ,
                                               L1=100, L2=150, maxshells=100, atol=1e-4),
                       [1.8090998587819007	3.140442549655362	0.;
                        -3.140442549655362	1.8090998587819007	0.;
                        0.	0.	-2.4385083545467614],
                       atol=1e-3)

        V = montecarlo.make_rkky_int_tensor(L1, L2, Vpar, Vperp, λ, renormalize=false)
    end

    @testset "MCSimulation" begin
        L1 = 10
        L2 = 15

        rng = MersenneTwister(123456)
        conf = Configuration(L1, L2, rng, inplane=true)

        β = 10.0
        Jpar = 2.0
        Jperp = 0.5
        Vpar = 0.2
        Vperp = 0.1
        λ = 0.1

        function energy_ref(S_vec)
            E = 0.5 * sum(
                 Jpar * (S_vec[1, i...] * S_vec[1, j...] + S_vec[2, i...] * S_vec[2, j...])
                 + Jperp * S_vec[3, i...] * S_vec[3, j...]
                 for i in eachindex(sim.conf), j in eachindex(sim.conf)
                 if (j ∈ montecarlo.neighbors(i, L1, L2))
            )
            E += sum(dot(S_vec[:, i...], sim.V[:, :, (j .- i)...], S_vec[:, j...])
                     for i in eachindex(sim.conf), j in eachindex(sim.conf) if i != j)
            return E
        end

        sim = MCSimulation(conf, β=β, Jpar=Jpar, Jperp=Jperp,
                           Vpar=Vpar, Vperp=Vperp, λ=λ, Vrenormalize=false)
        @test sim.energy ≈ energy_ref(sim.conf.S_vec)

        run!(sim,
             n_cycles=500,
             n_warmup_cycles=100,
             cycle_length=50,
             inplane=true,
             update_triangle_prob=0.2)
        @test sim.energy ≈ energy_ref(sim.conf.S_vec)

        @show acceptance_rates(sim)
        @show min_energy(sim)
        @show avg_energy(sim)
        @show var_energy(sim)
        @show avg_chirality(sim)
        @show var_chirality(sim)
    end

end
