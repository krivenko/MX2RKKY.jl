using MX2RKKY
using MX2RKKY.utility: range_of_nonmissing
using MX2RKKY.montecarlo

using Test
using LinearAlgebra: dot
using Random: MersenneTwister
using Aqua

@testset "MX2RKKY.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(MX2RKKY)
    end

    @testset "Utility" begin
        a = [1, 2, missing, 4, missing, missing, 7, 8, 9, missing]
        @test range_of_nonmissing(a, 2:5) == [2, 4, 7, 8]
        @test range_of_nonmissing(a) == [1, 2, 4, 7, 8, 9]
    end

    @testset "Lattice" begin
        L1, L2 = 3, 6
        @test montecarlo.neighbors((0, 0), L1, L2) ==
              [(1, 0), (0, 5), (2, 5), (2, 0), (0, 1), (1, 1)]
    end

    @testset "Configuration" begin
        cm1 = Configuration(9, 12, -1)
        cp1 = Configuration(9, 12, 1)
        @test all(montecarlo.chirality(cm1, i) == -1 for i in eachindex(cm1))
        @test all(montecarlo.chirality(cp1, i) == 1 for i in eachindex(cp1))

        c_rnd = Configuration(9, 12, MersenneTwister(123456))
    end

    @testset "Hamiltonians" begin
        L1 = 9
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
                                                  L1=96, L2=150, maxshells=100, atol=1e-4),
                       [1.8288029675935291	3.167579656904715	0.;
                        -3.167579656904715	1.8288029675935291	0.;
                        0.	0.	-2.4384039567913622],
                       atol=1e-3)

        V = montecarlo.make_rkky_int_tensor(L1, L2, Vpar, Vperp, λ, renormalize=false)
    end

    @testset "MCSimulation" begin
        L1 = 9
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
            if !isnothing(sim.V)
                E += sum(dot(S_vec[:, i...], sim.V[:, :, (j .- i)...], S_vec[:, j...])
                         for i in eachindex(sim.conf), j in eachindex(sim.conf) if i != j)
            end
            return E
        end

        sim = MCSimulation(conf, rng, β=β, Jpar=Jpar, Jperp=Jperp,
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
        @show binder_chirality(sim)
        @show corr_length_chirality(sim)
        @show specific_heat(sim)
    end

end
