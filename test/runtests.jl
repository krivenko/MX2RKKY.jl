using MX2RKKY
using MX2RKKY.utility: range_of_nonmissing
using MX2RKKY.lattice: l2c, neighbor_shifts, neighbors
using MX2RKKY.montecarlo
using MX2RKKY.observables

using Test
using LinearAlgebra: dot, norm
using Random: MersenneTwister
using Aqua

@testset "MX2RKKY.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(MX2RKKY)
    end

    @testset "Utility" begin
        a = [1, 2, missing, 4, missing, missing, 7, 8, 9, missing]
        @test range_of_nonmissing(a) == [1, 2, 4, 7, 8, 9]
        @test range_of_nonmissing(a, 2:5) == [2, 4, 7, 8]

        aa = [1  2  missing 4       missing missing 7  8  9       missing
              10 11 12      missing missing 13      14 15 missing 16]
        @test range_of_nonmissing(aa) == [1 2 7 8; 10 11 14 15]
        @test range_of_nonmissing(aa, 2:3) == [2 7; 11 14]
    end

    @testset "Lattice" begin
        @test norm.([l2c(s...) for s in neighbor_shifts(Val(1))]) ≈ ones(6)
        @test norm.([l2c(s...) for s in neighbor_shifts(Val(2))]) ≈ √3 * ones(6)
        @test norm.([l2c(s...) for s in neighbor_shifts(Val(3))]) ≈ 2.0 * ones(6)

        L1, L2 = 3, 6
        @test neighbors((0, 0), L1, L2) ==
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

        V = 3.0
        λ = 0.1

        i = (3, 2)

        @test montecarlo.rkky_int(i, V, λ) * cos(montecarlo.rkky_angle(i)) ≈
              1.821551601453564
        @test montecarlo.rkky_int(i, V, λ) * sin(montecarlo.rkky_angle(i)) ≈
              3.1550199223260273

        Varr = montecarlo.make_rkky_int_array(L1, L2, V, λ)
        ϕarr = montecarlo.make_rkky_angle_array(L1, L2)
    end

    @testset "MCSimulation" begin
        L1 = 9
        L2 = 15

        rng = MersenneTwister(123456)
        conf = Configuration(L1, L2, rng)

        β = 10.0
        J = 2.0
        V = 0.2
        λ = 0.1

        function energy_ref(θ)
            E = 0.5 * sum(J * cos(θ[i...] - θ[j...])
                 for i in eachindex(sim.conf), j in eachindex(sim.conf)
                 if (j ∈ montecarlo.neighbors(i, L1, L2))
            )
            if !isnothing(sim.V)
                E += sum(sim.V[(j .- i)...] * cos(θ[i...] - θ[j...] - sim.ϕ[(j .- i)...])
                         for i in eachindex(sim.conf), j in eachindex(sim.conf) if i != j)
            end
            return E
        end

        sim = MCSimulation(conf, rng, β=β, J=J, V=V, λ=λ)
        @test sim.energy ≈ energy_ref(sim.conf.θ)

        run!(sim, n_cycles=500, n_warmup_cycles=100, cycle_length=50)
        @test sim.energy ≈ energy_ref(sim.conf.θ)

        @show acceptance_rates(sim)
        @show min_energy(sim)
        @show avg_energy(sim)
        @show var_energy(sim)
        @show specific_heat(sim)
        @show avg_chirality(sim)
        @show var_chirality(sim)
        @show susc_chirality(sim)
        @show binder_chirality(sim)
        @show corr_length_chirality(sim)
        @show susc_spin_K(sim)
        @show binder_spin_K(sim)
        @show corr_length_spin_K(sim)
    end

end
