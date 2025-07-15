using MX2RKKY
using MX2RKKY.utility: range_of_nonmissing
using MX2RKKY.lattice: LatCoords, l2p, neighbor_shifts, periodize, neighbors
using MX2RKKY.montecarlo
using MX2RKKY.observables

using Test
using LinearAlgebra: dot, norm
using Random: MersenneTwister
using OffsetArrays: no_offset_view
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
        @test norm.([l2p(s[1], s[2]) for s in neighbor_shifts(Val(1))]) ≈ ones(6)
        @test norm.([l2p(s[1], s[2]) for s in neighbor_shifts(Val(2))]) ≈ √3 * ones(6)
        @test norm.([l2p(s[1], s[2]) for s in neighbor_shifts(Val(3))]) ≈ 2.0 * ones(6)

        L1, L2 = 3, 9
        @test neighbors(LatCoords(0, 0), L1, L2) ==
              [LatCoords(1, 0), LatCoords(0, 8), LatCoords(2, 8),
               LatCoords(2, 0), LatCoords(0, 1), LatCoords(1, 1)]
    end

    @testset "Configuration" begin
        cm1 = Configuration(9, 15, -1)
        cp1 = Configuration(9, 15, 1)
        @test all(montecarlo.chirality(cm1, i) == -1 for i in CartesianIndices(cm1))
        @test all(montecarlo.chirality(cp1, i) == 1 for i in CartesianIndices(cp1))

        c_rnd = Configuration(9, 15, MersenneTwister(123456))
    end

    @testset "Hamiltonians" begin
        L1 = 9
        L2 = 15

        conf = Configuration(L1, L2, MersenneTwister(123456))

        V = 3.0
        λ = 0.1

        i = LatCoords(3, 2)

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

        β = 10.0
        J = 2.0
        V = 0.2
        λ = 0.1

        shift_p(i, Δi) = periodize(i + Δi, L1, L2)

        @testset "Ordered init_conf" begin
            rng = MersenneTwister(123456)
            for conf in (Configuration(L1, L2, 1), Configuration(L1, L2, -1))
                for R in [0, 1, 2, 3]
                    sim = MCSimulation(conf, rng, β=β, J=J, V=V, λ=λ, rkky_range_cutoff=R)

                    # Check that all spins coupled to the effective field give equal
                    # contributions to system's energy in the ordered states
                    energies = -(cos.(sim.conf.θ) .* sim.eff_field[1, :, :] .+
                                 sin.(sim.conf.θ) .* sim.eff_field[2, :, :])
                    @test no_offset_view(energies) ≈ fill(energies[0, 0], L1, L2)
                end
            end
        end

        function energy_ref(θ, sim)
            # The 0.5 accounts for each link between nodes being visited twice
            E = 0.5 * sum(J * cos(θ[i] - θ[j])
                 for i in CartesianIndices(sim.conf),
                     j in CartesianIndices(sim.conf)
                 if j ∈ montecarlo.neighbors(i, L1, L2)
            )
            if !isnothing(sim.V)
                if sim.rkky_range_cutoff == 0
                    E += sum(sim.V[Δi] *
                             cos(θ[i] - θ[shift_p(i, Δi)] - sim.ϕ[Δi])
                             for i in CartesianIndices(sim.conf),
                                Δi in CartesianIndices(sim.V)
                             if Δi != CartesianIndex(0, 0))
                else
                    for r in 1:sim.rkky_range_cutoff
                        R = Val(r)
                        E += sum(sim.V[Δi] *
                                 cos(θ[i] - θ[shift_p(i, Δi)] - sim.ϕ[Δi])
                                 for i in CartesianIndices(sim.conf),
                                    Δi in neighbor_shifts(R))
                    end
                end
            end
            return E
        end

        @testset "Random init_conf" begin
            rng = MersenneTwister(123456)

            for R in [0, 1, 2, 3]
                conf = Configuration(L1, L2, rng)

                sim = MCSimulation(conf, rng, β=β, J=J, V=V, λ=λ, rkky_range_cutoff=R)
                @test sim.energy ≈ energy_ref(sim.conf.θ, sim)

                acceptance_rate = run!(sim, n_mcs=1000, n_warmup_mcs=200,
                                       or_sweeps_per_mcs=1)
                @show acceptance_rate
                @test sim.energy ≈ energy_ref(sim.conf.θ, sim)

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
    end

end
