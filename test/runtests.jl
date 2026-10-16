using MX2RKKY
using MX2RKKY.utility: range_of_nonmissing
using MX2RKKY.lattice: LatCoords, l2p, neighbor_shifts, periodize, neighbors
using MX2RKKY.montecarlo
using MX2RKKY.observables

using Test
using LinearAlgebra: dot, norm
using Random: MersenneTwister
using OffsetArrays: no_offset_view

using MPI
using Aqua

MPI.Init()

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
        @test all(c_rnd.occupied)

        occ = random_occupation(9, 15, 20, MersenneTwister(123456))
        @test axes(occ) == axes(c_rnd.θ)
        @test count(.!occ) == 20
        @test occ == random_occupation(9, 15, 20, MersenneTwister(123456))

        c_vac = Configuration(9, 15, 1, occupied=occ)
        @test montecarlo.n_spins(c_vac) == 9 * 15 - 20
        @test montecarlo.n_complete_triangles(c_vac) ==
              count(all(occ[j] for j in montecarlo.triangle_nodes(i, 9, 15))
                    for i in CartesianIndices(c_vac))
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

        β = 0.3
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
            occ = sim.conf.occupied
            # The 0.5 accounts for each link between nodes being visited twice
            E = 0.5 * sum(J * cos(θ[i] - θ[j])
                 for i in CartesianIndices(sim.conf),
                     j in CartesianIndices(sim.conf)
                 if j ∈ montecarlo.neighbors(i, L1, L2) && occ[i] && occ[j]
            )
            if !isnothing(sim.V)
                if sim.rkky_range_cutoff == 0
                    E += sum(sim.V[Δi] *
                             cos(θ[i] - θ[shift_p(i, Δi)] - sim.ϕ[Δi])
                             for i in CartesianIndices(sim.conf),
                                Δi in CartesianIndices(sim.V)
                             if Δi != CartesianIndex(0, 0) &&
                                occ[i] && occ[shift_p(i, Δi)])
                else
                    for r in 1:sim.rkky_range_cutoff
                        R = Val(r)
                        E += sum(sim.V[Δi] *
                                 cos(θ[i] - θ[shift_p(i, Δi)] - sim.ϕ[Δi])
                                 for i in CartesianIndices(sim.conf),
                                    Δi in neighbor_shifts(R)
                                 if occ[i] && occ[shift_p(i, Δi)])
                    end
                end
            end
            return E
        end

        @testset "Random init_conf" begin
            rank = MPI.Comm_rank(MPI.COMM_WORLD)
            rng = MersenneTwister(123456 + rank)
            conf = Configuration(L1, L2, rng)

            for R in [0, 1, 2, 3]
                conf = Configuration(L1, L2, rng)

                sim = MCSimulation(conf, rng, β=β, J=J, V=V, λ=λ, rkky_range_cutoff=R)
                @test sim.energy ≈ energy_ref(sim.conf.θ, sim)

                acceptance_rate = run!(sim, n_mcs=1000, n_warmup_mcs=200,
                                       or_sweeps_per_mcs=1,
                                       show_progress = (rank == 0))
                @show acceptance_rate
                @test sim.energy ≈ energy_ref(sim.conf.θ, sim)

                if rank == 0
                    @show sim.measurements.count
                    @show min_energy(sim)
                    @show avg_energy(sim)
                    @show var_energy(sim)
                    @show err_energy(sim)
                    @show specific_heat(sim)
                    @show avg_chirality(sim)
                    @show var_chirality(sim)
                    @show err_chirality(sim)
                    @show susc_chirality(sim)
                    @show binder_chirality(sim)
                    @show corr_length_chirality(sim)
                    @show avg_spin_K(sim)
                    @show avg_norm_spin_K(sim)
                    @show avg_dir_spin_K(sim)
                    @show var_spin_K(sim)
                    @show err_spin_K(sim)
                    @show var_norm_spin_K(sim)
                    @show susc_spin_K(sim)
                    @show susc_norm_spin_K(sim)
                    @show binder_spin_K(sim)
                    @show corr_length_spin_K(sim)
                    @show corr_chirality_spin_K(sim)
                    @show corr_chirality_norm_spin_K(sim)
                    @show cov_chirality_spin_K(sim)
                    @show cov_chirality_norm_spin_K(sim)
                    @show cov_chirality_long_spin_K(sim)
                    @show cov_chirality_trans_spin_K(sim)
                end
            end
        end

        @testset "Vacancies" begin
            rank = MPI.Comm_rank(MPI.COMM_WORLD)

            # An explicit all-occupied mask is equivalent to the default one
            occ_full = random_occupation(L1, L2, 0, MersenneTwister(1))
            for R in [0, 1, 2, 3]
                conf1 = Configuration(L1, L2, MersenneTwister(123456 + rank))
                conf2 = Configuration(L1, L2, MersenneTwister(123456 + rank),
                                      occupied=occ_full)
                sim1 = MCSimulation(conf1, MersenneTwister(42 + rank),
                                    β=β, J=J, V=V, λ=λ, rkky_range_cutoff=R)
                sim2 = MCSimulation(conf2, MersenneTwister(42 + rank),
                                    β=β, J=J, V=V, λ=λ, rkky_range_cutoff=R)
                @test sim1.energy == sim2.energy
                @test sim1.eff_field == sim2.eff_field
                run!(sim1, n_mcs=100, n_warmup_mcs=20, or_sweeps_per_mcs=1,
                     show_progress=false)
                run!(sim2, n_mcs=100, n_warmup_mcs=20, or_sweeps_per_mcs=1,
                     show_progress=false)
                @test sim1.conf.θ == sim2.conf.θ
                @test sim1.energy == sim2.energy
            end

            # The same vacancy pattern on all ranks
            occ = random_occupation(L1, L2, 20, MersenneTwister(654321))

            # Energy bookkeeping and frozen vacant sites
            rng = MersenneTwister(123456 + rank)
            for R in [0, 1, 2, 3]
                conf = Configuration(L1, L2, rng, occupied=occ)
                θ_init = copy(conf.θ)
                sim = MCSimulation(conf, rng, β=β, J=J, V=V, λ=λ, rkky_range_cutoff=R)
                @test sim.energy ≈ energy_ref(sim.conf.θ, sim)
                run!(sim, n_mcs=1000, n_warmup_mcs=200, or_sweeps_per_mcs=1,
                     show_progress=false)
                @test sim.energy ≈ energy_ref(sim.conf.θ, sim)
                @test sim.conf.θ[.!occ] == θ_init[.!occ]
            end

            # Normalization of measured chirality and magnetization
            K = (4π / 3, 0)
            for χ0 in (-1, 1)
                conf = Configuration(L1, L2, χ0, occupied=occ)
                sim = MCSimulation(conf, MersenneTwister(1), β=β, J=J, V=V, λ=λ)
                montecarlo.measure!(sim)
                @test sim.measurements.chirality_sums[1] ≈ χ0

                magn_at_K_ref = zeros(ComplexF64, 2)
                for i in CartesianIndices(conf)
                    occ[i] || continue
                    w = exp(-im * dot(K, l2p(i[1], i[2])))
                    magn_at_K_ref .+= w .* (cos(conf[i]), sin(conf[i]))
                end
                magn_at_K_ref ./= count(occ)
                @test sim.measurements.magnetization_at_K_sum ≈ magn_at_K_ref
            end
        end
    end

end
