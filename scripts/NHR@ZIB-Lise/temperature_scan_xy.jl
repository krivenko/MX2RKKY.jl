"""
Run a temperature scan for a system with only the nearest-neighbour
exchange coupling (XY-model).
"""

using MX2RKKY.montecarlo
using MX2RKKY.lattice
using MX2RKKY.observables

using Random: MersenneTwister
using HDF5
using MPI
using ArgParse

## Command line

s = ArgParseSettings()
@add_arg_table! s begin
    "--L"
        help = "linear size of the cluster"
        arg_type = Int
        required = true
    "--init_chi"
        help = "chirality of the initial configuration, either 1 or -1"
        arg_type = Int
        default = 1
    "--seed"
        help = "basic PRNG seed"
        arg_type = Int
        default = 128512
    "--n_vacancies"
        help = "number of vacant sites"
        arg_type = Int
        default = 0
    "--disorder_seed"
        help = "PRNG seed used to generate the vacancy pattern"
        arg_type = Int
        default = 0
end

parsed_args = parse_args(ARGS, s)
L = parsed_args["L"]
init_chi = parsed_args["init_chi"]
seed = parsed_args["seed"]
n_vacancies = parsed_args["n_vacancies"]
disorder_seed = parsed_args["disorder_seed"]

## MPI

MPI.Init()
world_size = MPI.Comm_size(MPI.COMM_WORLD)
rank = MPI.Comm_rank(MPI.COMM_WORLD)
if rank == 0
    println("Running simulation with MPI world size $(world_size)")
    @show L
    @show init_chi
    @show seed
    @show n_vacancies
    @show disorder_seed
end

## Parameters

function make_T_range(fine_window, dT_coarse, dT_fine)
    return vcat(collect(dT_coarse:dT_coarse:fine_window[1]),
                collect(fine_window[1]+dT_fine:dT_fine:fine_window[2]),
                collect(fine_window[2]+dT_coarse:dT_coarse:2.0))
end

dT1 = 0.05
dT2 = 0.005
T_range = make_T_range((0.4, 0.7), dT1, dT2)

if rank == 0
    println("T_range=", T_range)
    @show length(T_range)
end

J = 1.0
V = 0.0
λ = 0.1
rkky_range_cutoff = 0

maxΔθ = Float64(π/4)
n_warmup_mcs = 2_000
n_mcs = 20_000
m_sweeps_per_mcs = 1
or_sweeps_per_mcs = 1

outfile = "XY_T_scan_L$(L)_init_chi$(init_chi)_nvac$(n_vacancies)_dseed$(disorder_seed).h5"

# Vacancy pattern (identical on all MPI ranks)
occupied = random_occupation(L, L, n_vacancies, MersenneTwister(disorder_seed))

# Save parameters
if rank == 0
    h5open(outfile, "w") do ar
        # Parameters of the run
        params = create_group(ar, "params")
        params["L"] = L
        params["T_range"] = T_range
        params["J"] = J
        params["V"] = V
        params["lambda"] = λ
        params["rkky_range_cutoff"] = rkky_range_cutoff
        params["init_chi"] = init_chi
        params["max_dtheta"] = maxΔθ
        params["n_warmup_mcs"] = n_warmup_mcs
        params["n_mcs"] = n_mcs
        params["m_sweeps_per_mcs"] = m_sweeps_per_mcs
        params["or_sweeps_per_mcs"] = or_sweeps_per_mcs
        params["seed"] = seed
        params["n_vacancies"] = n_vacancies
        params["disorder_seed"] = disorder_seed
        params["occupied"] = UInt8.(parent(occupied))
    end
end

rng = MersenneTwister(seed + 1000 * rank)
conf = Configuration(L, L, init_chi, occupied=occupied)

total_time = 0.0
for (n, T) in enumerate(T_range)
    if rank == 0
        println("T = $(T)")
    end
    sim = MCSimulation(conf, rng, β=1/T, J=J, V=V, λ=λ,
                       rkky_range_cutoff=rkky_range_cutoff)
    run_stats = @timed run!(sim,
                            n_mcs=n_mcs,
                            n_warmup_mcs=n_warmup_mcs,
                            m_sweeps_per_mcs=m_sweeps_per_mcs,
                            or_sweeps_per_mcs=or_sweeps_per_mcs,
                            maxΔθ=maxΔθ,
                            show_progress = (world_size == 0))
    if rank == 0
        println("run_stats = ", run_stats)
        println("acceptance_rate = ", run_stats.value)

        global total_time += run_stats.time
        est_time = (total_time / n) * (length(T_range) - n)
        println("Estimated remaining time = ", est_time)
    end

    # Save results
    if rank == 0
        h5open(outfile, "cw") do ar
            out = create_group(ar, "T$(T)")
            out["final_conf"] = parent(sim.conf.θ)
            # Measurements
            for measure in fieldnames(typeof(sim.measurements))
                name = string(measure)
                startswith(name, "fourier") && continue
                out[name] = getfield(sim.measurements, measure)
            end
            # Observables
            out["min_energy"] = min_energy(sim)
            out["avg_energy"] = avg_energy(sim)
            out["var_energy"] = var_energy(sim)
            out["err_energy"] = err_energy(sim)
            out["specific_heat"] = specific_heat(sim)
            out["avg_chirality"] = avg_chirality(sim)
            out["var_chirality"] = var_chirality(sim)
            out["err_chirality"] = err_chirality(sim)
            out["susc_chirality"] = susc_chirality(sim)
            out["binder_chirality"] = binder_chirality(sim)
            out["corr_length_chirality"] = corr_length_chirality(sim)
            out["avg_spin_K"] = avg_spin_K(sim)
            out["var_spin_K"] = var_spin_K(sim)
            out["err_spin_K"] = err_spin_K(sim)
            out["avg_norm_spin_K"] = avg_norm_spin_K(sim)
            out["avg_dir_spin_K"] = avg_dir_spin_K(sim)
            out["var_norm_spin_K"] = var_norm_spin_K(sim)
            out["susc_spin_K"] = susc_spin_K(sim)
            out["susc_norm_spin_K"] = susc_norm_spin_K(sim)
            out["binder_spin_K"] = binder_spin_K(sim)
            out["corr_length_spin_K"] = corr_length_spin_K(sim)
            out["corr_chirality_spin_K"] = corr_chirality_spin_K(sim)
            out["corr_chirality_norm_spin_K"] = corr_chirality_norm_spin_K(sim)
            out["cov_chirality_spin_K"] = cov_chirality_spin_K(sim)
            out["cov_chirality_norm_spin_K"] = cov_chirality_norm_spin_K(sim)
            out["cov_chirality_long_spin_K"] = cov_chirality_long_spin_K(sim)
            out["cov_chirality_trans_spin_K"] = cov_chirality_trans_spin_K(sim)
        end
    end

    global conf = deepcopy(sim.conf)
end
