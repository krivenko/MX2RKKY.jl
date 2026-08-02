"""
Run a temperature scan for a system with both the nearest-neighbour
exchange coupling and the long-range RKKY interaction.
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
    "--V"
        help = "RKKY interaction strength"
        arg_type = Float64
        required = true
    "--lambda"
        help = "RKKY interaction radius"
        arg_type = Float64
        required = true
    "--rkky_range_cutoff"
        help = "RKKY interaction cut-off"
        arg_type = Int
        default = 0
    "--init_chi"
        help = "chirality of the initial configuration, either 1 or -1"
        arg_type = Int
        default = 1
    "--seed"
        help = "basic PRNG seed"
        arg_type = Int
        default = 128512
end

parsed_args = parse_args(ARGS, s)
L = parsed_args["L"]
V = parsed_args["V"]
λ = parsed_args["lambda"]
rkky_range_cutoff = parsed_args["rkky_range_cutoff"]
init_chi = parsed_args["init_chi"]
seed = parsed_args["seed"]

## MPI

MPI.Init()
world_size = MPI.Comm_size(MPI.COMM_WORLD)
rank = MPI.Comm_rank(MPI.COMM_WORLD)
if rank == 0
    println("Running simulation with MPI world size $(world_size)")
    @show L
    @show V
    @show λ
    @show rkky_range_cutoff
    @show init_chi
    @show seed
end

## Parameters

function make_T_range(fine_window, dT_coarse, dT_fine)
    return vcat(collect(dT_coarse:dT_coarse:fine_window[1]),
                collect(fine_window[1]+dT_fine:dT_fine:fine_window[2]),
                collect(fine_window[2]+dT_coarse:dT_coarse:2.0))
end

# Temperature range for rkky_range_cutoff=0
T_range = make_T_range((0.5, 1.3), 0.1, 0.025)

#dT1 = 0.05
#dT2 = 0.005
#
# Temperature ranges for rkky_range_cutoff=1
#T_range = Dict(
#    0.001 => make_T_range((0.45, 0.75), dT1, dT2),
#    0.005 => make_T_range((0.5, 0.8), dT1, dT2),
#    0.01 => make_T_range((0.55, 0.85), dT1, dT2),
#    0.05 => make_T_range((0.8, 1.3), dT1, dT2),
#    0.1 => make_T_range((1.2, 1.7), dT1, dT2)
#)[V]

if rank == 0
    println("T_range=", T_range)
    @show length(T_range)
end

J = 1.0

maxΔθ = Float64(π/4)

# #MCS for rkky_range_cutoff=0
n_warmup_mcs = 100
n_mcs = 1_000

# #MCS for rkky_range_cutoff=1
#n_warmup_mcs = 1_000
#n_mcs = 10_000

m_sweeps_per_mcs = 1
or_sweeps_per_mcs = 1

outfile = "RKKY_T_scan_V$(V)_lambda$(λ)_R$(rkky_range_cutoff)_L$(L)_init_chi$(init_chi).h5"

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
    end
end

rng = MersenneTwister(seed + 1000 * rank)
conf = Configuration(L, L, init_chi)

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
            out["final_conf"] = sim.conf.θ
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
