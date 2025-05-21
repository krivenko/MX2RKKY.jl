module observables

using LinearAlgebra: norm
using Statistics: mean, std

using MX2RKKY.utility: range_of_nonmissing
using MX2RKKY.montecarlo: MCSimulation

export min_energy, avg_energy, var_energy, specific_heat
export avg_chirality, var_chirality, susc_chirality, binder_chirality, corr_length_chirality
export susc_spin_K, binder_spin_K, corr_length_spin_K

const OptionalAbstractRange = Union{AbstractRange, Nothing}

#
# Extract observables from sample arrays
#

"Extract the lowest energy from an array of energy samples."
function min_energy(energy_samples::AbstractVector,
                    sample_range::OptionalAbstractRange = nothing)
    return minimum(range_of_nonmissing(energy_samples, sample_range))
end

"Extract average energy of the system from an array of energy samples."
function avg_energy(energy_samples::AbstractVector,
                    sample_range::OptionalAbstractRange = nothing)
    return mean(range_of_nonmissing(energy_samples, sample_range))
end

"Extract Monte Carlo variance of energy from an array of energy samples."
function var_energy(energy_samples::AbstractVector,
                    sample_range::OptionalAbstractRange = nothing)
    return std(range_of_nonmissing(energy_samples, sample_range))
end

"Extract specific heat of the system from an array of energy samples."
function specific_heat(energy_samples::AbstractVector,
                       β::Float64,
                       sample_range::OptionalAbstractRange = nothing)
    E2 = range_of_nonmissing(energy_samples, sample_range) .^ 2
    return β^2 * (mean(E2) - avg_energy(energy_samples, sample_range) ^ 2)
end

"Extract average chirality from an array of chirality samples."
function avg_chirality(chirality_samples::AbstractVector,
                       sample_range::OptionalAbstractRange = nothing)
    return mean(range_of_nonmissing(chirality_samples, sample_range))
end

"Extract Monte Carlo variance of chirality from an array of chirality samples."
function var_chirality(chirality_samples::AbstractVector,
                       sample_range::OptionalAbstractRange = nothing)
    return std(range_of_nonmissing(chirality_samples, sample_range))
end

"Extract chiral susceptibility from an array of chirality samples."
function susc_chirality(chirality_samples::AbstractVector,
                        N::Int,
                        β::Float64,
                        sample_range::OptionalAbstractRange = nothing)
    χ_range = range_of_nonmissing(chirality_samples, sample_range)
    return N * β * (mean(χ_range .^ 2) - mean(χ_range) ^ 2)
end

"Extract Binder cumulant of chirality from an array of chirality samples."
function binder_chirality(chirality_samples::AbstractVector,
                          sample_range::OptionalAbstractRange = nothing)
    χ_range = range_of_nonmissing(chirality_samples, sample_range)
    χ2_avg = mean(χ_range .^ 2)
    χ4_avg = mean(χ_range .^ 4)
    return 3 / 2 * (1 - χ4_avg / (2 * χ2_avg ^ 2))
end

"Extract correlation length of chirality from arrays of chirality samples."
function corr_length_chirality(
    chirality_samples::AbstractVector,
    chirality_at_q_min_samples::AbstractVector,
    L1::Int,
    sample_range::OptionalAbstractRange = nothing)
    χ2_avg = mean(range_of_nonmissing(chirality_samples, sample_range) .^ 2)
    χ_at_q_min_2_avg = mean(
        abs.(range_of_nonmissing(chirality_at_q_min_samples, sample_range)) .^ 2
    )
    return sqrt(complex((χ2_avg / χ_at_q_min_2_avg) - 1)) / (2 * sin(π / L1))
end

"Extract K-point spin susceptibility from an array of magnetization samples."
function susc_spin_K(magnetization_at_K_samples::AbstractMatrix,
                     N::Int,
                     β::Float64,
                     sample_range::OptionalAbstractRange = nothing)
    magn_range = range_of_nonmissing(magnetization_at_K_samples, sample_range)
    magn_mean = mean(magn_range, dims=2)
    return N * β * (mean(map(norm, eachslice(magn_range, dims=2)) .^ 2)
                    - norm(magn_mean) ^ 2)
end

"Extract K-point spin Binder cumulant from an array of magnetization samples."
function binder_spin_K(magnetization_at_K_samples::AbstractMatrix,
                       sample_range::OptionalAbstractRange = nothing)
    magn_range = range_of_nonmissing(magnetization_at_K_samples, sample_range)
    magn_norm = map(norm, eachslice(magn_range, dims=2))
    magn2_avg = mean(magn_norm .^ 2)
    magn4_avg = mean(magn_norm .^ 4)
    return 2.0 - magn4_avg / (magn2_avg ^ 2)
end

"Extract correlation length of K-point magnetization from arrays of magnetization samples."
function corr_length_spin_K(magnetization_at_K_samples::AbstractMatrix,
                            magnetization_at_K_q_min_samples::AbstractMatrix,
                            L1::Int,
                            sample_range::OptionalAbstractRange = nothing)
    magn_norm = map(norm, eachslice(
        range_of_nonmissing(magnetization_at_K_samples, sample_range), dims=2)
    )
    magn_at_q_min_range = map(norm, eachslice(
        range_of_nonmissing(magnetization_at_K_q_min_samples, sample_range), dims=2)
    )
    magn2_avg = mean(magn_norm .^ 2)
    magn_at_q_min_2_avg = mean(magn_at_q_min_range .^ 2)
    return sqrt(complex((magn2_avg / magn_at_q_min_2_avg) - 1)) / (2 * sin(π / L1))
end

#
# Extract observables from a simulation structure
#

"Extract the lowest encountered energy from a simulation structure."
function min_energy(sim::MCSimulation, sample_range::OptionalAbstractRange = nothing)
    return min_energy(sim.measurements.energy, sample_range)
end

"Extract average energy of the system from a simulation structure."
function avg_energy(sim::MCSimulation, sample_range::OptionalAbstractRange = nothing)
    return avg_energy(sim.measurements.energy, sample_range)
end

"Extract Monte Carlo variance of energy from a simulation structure."
function var_energy(sim::MCSimulation, sample_range::OptionalAbstractRange = nothing)
    return var_energy(sim.measurements.energy, sample_range)
end

"Extract specific heat of the system from a simulation structure."
function specific_heat(sim::MCSimulation, sample_range::OptionalAbstractRange = nothing)
    return specific_heat(sim.measurements.energy, sim.β, sample_range)
end

"Extract average chirality from a simulation structure."
function avg_chirality(sim::MCSimulation, sample_range::OptionalAbstractRange = nothing)
    return avg_chirality(sim.measurements.chirality, sample_range)
end

"Extract Monte Carlo variance of chirality from a simulation structure."
function var_chirality(sim::MCSimulation, sample_range::OptionalAbstractRange = nothing)
    return var_chirality(sim.measurements.chirality, sample_range)
end

"Extract chiral susceptibility from a simulation structure."
function susc_chirality(sim::MCSimulation, sample_range::OptionalAbstractRange = nothing)
    L1, L2 = size(sim.conf)
    return susc_chirality(sim.measurements.chirality, L1 * L2, sim.β, sample_range)
end

"Extract Binder cumulant of chirality from a simulation structure."
function binder_chirality(sim::MCSimulation, sample_range::OptionalAbstractRange = nothing)
    return binder_chirality(sim.measurements.chirality, sample_range)
end

"Extract correlation length of chirality from a simulation structure."
function corr_length_chirality(sim::MCSimulation,
                               sample_range::OptionalAbstractRange = nothing)
    return corr_length_chirality(sim.measurements.chirality,
                                 sim.measurements.chirality_at_q_min,
                                 size(sim.conf)[1],
                                 sample_range)
end

"Extract K-point spin susceptibility from a simulation structure."
function susc_spin_K(sim::MCSimulation, sample_range::OptionalAbstractRange = nothing)
    L1, L2 = size(sim.conf)
    return susc_spin_K(sim.measurements.magnetization_at_K, L1 * L2, sim.β, sample_range)
end

"Extract K-point spin Binder cumulant from a simulation structure."
function binder_spin_K(sim::MCSimulation, sample_range::OptionalAbstractRange = nothing)
    return binder_spin_K(sim.measurements.magnetization_at_K, sample_range)
end

"Extract correlation length of K-point magnetization from a simulation structure."
function corr_length_spin_K(sim::MCSimulation,
                            sample_range::OptionalAbstractRange = nothing)
    return corr_length_spin_K(sim.measurements.magnetization_at_K,
                              sim.measurements.magnetization_at_K_q_min,
                              size(sim.conf)[1],
                              sample_range)
end

end # module observables
