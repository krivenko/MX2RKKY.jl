module observables

using LinearAlgebra: norm
using Statistics: mean, std

using MX2RKKY.utility: range_of_nonmissing
using MX2RKKY.montecarlo: MCSimulation

export min_energy, avg_energy, var_energy, err_energy, specific_heat
export avg_chirality, var_chirality, err_chirality, susc_chirality, binder_chirality
export corr_length_chirality
export avg_spin_K, var_spin_K, err_spin_K, susc_spin_K, binder_spin_K, corr_length_spin_K
export corr_chirality_spin_K, cov_chirality_spin_K

#
# Energy
#

"Extract the lowest encountered energy from a simulation structure."
min_energy(sim::MCSimulation) = sim.measurements.energy_min

"Extract average energy of the system from a simulation structure."
avg_energy(sim::MCSimulation) = sim.measurements.energy_sums[1] / sim.measurements.count

"Extract variance of energy from a simulation structure."
function var_energy(sim::MCSimulation)
    m = sim.measurements
    N = m.count
    return sqrt((m.energy_sums[2] - m.energy_sums[1] ^ 2 / N) / (N - 1))
end

"""
Extract estimated Monte Carlo error of the average energy of the system from a simulation
structure.
"""
err_energy(sim::MCSimulation) = var_energy(sim) / sqrt(sim.measurements.count)

"Extract specific heat of the system from a simulation structure."
specific_heat(sim::MCSimulation) = (sim.β * var_energy(sim)) ^ 2

#
# Chirality
#

"Extract average chirality from a simulation structure."
function avg_chirality(sim::MCSimulation)
    return sim.measurements.chirality_sums[1] / sim.measurements.count
end

"Extract variance of chirality from a simulation structure."
function var_chirality(sim::MCSimulation)
    m = sim.measurements
    N = m.count
    return sqrt((m.chirality_sums[2] - m.chirality_sums[1] ^ 2 / N) / (N - 1))
end

"Extract estimated Monte Carlo error of the average chirality from a simulation structure."
err_chirality(sim::MCSimulation) = var_chirality(sim) / sqrt(sim.measurements.count)

"Extract chiral susceptibility from a simulation structure."
function susc_chirality(sim::MCSimulation)
    L1, L2 = size(sim.conf)
    return L1 * L2 * sim.β * var_chirality(sim) ^ 2
end

"Extract Binder cumulant of chirality from a simulation structure."
function binder_chirality(sim::MCSimulation)
    m = sim.measurements
    χ2_avg = m.chirality_sums[2] / m.count
    χ4_avg = m.chirality_sums[4] / m.count
    return 3 / 2 * (1 - χ4_avg / (2 * χ2_avg ^ 2))
end

"Extract correlation length of chirality from a simulation structure."
function corr_length_chirality(sim::MCSimulation)
    L1, _ = size(sim.conf)
    m = sim.measurements
    χ2_avg = m.chirality_sums[2] / m.count
    χ_at_q_min_2_avg = m.chirality_at_q_min_2_sum / m.count
    return sqrt(complex((χ2_avg / χ_at_q_min_2_avg) - 1)) / (2 * sin(π / L1))
end

#
# Magnetization at K-point
#

"Extract average K-point magnetization from a simulation structure."
function avg_spin_K(sim::MCSimulation)
    return sim.measurements.magnetization_at_K_sum / sim.measurements.count
end

"Extract variance of K-point magnetization from a simulation structure."
function var_spin_K(sim::MCSimulation)
    m = sim.measurements
    N = m.count
    magn_avg = avg_spin_K(sim)
    magn2_avg = m.magnetization_at_K_norm_sums[2] / m.count
    return sqrt((magn2_avg - norm(magn_avg) ^ 2) * N / (N - 1))
end

"""
Extract estimated Monte Carlo error of the average K-point magnetization from a simulation
structure.
"""
err_spin_K(sim::MCSimulation) = var_spin_K(sim) / sqrt(sim.measurements.count)

"Extract K-point spin susceptibility from a simulation structure."
function susc_spin_K(sim::MCSimulation)
    L1, L2 = size(sim.conf)
    return L1 * L2 * sim.β * var_spin_K(sim)
end

"Extract K-point spin Binder cumulant from a simulation structure."
function binder_spin_K(sim::MCSimulation)
    m = sim.measurements
    magn2_avg = m.magnetization_at_K_norm_sums[2] / m.count
    magn4_avg = m.magnetization_at_K_norm_sums[4] / m.count
    return 2.0 - magn4_avg / (magn2_avg ^ 2)
end

"Extract correlation length of K-point magnetization from a simulation structure."
function corr_length_spin_K(sim::MCSimulation)
    L1, _ = size(sim.conf)
    m = sim.measurements
    magn2_avg = m.magnetization_at_K_norm_sums[2] / m.count
    magn_at_q_min_2_avg = m.magnetization_at_K_q_min_norm_sums[2] / m.count
    return sqrt(complex((magn2_avg / magn_at_q_min_2_avg) - 1)) / (2 * sin(π / L1))
end

"""
Extract correlation between the chirality and the K-point magnetization from a simulation
structure.
"""
function corr_chirality_spin_K(sim::MCSimulation)
    m = sim.measurements
    return m.chirality_magnetization_at_K_sum / m.count
end

"""
Extract covariance between the chirality and the K-point magnetization from a simulation
structure.
"""
function cov_chirality_spin_K(sim::MCSimulation)
    m = sim.measurements
    N = m.count
    χ_magn_at_K_avg = corr_chirality_spin_K(sim)
    χ_avg = avg_chirality(sim)
    magn_at_K_avg = m.magnetization_at_K_sum / m.count
    return (χ_magn_at_K_avg .- χ_avg .* magn_at_K_avg) * N / (N - 1)
end

end # module observables
