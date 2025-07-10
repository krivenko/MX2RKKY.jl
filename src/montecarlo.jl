module montecarlo

using OffsetArrays: OffsetArray, Origin
using LinearAlgebra: dot, issymmetric, diag, norm
using SpecialFunctions: besselj0, besselj1, bessely0, bessely1
using Random: AbstractRNG, MersenneTwister, rand
using StaticArrays: MVector, @MVector
using StatsBase: sample, Weights
using ProgressMeter: @showprogress
using MPI

using MX2RKKY.utility: clamp_angle
using MX2RKKY.lattice: LatCoords, neighbor_shifts, periodize, neighbors, triangle_nodes, l2p
using MX2RKKY.lattice: LatticeArray

export Configuration, MCSimulation, run!

#
# Configuration
#

"Configuration as an array of classical spins on a cluster"
struct Configuration
    "Orientations of the spins θ ∈ [0; 2π)"
    θ::OffsetArray{Float64, 2}
end

"""
Make a configuration of spins corresponding to the 120° Néel state with a given chirality χ
(either -1 or 1). Linear sizes of the configuration `L1` and `L2` must be multiples of 3.
"""
function Configuration(L1::Int, L2::Int, χ::Int)
    @assert L1 > 0
    @assert L2 > 0
    @assert isodd(L1) && (L1 % 3 == 0) "L1 must be odd and a multiple of 3"
    @assert isodd(L2) && (L2 % 3 == 0) "L2 must be odd and a multiple of 3"
    @assert χ ∈ (-1, 1)
    θ_pattern = (χ == -1) ? (π / 2, 7π / 6, 11π / 6) : (π / 2, 11π / 6, 7π / 6)
    θ = LatticeArray(i -> θ_pattern[(i[1] + i[2]) % 3 + 1], Float64, 1, L1, L2)
    return Configuration(θ)
end

"""
Make a disordered configuration of spins with given linear sizes `L1` and `L2`.
Both linear sizes must be multiples of 3.
`rng` is the random number generator used to initialize components of the spins.
"""
function Configuration(L1::Int, L2::Int, rng::AbstractRNG)
    @assert L1 > 0
    @assert L2 > 0
    @assert isodd(L1) && (L1 % 3 == 0) "L1 must be odd and a multiple of 3"
    @assert isodd(L2) && (L2 % 3 == 0) "L2 must be odd and a multiple of 3"
    return Configuration(LatticeArray(i -> 2π * rand(rng), Float64, 1, L1, L2))
end

"Extract linear sizes (L1, L2) of the cluster from a configuration"
Base.size(conf::Configuration) = size(conf.θ)

"Iterate over all lattice coordinates of nodes in the cluster from a configuration"
Base.CartesianIndices(conf::Configuration) = CartesianIndices(conf.θ)

"Read orientation of a single spin in a configuration"
Base.getindex(conf::Configuration, i::LatCoords) = conf.θ[i]

"Set orientation of a single spin in a configuration"
function Base.setindex!(conf::Configuration, θ::Float64, i::LatCoords)
    conf.θ[i] = θ
end

"""
Compute chirality of a single triangle in a given configuration.

The triangle is specified by the lattice coordinates of its lowest (having the smallest `y`
coordinate) node.

`conf` - spin configuration.
`i` - lattice coordinates of the node.
"""
@inline function chirality(conf::Configuration, i::LatCoords)
    L1, L2 = size(conf)
    tn = triangle_nodes(i, L1, L2)
    θ1 = conf[tn[1]]
    θ2 = conf[tn[2]]
    θ3 = conf[tn[3]]
    χ = -(clamp_angle(θ2 - θ1) + clamp_angle(θ3 - θ2) + clamp_angle(θ1 - θ3)) / (2π)
    @assert (isapprox(χ, -1, atol=1e-10) ||
             isapprox(χ, 0, atol=1e-10) ||
             isapprox(χ, 1, atol=1e-10))
    return χ
end

#
# Heisenberg XY model
#

"""
Compute effective field felt by a single node from the Heisenberg XY interaction with its
neighbors.

`conf` - configuration of the interacting spins.
`i` - lattice coordinates of the node.
`J` - Heisenberg interaction constant J.
"""
@inline function xy_eff_field_contrib(conf::Configuration,
                                      i::LatCoords;
                                      J::Float64)::MVector{2, Float64}
    L1, L2 = size(conf)
    res = @MVector zeros(Float64, 2)
    for j in neighbors(i, L1, L2)
        s, c = sincos(conf[j])
        res[1] += -J * c / 2
        res[2] += -J * s / 2
    end
    return res
end

"""
Recompute contribution of the Heisenberg XY interaction to the effective field after a
change of a single spin in the cluster.

`eff_field` - effective field.
`conf` - configuration of the interacting spins.
`i` - lattice coordinates of the updated spin.
`Δθ` - angle change of the spin.
`J` - Heisenberg interaction constant J.
"""
@inline function xy_update_eff_field!(eff_field::OffsetArray{Float64, 3},
                                      conf::Configuration,
                                      i::LatCoords,
                                      Δθ::Float64;
                                      J::Float64)
    L1, L2 = size(conf)
    a = J * sin(Δθ / 2)
    s, c = sincos(conf[i] + Δθ / 2)
    ΔHx, ΔHy = a * s, -a * c
    for j in neighbors(i, L1, L2)
        eff_field[1, j] += ΔHx
        eff_field[2, j] += ΔHy
    end
end

#
# RKKY interaction
#

"""
Compute magnitude of the RKKY interaction V(R).

`i` - integer coordinates of R.
`V` - RKKY interaction constant V.
`λ` - interaction range parameter k_F a0.
"""
function rkky_int(i::LatCoords, V::Float64, λ::Float64)
    r = sqrt(i[1]^2 - i[1] * i[2] + i[2]^2) # r = |i1 a1 + i2 a2| / a0
    return V * (besselj0(λ * r) * bessely0(λ * r) + besselj1(λ * r) * bessely1(λ * r))
end

"""
Compute spin rotation angle of the RKKY interaction ϕ(R).

`i` - integer coordinates of R.
"""
rkky_angle(i::LatCoords) = 8π / 3 * (i[1] - i[2] / 2)

"""
Make an array of RKKY interactions V_{i1, i2}.

i1 and i2 are integer coordinates from the ranges -(L1 - 1)/2:(L1 - 1)/2 and
-(L2 - 1)/2:(L2 - 1)/2 respectively.

`L1` and `L2` - linear sizes of the cluster.
`V` - RKKY interaction constant V.
`λ` - interaction range parameter k_F a0.
"""
function make_rkky_int_array(L1::Int, L2::Int, V::Float64, λ::Float64)
    i1r = -div(L1 - 1, 2):div(L1 - 1, 2)
    i2r = -div(L2 - 1, 2):div(L2 - 1, 2)
    Varr = OffsetArray(zeros(Float64, length(i1r), length(i2r)), i1r, i2r)
    for Δi in CartesianIndices(Varr)
        (Δi[1] == 0 && Δi[2] == 0) && continue
        Varr[Δi] = rkky_int(Δi, V, λ)
    end
    return Varr
end

"""
Make an array of RKKY rotation angles ϕ_{i1,i2}.

i1 and i2 are integer coordinates from the ranges -(L1 - 1)/2:(L1 - 1)/2 and
-(L2 - 1)/2:(L2 - 1)/2 respectively.

`L1` and `L2` - linear sizes of the cluster.
"""
function make_rkky_angle_array(L1::Int, L2::Int)
    i1r = -div(L1 - 1, 2):div(L1 - 1, 2)
    i2r = -div(L2 - 1, 2):div(L2 - 1, 2)
    ϕ = OffsetArray(zeros(Float64, length(i1r), length(i2r)), i1r, i2r)
    for Δi in CartesianIndices(ϕ)
        (Δi[1] == 0 && Δi[2] == 0) && continue
        ϕ[Δi] = rkky_angle(Δi)
    end
    return ϕ
end

"""
Compute effective field felt by a single node from the RKKY interaction with all other
nodes.

`conf` - configuration of the interacting spins.
`i` - Lattice coordinates of the node.
`V` - RKKY interaction array.
`ϕ` - RKKY rotation angle array.
"""
@inline function rkky_eff_field_contrib(conf::Configuration, i::LatCoords,
                                        ::Val{0};
                                        V::OffsetArray{Float64, 2},
                                        ϕ::OffsetArray{Float64, 2})::MVector{2, Float64}
    L1, L2 = size(conf)
    res = @MVector zeros(Float64, 2)
    for Δi in CartesianIndices(V)
        (Δi == CartesianIndex(0, 0)) && continue # No self-interaction
        j = periodize(i + Δi, L1, L2)
        s, c = sincos(conf[j] + ϕ[Δi])
        v = V[Δi]
        res[1] += -v * c
        res[2] += -v * s
    end
    return res
end

"""
Compute effective field felt by a single node from the RKKY interaction with all other
nodes within a given range `R`.

`conf` - configuration of the interacting spins.
`i` - Lattice coordinates of the node.
`V` - RKKY interaction array.
`ϕ` - RKKY rotation angle array.
"""
@inline function rkky_eff_field_contrib(
    conf::Configuration, i::LatCoords,
    ::Val{R};
    V::OffsetArray{Float64, 2},
    ϕ::OffsetArray{Float64, 2})::MVector{2, Float64} where R
    L1, L2 = size(conf)
    res = @MVector zeros(Float64, 2)
    for r in 1:R
        for Δi in neighbor_shifts(Val(r))
            j = periodize(i + Δi, L1, L2)
            s, c = sincos(conf[j] + ϕ[Δi])
            v = V[Δi]
            res[1] += -v * c
            res[2] += -v * s
        end
    end
    return res
end

"""
Recompute contribution of the RKKY interaction with all other nodes to the effective field
after a change of a single spin in the cluster.

`eff_field` - effective field.
`conf` - configuration of the interacting spins.
`i` - Lattice coordinates of the updated spin.
`Δθ` - angle change of the spin.
`V` - RKKY interaction array.
`ϕ` - RKKY rotation angle array.
"""
@inline function rkky_update_eff_field!(eff_field::OffsetArray{Float64, 3},
                                        conf::Configuration,
                                        i::LatCoords,
                                        ::Val{0},
                                        Δθ::Float64;
                                        V::OffsetArray{Float64, 2},
                                        ϕ::OffsetArray{Float64, 2})
    L1, L2 = size(conf)
    a = 2 * sin(Δθ / 2)
    ψ = conf[i] + Δθ / 2
    for Δi in CartesianIndices(V)
        (Δi == CartesianIndex(0, 0)) && continue # No self-interaction
        j = periodize(i + Δi, L1, L2)
        s, c = sincos(ψ + ϕ[-Δi])
        v = V[-Δi]
        eff_field[1, j] += a * v * s
        eff_field[2, j] += -a * v * c
    end
end

"""
Recompute contribution of the RKKY interaction with all other nodes within a given range `R`
to the effective field after a change of a single spin in the cluster.

`eff_field` - effective field.
`conf` - configuration of the interacting spins.
`i` - Lattice coordinates of the updated spin.
`Δθ` - angle change of the spin.
`V` - RKKY interaction array.
`ϕ` - RKKY rotation angle array.
"""
@inline function rkky_update_eff_field!(eff_field::OffsetArray{Float64, 3},
                                        conf::Configuration,
                                        i::LatCoords,
                                        ::Val{R},
                                        Δθ::Float64;
                                        V::OffsetArray{Float64, 2},
                                        ϕ::OffsetArray{Float64, 2}) where R
    L1, L2 = size(conf)
    a = 2 * sin(Δθ / 2)
    ψ = conf[i] + Δθ / 2
    for r in 1:R
        for Δi in neighbor_shifts(Val(r))
            j = periodize(i .+ Δi, L1, L2)
            s, c = sincos(ψ - ϕ[Δi])
            v = V[Δi]
            eff_field[1, j] += a * v * s
            eff_field[2, j] += -a * v * c
        end
    end
end

#
# Monte Carlo simulation
#

"Collection of measurements"
mutable struct Measurements
    "Number of measurements taken"
    count::Int
    "Encountered minimum of energy"
    energy_min::Float64
    "Powers of energy E, E^2, E^3, E^4 summed over MC samples"
    energy_sums::Vector{Float64}
    "Magnetization m(K) at K = (4π / 3) e_x summed over MC samples"
    magnetization_at_K_sum::Vector{ComplexF64}
    """
    Powers of norm of magnetization |m(K)|^p at K = (4π / 3) e_x for p=1:4
    summed over MC samples
    """
    magnetization_at_K_norm_sums::Vector{Float64}
    """
    Powers of norm of magnetization |m(K + q_min)|^p at K + q_min = (4π / 3 + 2π / L_1) e_x
    for p=1:4 summed over MC samples
    """
    magnetization_at_K_q_min_norm_sums::Vector{Float64}
    "Powers of chirality χ, χ^2, χ^3, χ^4 summed over MC samples"
    chirality_sums::Vector{Float64}
    "Squared chirality |χ(q)|^2 at q=q_min=(2π / L_1) e_x summed over MC samples"
    chirality_at_q_min_2_sum::ComplexF64
    "Fourier weights exp(-i q_min r_i)"
    fourier_weights_at_q_min::OffsetArray{ComplexF64, 2}
    "Fourier weights exp(-i K r_i)"
    fourier_weights_at_K::OffsetArray{ComplexF64, 2}
    "Fourier weights exp(-i (K + q_min) r_i)"
    fourier_weights_at_K_q_min::OffsetArray{ComplexF64, 2}

    function Measurements(conf::Configuration)
        L1, _ = size(conf)
        q_min = (2π / L1, 0.0)
        K = (4π / 3, 0)
        fourier_weights_at_q_min = LatticeArray(
            i -> exp(-im * dot(q_min, l2p(i[1], i[2]))), ComplexF64, 1, size(conf)...
        )
        fourier_weights_at_K = LatticeArray(
            i -> exp(-im * dot(K, l2p(i[1], i[2]))), ComplexF64, 1, size(conf)...
        )
        return new(0, Inf64, zeros(Float64, 4),
                   zeros(ComplexF64, 2), zeros(Float64, 4), zeros(Float64, 4),
                   zeros(Float64, 4), .0,
                   fourier_weights_at_q_min,
                   fourier_weights_at_K,
                   fourier_weights_at_K .* fourier_weights_at_q_min)
    end
end

"Clear measurement results and pre-allocate sample arrays"
function reset!(m::Measurements)
    m.count = 0
    m.energy_min = Inf64
    fill!(m.energy_sums, .0)
    fill!(m.magnetization_at_K_sum, 0.)
    fill!(m.magnetization_at_K_norm_sums, 0.)
    fill!(m.magnetization_at_K_q_min_norm_sums, 0.)
    fill!(m.chirality_sums, .0)
    m.chirality_at_q_min_2_sum = 0
end

"MPI-reduction (summation) of measurement results"
function Allreduce!(m::Measurements, comm::MPI.Comm)
    m.count = MPI.Allreduce(m.count, +, comm)
    m.energy_min = MPI.Allreduce(m.energy_min, min, comm)
    MPI.Allreduce!(m.energy_sums, +, comm)
    MPI.Allreduce!(m.magnetization_at_K_sum, +, comm)
    MPI.Allreduce!(m.magnetization_at_K_norm_sums, +, comm)
    MPI.Allreduce!(m.magnetization_at_K_q_min_norm_sums, +, comm)
    MPI.Allreduce!(m.chirality_sums, +, comm)
    m.chirality_at_q_min_2_sum = MPI.Allreduce(m.chirality_at_q_min_2_sum, +, comm)
end

"State of a Monte Carlo simulation"
mutable struct MCSimulation
    "Inverse temperature"
    const β::Float64
    "Heisenberg interaction parameter J"
    const J::Float64
    "RKKY interaction array"
    const V::Union{OffsetArray{Float64, 2}, Nothing}
    "RKKY rotation angle array"
    const ϕ::Union{OffsetArray{Float64, 2}, Nothing}
    "Range cutoff of RKKY interaction (<= 3, 0 = full range)"
    const rkky_range_cutoff::Int
    "Current spin configuration"
    conf::Configuration
    "Components of the effective field H[α, i, j] felt by a spin at (i, j), α = x, y"
    eff_field::OffsetArray{Float64, 3}
    "Total energy of the system"
    energy::Float64
    "Pseudo-random number generator"
    rng::MersenneTwister
    "Taken measurements"
    const measurements::Measurements
    "MPI communicator"
    comm::MPI.Comm
end

"Construct a simulation structure"
function MCSimulation(init_conf::Configuration,
                      rng::AbstractRNG;
                      β::Float64,
                      J::Float64,
                      V::Float64, λ::Float64, rkky_range_cutoff::Int = 0,
                      comm::MPI.Comm = MPI.COMM_WORLD)
    L1, L2 = size(init_conf)
    @assert rkky_range_cutoff <= 3 "rkky_range_cutoff > 3 is not implemented"

    enable_rkky = !iszero(V)
    if enable_rkky
        Varr = make_rkky_int_array(L1, L2, V, λ)
        ϕarr = make_rkky_angle_array(L1, L2)
    end

    eff_field = LatticeArray(Float64, 1, L1, L2, extra_leading_dims=(2,))
    for i in CartesianIndices(init_conf)
        eff_field[:, i] += xy_eff_field_contrib(init_conf, i, J=J)
        if enable_rkky
            eff_field[:, i] += rkky_eff_field_contrib(init_conf, i,
                                                      Val(rkky_range_cutoff),
                                                      V=Varr, ϕ=ϕarr)
        end
    end

    energy = -sum(cos.(init_conf.θ) .* eff_field[1, :, :] .+
                  sin.(init_conf.θ) .* eff_field[2, :, :])

    return MCSimulation(β, J,
                        enable_rkky ? Varr : nothing,
                        enable_rkky ? ϕarr : nothing,
                        rkky_range_cutoff,
                        init_conf,
                        eff_field, energy,
                        rng,
                        Measurements(init_conf),
                        comm)
end

"""
Run Monte Carlo simulation.

`sim` - MC simulation structure.
`n_mcs` - Number of MC steps per spin (MCS).
`n_warmup_mcs` - Number of warm-up MC steps per spin.
`m_sweeps_per_mcs` - Number of Metropolis lattice sweeps per MCS.
`or_sweeps_per_mcs` - Number of overrelaxation lattice sweeps per MCS.
`maxΔθ` - In the Metropolis sweeps, proposed spin rotations belong to the interval
          [-maxΔθ, maxΔθ].
`show_progress` - Show progress meter
"""
function run!(sim::MCSimulation;
              n_mcs::Int, n_warmup_mcs::Int,
              m_sweeps_per_mcs::Int = 1,
              or_sweeps_per_mcs::Int = 0,
              maxΔθ::Float64 = Float64(π),
              show_progress::Bool = true)
    @assert n_mcs >= 0
    @assert n_warmup_mcs >= 0
    @assert m_sweeps_per_mcs > 0
    @assert 0 < maxΔθ <= π

    mpi_size = MPI.Comm_size(sim.comm)
    mpi_rank = MPI.Comm_rank(sim.comm)

    # Metropolis update
    m_update = MetropolisUpdate(maxΔθ)

    # Initialize measurements
    reset!(sim.measurements)

    @showprogress(enabled=show_progress,
                  desc="[Rank $(mpi_rank)/$(mpi_size)] Running MC simulation",
    for msc in 1:(n_warmup_mcs + n_mcs)
        # Metropolis sweeps
        for sweep in 1:m_sweeps_per_mcs
            for i in CartesianIndices(sim.conf)
                if propose!(m_update, i, sim) > rand(sim.rng, Float64)
                    accept!(m_update, sim)
                end
            end
        end

        # Overrelaxation sweeps
        for sweep in 1:or_sweeps_per_mcs
            for i in CartesianIndices(sim.conf)
                overrelaxation!(i, sim)
            end
        end

        (msc <= n_warmup_mcs) && continue
        measure!(sim)
    end)

    Allreduce!(sim.measurements, sim.comm)

    # Return Metropolis acceptance rate
    return m_update.n_accepted / m_update.n_proposed
end

#
# UpdateMetropolis
#

"Metropolis update: rotate a given spin by a random angle"
mutable struct MetropolisUpdate
    "Proposed spin rotations belong to the interval [-maxΔθ, maxΔθ]"
    const maxΔθ::Float64
    "Lattice coordinates of the spin to be updated"
    i::LatCoords
    "Proposed angle change"
    Δθ::Float64
    "Total energy change caused by the update"
    ΔE::Float64
    "Number of proposed updates"
    n_proposed::Int
    "Number of accepted updates"
    n_accepted::Int

    MetropolisUpdate(maxΔθ::Float64) = new(maxΔθ, LatCoords(0, 0), .0, .0, 0, 0)
end

"Propose update of a single spin in the cluster"
function propose!(update::MetropolisUpdate, i::LatCoords, sim::MCSimulation)::Float64
    update.n_proposed += 1

    update.i = i
    update.Δθ = update.maxΔθ * (2rand(sim.rng, Float64) - 1)

    s, c = sincos(sim.conf[i] + update.Δθ / 2)
    H = @view sim.eff_field[:, i]
    update.ΔE = 4 * sin(update.Δθ / 2) * (s * H[1] - c * H[2])
    return exp(-sim.β * update.ΔE)
end

"Accept update of a single spin in the cluster"
function accept!(update::MetropolisUpdate, sim::MCSimulation)
    update.n_accepted += 1
    xy_update_eff_field!(sim.eff_field, sim.conf, update.i, update.Δθ, J=sim.J)
    if !isnothing(sim.V)
        rkky_update_eff_field!(sim.eff_field, sim.conf, update.i,
                               Val(sim.rkky_range_cutoff), update.Δθ, V=sim.V, ϕ=sim.ϕ)
    end
    sim.conf[update.i] = mod2pi(sim.conf[update.i] + update.Δθ)
    sim.energy += update.ΔE
end

#
# Overrelaxation
#

"""
Apply the overrelaxation update to a single spin.

`i` - Lattice coordinates of the updated spin.
`sim` - `sim` - MC simulation structure.
"""
function overrelaxation!(i::LatCoords, sim::MCSimulation)
    η = atan(sim.eff_field[2, i], sim.eff_field[1, i])
    θ = mod2pi(2η - sim.conf[i])
    Δθ = θ - sim.conf[i]
    xy_update_eff_field!(sim.eff_field, sim.conf, i, Δθ, J=sim.J)
    if !isnothing(sim.V)
        rkky_update_eff_field!(sim.eff_field, sim.conf, i,
                               Val(sim.rkky_range_cutoff), Δθ, V=sim.V, ϕ=sim.ϕ)
    end
    sim.conf[i] = θ
end

#
# Measurements
#

"Update measurement data"
function measure!(sim::MCSimulation)
    L1, L2 = size(sim.conf)
    N = L1 * L2

    m = sim.measurements
    m.count += 1

    # Energy
    m.energy_min = min(m.energy_min, sim.energy)
    m.energy_sums .+= sim.energy .^ (1:4)

    χ::Float64 = 0                         # Chirality at q = 0
    χ_at_q_min::ComplexF64 = 0             # Chirality at q = q_min = (2π / L_1) e_x
    magn = zeros(Float64, 2)               # Magnetization at q = 0
    magn_at_K = zeros(ComplexF64, 2)       # Magnetization at K = (4π / 3) e_x
    magn_at_K_q_min = zeros(ComplexF64, 2) # Magnetization at K + q_min

    for i in CartesianIndices(sim.conf)
        # Chirality
        χ_i = chirality(sim.conf, i)
        χ += χ_i
        χ_at_q_min += m.fourier_weights_at_q_min[i] * χ_i

        # Magnetization
        magn_y_i, magn_x_i = sincos(sim.conf[i])

        magn .+= (magn_x_i, magn_y_i)

        w_at_K = m.fourier_weights_at_K[i]
        magn_at_K .+= w_at_K .* (magn_x_i, magn_y_i)

        w_at_K_q_min = m.fourier_weights_at_K_q_min[i]
        magn_at_K_q_min .+= w_at_K_q_min .* (magn_x_i, magn_y_i)
    end

    m.chirality_sums .+= (χ / N) .^ (1:4)
    m.chirality_at_q_min_2_sum += abs(χ_at_q_min / N) ^ 2

    m.magnetization_at_K_sum .+= magn_at_K ./ N
    m.magnetization_at_K_norm_sums .+= norm(magn_at_K ./ N) .^ (1:4)
    m.magnetization_at_K_q_min_norm_sums .+= norm(magn_at_K_q_min ./ N) .^ (1:4)
end

end # module montecarlo
