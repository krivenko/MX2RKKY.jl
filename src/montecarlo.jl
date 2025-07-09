module montecarlo

using OffsetArrays: OffsetArray, Origin
using LinearAlgebra: dot, issymmetric, diag
using SpecialFunctions: besselj0, besselj1, bessely0, bessely1
using Random: AbstractRNG, MersenneTwister, rand
using StatsBase: sample, Weights
using ProgressMeter: @showprogress
using Polyester: @batch

using MX2RKKY.utility: clamp_angle
using MX2RKKY.lattice: IntCoords, neighbor_shifts, periodize, neighbors, triangle_nodes, l2c
using MX2RKKY.lattice: LatticeArray

export Configuration, MCSimulation, run!

#
# Configuration
#

"Configuration as an array of classical spins on a cluster"
struct Configuration
    "Orientations of the spins"
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

"Iterate over all integer coordinates of nodes in the cluster from a configuration"
Base.eachindex(conf::Configuration) = Base.product(axes(conf.θ, 1), axes(conf.θ, 2))

"Read orientation of a single spin in a configuration"
Base.getindex(conf::Configuration, i::IntCoords) = conf.θ[i...]

"Set orientation of a single spin in a configuration"
function Base.setindex!(conf::Configuration, θ::Float64, i::IntCoords)
    conf.θ[i...] = θ
end

"""
Compute chirality of a single triangle in a given configuration.

The triangle is specified by the integer coordinates of its lowest (having the smallest `y`
coordinate) node.

`conf` - spin configuration.
`i` - integer coordinates of the node.
"""
@inline function chirality(conf::Configuration, i::IntCoords)
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
Compute energy contributions from the Heisenberg XY interaction of a single node with its
neighbors. Add it to a given pair interaction energy array.

`conf` - configuration of the interacting spins.
`i` - integer coordinates of the node.
`pair_int_energy` - interaction energies of `i` with each node.
`J` - Heisenberg interaction constant J.
"""
@inline function xy_energy_contrib!(conf::Configuration, i::IntCoords,
                                    pair_int_energy::AbstractArray{Float64, 2};
                                    J::Float64)
    L1, L2 = size(conf)
    θ_i = conf[i]
    @batch for j in neighbors(i, L1, L2)
        θ_j = conf[j]
        pair_int_energy[j...] += J * cos(θ_i - θ_j) / 2
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
function rkky_int(i::IntCoords, V::Float64, λ::Float64)
    r = sqrt(i[1]^2 - i[1] * i[2] + i[2]^2) # r = |i1 a1 + i2 a2| / a0
    return V * (besselj0(λ * r) * bessely0(λ * r) + besselj1(λ * r) * bessely1(λ * r))
end

"""
Compute spin rotation angle of the RKKY interaction ϕ(R).

`i` - integer coordinates of R.
"""
rkky_angle(i::IntCoords) = 8π / 3 * (i[1] - i[2] / 2)

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
        Varr[Δi] = rkky_int(Tuple(Δi), V, λ)
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
        ϕ[Δi] = rkky_angle(Tuple(Δi))
    end
    return ϕ
end

"""
Compute energy contribution from the RKKY interaction of a single node with all other nodes.
Add it to a given pair interaction array.

`conf` - configuration of the interacting spins.
`i` - integer coordinates of the node.
`pair_int_energy` - interaction energies of `i` with each node.
`V` - RKKY interaction array.
`ϕ` - RKKY rotation angle array.
"""
@inline function rkky_energy_contrib!(conf::Configuration, i::IntCoords,
                                      pair_int_energy::AbstractArray{Float64, 2},
                                      ::Val{0};
                                      V::OffsetArray{Float64, 2},
                                      ϕ::OffsetArray{Float64, 2})
    L1, L2 = size(conf)
    θ_i = conf[i]
    @batch for Δi in CartesianIndices(V)
        (Δi == CartesianIndex(0, 0)) && continue # No self-interaction
        j = periodize(i .+ Tuple(Δi), L1, L2)
        θ_j = conf[j]
        pair_int_energy[j...] += V[Δi] * cos(θ_i - θ_j - ϕ[Δi])
    end
end

"""
Compute energy contribution from the RKKY interaction of a single node with all nodes
within a given range `R`. Add it to a given pair interaction array.

`conf` - configuration of the interacting spins.
`i` - integer coordinates of the node.
`pair_int_energy` - interaction energies of `i` with each node.
`V` - RKKY interaction array.
`ϕ` - RKKY rotation angle array.
"""
@inline function rkky_energy_contrib!(conf::Configuration, i::IntCoords,
                                      pair_int_energy::AbstractArray{Float64, 2},
                                      ::Val{R};
                                      V::OffsetArray{Float64, 2},
                                      ϕ::OffsetArray{Float64, 2}) where R
    L1, L2 = size(conf)
    θ_i = conf[i]
    for r in 1:R
        for Δi in neighbor_shifts(Val(r))
            j = periodize(i .+ Δi, L1, L2)
            θ_j = conf[j]
            pair_int_energy[j...] += V[Δi...] * cos(θ_i - θ_j - ϕ[Δi...])
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
    "Samples: Energy of the system"
    energy::Vector{Union{Float64, Missing}}
    "Samples: Magnetization"
    magnetization::Array{Union{Float64, Missing}, 2}
    "Samples: Magnetization at K = (4π / 3) e_x"
    magnetization_at_K::Array{Union{ComplexF64, Missing}, 2}
    "Samples: Magnetization at K + q_min = (4π / 3 + 2π / L_1) e_x"
    magnetization_at_K_q_min::Array{Union{ComplexF64, Missing}, 2}
    "Samples: Chirality at q=0"
    chirality::Vector{Union{Float64, Missing}}
    "Samples: Chirality at q=q_min=(2π / L_1) e_x"
    chirality_at_q_min::Vector{Union{ComplexF64, Missing}}
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
            i -> exp(-im * dot(q_min, l2c(i...))), ComplexF64, 1, size(conf)...
        )
        fourier_weights_at_K = LatticeArray(
            i -> exp(-im * dot(K, l2c(i...))), ComplexF64, 1, size(conf)...
        )
        return new(0, [],
                   zeros(2, 0), zeros(ComplexF64, 2, 0), zeros(ComplexF64, 2, 0),
                   [], [],
                   fourier_weights_at_q_min,
                   fourier_weights_at_K,
                   fourier_weights_at_K .* fourier_weights_at_q_min)
    end
end

"Clear measurement results and pre-allocate sample arrays"
function reset!(m::Measurements, n_samples::Int = 0)
    m.count = 0
    m.energy = fill(missing, n_samples)
    m.magnetization = fill(missing, 2, n_samples)
    m.magnetization_at_K = fill(missing, 2, n_samples)
    m.magnetization_at_K_q_min = fill(missing, 2, n_samples)
    m.chirality = fill(missing, n_samples)
    m.chirality_at_q_min = fill(missing, n_samples)
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
    "Interaction energy (1/2) E[i1,i2,j1,j2] between nodes (i1, i2) and (j1, j2)"
    pair_int_energy::OffsetArray{Float64, 4}
    "Total energy of the system"
    energy::Float64
    "Pseudo-random number generator"
    rng::MersenneTwister
    "Taken measurements"
    const measurements::Measurements
end

"Construct a simulation structure"
function MCSimulation(init_conf::Configuration,
                      rng::AbstractRNG;
                      β::Float64,
                      J::Float64,
                      V::Float64, λ::Float64, rkky_range_cutoff::Int = 0)
    L1, L2 = size(init_conf)
    @assert rkky_range_cutoff <= 3 "rkky_range_cutoff > 3 is not implemented"

    enable_rkky = !iszero(V)
    if enable_rkky
        Varr = make_rkky_int_array(L1, L2, V, λ)
        ϕarr = make_rkky_angle_array(L1, L2)
    end

    pair_int_energy = LatticeArray(Float64, 2, L1, L2)
    for i in eachindex(init_conf)
        pair_int_energy_i = @view pair_int_energy[:, :, i...]
        xy_energy_contrib!(init_conf, i, pair_int_energy_i, J=J)
        if enable_rkky
            rkky_energy_contrib!(init_conf, i, pair_int_energy_i, Val(rkky_range_cutoff),
                                 V=Varr, ϕ=ϕarr)
        end
    end

    # Check that pair_int_energy is symmetric and does not contain self-interaction terms
    N = L1 * L2
    pair_int_energy_mat = reshape(pair_int_energy, N, N)
    @assert isapprox(pair_int_energy_mat, pair_int_energy_mat', atol=1e-10)
    @assert iszero(diag(pair_int_energy_mat))

    energy = sum(pair_int_energy)

    return MCSimulation(β, J,
                        enable_rkky ? Varr : nothing,
                        enable_rkky ? ϕarr : nothing,
                        rkky_range_cutoff,
                        init_conf,
                        pair_int_energy, energy,
                        rng,
                        Measurements(init_conf))
end

"""
Run Monte Carlo simulation.

`sim` - MC simulation structure.
`n_mcs` - Number of MC steps per spin (MCS).
`n_warmup_mcs` - Number of warm-up MC steps per spin.
`m_sweeps_per_mcs` - Number of Metropolis lattice sweeps per MCS.
`maxΔθ` - In the Metropolis sweeps, proposed spin rotations belong to the interval
          [-maxΔθ, maxΔθ].
`show_progress` - Show progress meter
"""
function run!(sim::MCSimulation;
              n_mcs::Int, n_warmup_mcs::Int,
              m_sweeps_per_mcs::Int = 1,
              maxΔθ::Float64 = Float64(π),
              show_progress::Bool = true)
    @assert n_mcs >= 0
    @assert n_warmup_mcs >= 0
    @assert m_sweeps_per_mcs > 0
    @assert 0 < maxΔθ <= π

    L1, L2 = size(sim.conf)

    # Metropolis update
    m_update = MetropolisUpdate(L1, L2, maxΔθ)

    # Initialize measurements
    reset!(sim.measurements, n_mcs)

    @showprogress(enabled=show_progress, desc="Running MC simulation",
    for msc in 1:(n_warmup_mcs + n_mcs)
        # Metropolis sweeps
        for sweep in 1:m_sweeps_per_mcs
            for i in eachindex(sim.conf)
                if propose!(m_update, i, sim) > rand(sim.rng, Float64)
                    accept!(m_update, sim)
                else
                    reject!(m_update, sim)
                end
            end
        end

        (msc <= n_warmup_mcs) && continue
        measure!(sim)
    end)

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
    "Integer coordinates of the spin to be updated"
    i::IntCoords
    "Angle before update"
    θ_old::Float64
    "Contributions of the selected spin to system's energy after update"
    pair_int_energy_new::OffsetArray{Float64, 2}
    "Total energy change caused by the update"
    ΔE::Float64
    "Number of proposed updates"
    n_proposed::Int
    "Number of accepted updates"
    n_accepted::Int

    MetropolisUpdate(L1::Int, L2::Int, maxΔθ::Float64) = new(
        maxΔθ, (0, 0), .0, LatticeArray(Float64, 1, L1, L2), .0, 0, 0
    )
end

"Propose update of a single spin in the cluster"
function propose!(update::MetropolisUpdate, i::IntCoords, sim::MCSimulation)::Float64
    update.n_proposed += 1
    update.i = i
    update.θ_old = sim.conf[i]
    sim.conf[i] = update.maxΔθ * (2.0 * rand(sim.rng, Float64) - 1.0)

    update.pair_int_energy_new .= 0
    xy_energy_contrib!(sim.conf, i, update.pair_int_energy_new, J=sim.J)
    if !isnothing(sim.V)
        rkky_energy_contrib!(
            sim.conf, i, update.pair_int_energy_new, Val(sim.rkky_range_cutoff),
            V=sim.V, ϕ=sim.ϕ
        )
    end

    pair_int_energy_old = @view sim.pair_int_energy[:, :, i...]
    update.ΔE = 2 * (sum(update.pair_int_energy_new) - sum(pair_int_energy_old))
    return exp(-sim.β * update.ΔE)
end

"Accept update of a single spin in the cluster"
function accept!(update::MetropolisUpdate, sim::MCSimulation)
    update.n_accepted += 1
    sim.pair_int_energy[:, :, update.i...] .= update.pair_int_energy_new
    sim.pair_int_energy[update.i..., :, :] .= update.pair_int_energy_new
    sim.energy += update.ΔE
end

"Reject update of a single spin in the cluster"
function reject!(update::MetropolisUpdate, sim::MCSimulation)
    sim.conf[update.i] = update.θ_old
end

#
# Measurements
#

"Update measurement data"
function measure!(sim::MCSimulation)
    measures = sim.measurements
    measures.count += 1
    measures.energy[measures.count] = sim.energy

    L1, L2 = size(sim.conf)
    N = L1 * L2

    χ::Float64 = 0                   # Chirality at q = 0
    χ_at_q_min::ComplexF64 = 0       # Chirality at q = q_min = (2π / L_1) e_x
    # Magnetization at q = 0
    m_x, m_y = zeros(Float64, 2)
    # Magnetization at K = (4π / 3) e_x
    m_at_K_x, m_at_K_y = zeros(ComplexF64, 2)
    # Magnetization at K + q_min = (4π / 3 + 2π / L_1) e_x
    m_at_K_q_min_x, m_at_K_q_min_y = zeros(ComplexF64, 2)

    @batch reduction=((+, m_x), (+, m_y),
                      (+, m_at_K_x), (+, m_at_K_y),
                      (+, m_at_K_q_min_x), (+, m_at_K_q_min_y),
                      (+, χ), (+, χ_at_q_min)) for i in eachindex(sim.conf)
        # Chirality
        χ_i = chirality(sim.conf, i)
        χ += χ_i
        χ_at_q_min += measures.fourier_weights_at_q_min[i...] * χ_i

        # Magnetization
        m_x_i = cos(sim.conf[i])
        m_y_i = sin(sim.conf[i])

        m_x += m_x_i
        m_y += m_y_i

        w_at_K = measures.fourier_weights_at_K[i...]
        m_at_K_x += w_at_K * m_x_i
        m_at_K_y += w_at_K * m_y_i

        w_at_K_q_min = measures.fourier_weights_at_K_q_min[i...]
        m_at_K_q_min_x += w_at_K_q_min * m_x_i
        m_at_K_q_min_y += w_at_K_q_min * m_y_i
    end

    measures.chirality[measures.count] = χ / N
    measures.chirality_at_q_min[measures.count] = χ_at_q_min / N
    measures.magnetization[:, measures.count] .= [m_x, m_y] / N
    measures.magnetization_at_K[:, measures.count] .= [m_at_K_x, m_at_K_y] / N
    measures.magnetization_at_K_q_min[:, measures.count] .= [m_at_K_q_min_x,
                                                             m_at_K_q_min_y] / N
end

end # module montecarlo
