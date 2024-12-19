module MX2RKKY

using OffsetArrays: OffsetArray, Origin
using LinearAlgebra: dot
using SpecialFunctions: besselj0, besselj1, bessely0, bessely1
using Random: AbstractRNG, MersenneTwister, rand
using StatsBase: sample, Weights
using ProgressMeter: @showprogress

export Configuration, MCSimulation, run!, acceptance_rates
export avg_energy, var_energy, avg_chirality, var_chirality

#
# Triangular lattice
#

"Integer coordinates of 2D lattice vectors in the basis a1 = (1, 0), a2 = (-1/2, √3/2)"
const IntCoords = Tuple{Int, Int}

"""
Given coordinates of a node `i`, return coordinates of its neighbors on
a parallelogram-shaped cluster of the triangular lattice with periodic boundary conditions.
The neighbors are listed in the clockwise order.
The shape of the cluster is parametrized by two linear sizes, `L1` and `L2`.
"""
function neighbors(i::IntCoords, L1::Int, L2::Int)::Vector{IntCoords}
    i1, i2 = i
    return [
        (mod(i1 + 1, L1), i2),
        (i1, mod(i2 - 1, L2)),
        (mod(i1 - 1, L1), mod(i2 - 1, L2)),
        (mod(i1 - 1, L1), i2),
        (i1, mod(i2 + 1, L2)),
        (mod(i1 + 1, L1), mod(i2 + 1, L2))
    ]
end

"""
Given coordinates of a node `i`, return coordinates of the nodes belonging to the same
hexagon. The nodes belong to a parallelogram-shaped cluster of the triangular lattice
with periodic boundary conditions and are listed in the clockwise order.
`L1` and `L2` are linear sizes of the cluster.
"""
function hexagon_nodes(i::IntCoords, L1::Int, L2::Int)::Vector{IntCoords}
    i1, i2 = i
    return [i, (i1, mod(i2 + 1, L2)), (mod(i1 + 1, L1), mod(i2 + 1, L2))]
end

#
# Spins
#

"Convert spherical coordinates (1, θ, φ) to Cartesian components (x, y, z)"
function sph2cart(θ::Float64, ϕ::Float64)::Tuple{Float64, Float64, Float64}
    return (sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ))
end

"Cast a given difference of azimuthal angles Δφ into the range [-π; π]"
function clampΔφ(Δϕ::Float64)
    if Δϕ > π
        return Δϕ - 2π
    elseif Δϕ < -π
        return Δϕ + 2π
    else
        return Δϕ
    end
end

"configuration as an array of classical spins on a cluster"
struct Configuration
    "Length of the spin vectors"
    S::Float64
    "Azimuthal angles of the spins"
    ϕ::OffsetArray{Float64, 2}
    "Polar angles of the spins"
    θ::OffsetArray{Float64, 2}
    """
    Cartesian components of the spins S_{α,i1,i2}, α = x,y,z.
    i1 and i2 are integer coordinates of the spin on the cluster.
    """
    S_vec::OffsetArray{Float64, 3}
end

"""
Make a completely polarized (ϕ = 0) in-plane configuration of spins `S`
with given linear sizes `L1` and `L2`.
"""
function Configuration(L1::Int, L2::Int; S::Float64 = 1.0)
    @assert L1 > 0
    @assert L2 > 0
    θ = OffsetArray(fill(π / 2, L1, L2), Origin(0))
    ϕ = OffsetArray(fill(.0, L1, L2), Origin(0))
    S_vec = [S * n[α] for α in 1:3, n in sph2cart.(θ, ϕ)]
    return Configuration(S, ϕ, θ, OffsetArray(S_vec, Origin(1, 0, 0)))
end

"""
Make a disordered configuration of spins `S` with given linear sizes `L1` and `L2`.
`rng` is the random number generator used to initialize components of the spins.
z-components of all spins are set to zero, if `inplane = true`.
"""
function Configuration(L1::Int, L2::Int, rng::AbstractRNG;
                       S::Float64 = 1.0, inplane::Bool = true)
    @assert L1 > 0
    @assert L2 > 0
    θ = OffsetArray(inplane ? fill(π / 2, L1, L2) : (π * rand(rng, L1, L2)), Origin(0))
    ϕ = OffsetArray(2π * rand(rng, L1, L2), Origin(0))
    S_vec = [S * n[α] for α in 1:3, n in sph2cart.(θ, ϕ)]
    return Configuration(S, ϕ, θ, OffsetArray(S_vec, Origin(1, 0, 0)))
end

"Extract linear sizes (L1, L2) of the cluster from a configuration"
Base.size(conf::Configuration) = size(conf.S_vec)[2:3]

"Iterate over all integer coordinates of nodes in the cluster from a configuration"
Base.eachindex(conf::Configuration) = Base.product(axes(conf.S_vec, 2), axes(conf.S_vec, 3))

#
# Heisenberg XXZ model
#

"""
Energy contribution from the Heisenberg XXZ interaction of a single node with its neighbors.

`conf` - configuration of the interacting spins.
`i` - integer coordinates of the node.
`Jpar` - longitudinal Heisenberg interaction constant J_{\\parallel}.
`Jperp` - transversal Heisenberg interaction constant J_{\\perp}.
"""
@inline function xxz_energy_term(conf::Configuration, i::IntCoords;
                                 Jpar::Float64, Jperp::Float64)
    L1, L2 = size(conf)
    S_i = conf.S_vec[:, i...]
    return sum(neighbors(i, L1, L2)) do j
        S_j = conf.S_vec[:, j...]
        Jpar * (S_i[1] * S_j[1] + S_i[2] * S_j[2]) + Jperp * S_i[3] * S_j[3]
    end
end

"""
Energy contributions from the Heisenberg XXZ terms.

`conf` - configuration of the interacting spins.
`Jpar` - longitudinal Heisenberg interaction constant J_{\\parallel}.
`Jperp` - transversal Heisenberg interaction constant J_{\\perp}.
"""
function xxz_energy(conf::Configuration; Jpar::Float64, Jperp::Float64)
    L1, L2 = size(conf)
    return 0.5 * sum(eachindex(conf)) do i
        xxz_energy_term(conf, i, Jpar=Jpar, Jperp=Jperp)
    end
end

#
# RKKY interaction
#

"""
Bare RKKY interaction V(R) as a 3x3 matrix in the space of Cartesian spin components.

`i` - integer coordinates of R.
`Vpar` - longitudinal RKKY interaction constant V_{\\parallel}.
`Vperp` - transversal RKKY interaction constant V_{\\perp}.
`λ` - interaction range parameter k_F a0.
"""
function rkky_int(i::IntCoords, Vpar::Float64, Vperp::Float64, λ::Float64)
    r = sqrt(i[1]^2 - i[1] * i[2] + i[2]^2) # r = |i1 a1 + i2 a2| / a0
    ϕ = 8π / 3 * (i[1] - i[2] / 2)
    v = besselj0(λ * r) * bessely0(λ * r) + besselj1(λ * r) * bessely1(λ * r)
    return [ Vpar*cos(ϕ) Vpar*sin(ϕ) 0;
            -Vpar*sin(ϕ) Vpar*cos(ϕ) 0;
             0           0           Vperp] .* v
end

"""
Numerically compute renormalized RKKY interaction V(R) by summing over replicas of
the main cluster.

`i` - integer coordinates of R.
`Vpar` - longitudinal RKKY interaction constant V_{\\parallel}.
`Vperp` - transversal RKKY interaction constant V_{\\perp}.
`λ` - interaction range parameter k_F a0.
`L1` and `L2` - linear sizes of the cluster.
`maxshells` - maximal number of replica 'shells' to sum over.
`atol` - absolute tolerance for result convergence check.
"""
function renorm_rkky_int(i::IntCoords, Vpar::Float64, Vperp::Float64, λ::Float64;
                         L1::Int, L2::Int, maxshells::Int = 100, atol::Float64 = 1e-4)
    @assert i != (0, 0)
    # Sum over 'shells' of replicas surrounding the main cluster
    res = rkky_int(i, Vpar, Vperp, λ)
    (i1, i2) = i
    for shell in 1:(maxshells-1)
        # Sum over all replicas within a shell
        shell_contrib = zeros(Float64, 3, 3)
        for s in 0:(shell-1)
            shell_contrib .+= rkky_int((i1 + L1 * (shell - s), i2 - L2 * s), Vpar, Vperp, λ)
            shell_contrib .+= rkky_int((i1 - L1 * (shell - s), i2 + L2 * s), Vpar, Vperp, λ)
            shell_contrib .+= rkky_int((i1 - L1 * s, i2 - L2 * (shell - s)), Vpar, Vperp, λ)
            shell_contrib .+= rkky_int((i1 + L1 * s, i2 + L2 * (shell - s)), Vpar, Vperp, λ)
        end
        isapprox(shell_contrib, zeros(3, 3), atol=atol) && return res
        res += shell_contrib
    end
    throw(ErrorException("Could not obtain a converged result for i = $(i) "
                         * "after summing over $(maxshells) shells of cluster replicas"))
end

"""
Make the RKKY interaction tensor V_{α1,α2,i1,i2}, α1, α2 = x,y,z.

i1 and i2 are integer coordinates from the ranges (1-L1):(L1-1) and (1-L2):(L2-1)
respectively.

`L1` and `L2` - linear sizes of the cluster.
`Vpar` - longitudinal RKKY interaction constant V_{\\parallel}.
`Vperp` - transversal RKKY interaction constant V_{\\perp}.
`λ` - interaction range parameter k_F a0.
`renormalize` - use the renormalized values of the interaction.
`maxshells` - maximal number of replica 'shells' to sum over when computing
the renormalized interaction.
`atol` - absolute tolerance for result convergence check when computing
the renormalized interaction.
"""
function make_rkky_int_tensor(L1::Int, L2::Int, Vpar::Float64, Vperp::Float64, λ::Float64;
                              renormalize::Bool = true,
                              maxshells::Int = 100, atol::Float64 = 1e-4)

    V = OffsetArray(Array{Float64, 4}(undef, 3, 3, 2 * L1 - 1, 2 * L2 - 1),
                    1:3, 1:3, (1-L1):(L1-1), (1-L2):(L2-1))
    for i in CartesianIndices(V[1, 1, :, :])
        (i[1] == 0 && i[2] == 0) && continue
        if renormalize
            V[:, :, i] = renorm_rkky_int(
                Tuple(i), Vpar, Vperp, λ, L1=L1, L2=L2, maxshells=maxshells, atol=atol
            )
        else
            V[:, :, i] = rkky_int(Tuple(i), Vpar, Vperp, λ)
        end
    end
    return V
end

"""
Energy contribution from the RKKY interaction of a single node with all other nodes.

`conf` - configuration of the interacting spins.
`i` - integer coordinates of the node.
`V` - RKKY interaction tensor.
"""
@inline function rkky_energy_term(conf::Configuration, i::IntCoords,
                                  V::OffsetArray{Float64, 4})
    L1, L2 = size(conf)
    S_i = conf.S_vec[:, i...]
    return sum(eachindex(conf)) do j
        (i == j) && return .0 # No self-interaction
        S_j = conf.S_vec[:, j...]
        Δi = i .- j
        return dot(S_i, V[:, :, Δi...], S_j)
    end
end

"""
Energy contributions from the RKKY terms.

`conf` - configuration of the interacting spins.
`V` - RKKY interaction tensor.
"""
function rkky_energy(conf::Configuration, V::OffsetArray{Float64, 4})
    L1, L2 = size(conf)
    return 0.5 * sum(eachindex(conf)) do i
        rkky_energy_term(conf, i, V)
    end
end

#
# Monte Carlo simulation
#

"Collection of measurements"
mutable struct Measurements
    "Number of measurements taken"
    count::Int
    "Accumulated energy of the system (unnormalized)"
    energy::Float64
    "Accumulated energy squared"
    energy2::Float64
    "Accumulated chirality (unnormalized)"
    chirality::Float64
    "Accumulated chirality squared"
    chirality2::Float64
    Measurements() = new(0, .0, .0, .0, .0)
end

"State of a Monte Carlo simulation"
mutable struct MCSimulation
    "Inverse temperature"
    const β::Float64
    "Heisenberg interaction parameter J_{\\parallel}"
    const Jpar::Float64
    "Heisenberg interaction parameter J_{\\perp}"
    const Jperp::Float64
    "RKKY interaction tensor"
    const V::OffsetArray{Float64, 4}
    "Current spin configuration"
    conf::Configuration
    "Node-resolved contributions to system's energy"
    energy_per_node::OffsetArray{Float64, 2}
    "Total energy of the system"
    energy::Float64
    "Pseudo-random number generator"
    rng::MersenneTwister
    "Monte Carlo updates"
    updates::Vector
    "Number of proposed updates of each type"
    n_proposed::Vector{Int}
    "Number of accepted updates of each type"
    n_accepted::Vector{Int}
    "Taken measurements"
    const measurements::Measurements
end

"Construct a simulation structure"
function MCSimulation(init_conf::Configuration;
                      β::Float64,
                      Jpar::Float64, Jperp::Float64,
                      Vpar::Float64, Vperp::Float64, λ::Float64,
                      Vrenormalize::Bool = True,
                      Vmaxshells::Int = 100,
                      Vatol::Float64 = 1e-4)
    L1, L2 = size(init_conf)
    V = make_rkky_int_tensor(L1, L2, Vpar, Vperp, λ,
                             renormalize=Vrenormalize, maxshells=Vmaxshells, atol=Vatol)
    energy_per_node = [xxz_energy_term(init_conf, i, Jpar=Jpar, Jperp=Jperp) +
                       rkky_energy_term(init_conf, i, V)
                       for i in eachindex(init_conf)]
    energy = 0.5 * sum(energy_per_node)
    return MCSimulation(β, Jpar, Jperp, V,
                        init_conf,
                        OffsetArray(energy_per_node, Origin(0)), energy,
                        MersenneTwister(),
                        [], zeros(Int, 1), zeros(Int, 1),
                        Measurements())
end

"""
Run Monte Carlo simulation.

`sim` - MC simulation structure.
`n_cycles` - Number of measurement cycles.
`n_warmup_cycles` - Number of warm-up cycles.
`cycle_length` - Number of MC updates per cycle.
`seed` - seed to initialize the PRNG.
`inplane` - Update only the azimuthal angles of the spins.
"""
function run!(sim::MCSimulation;
              n_cycles::Int, n_warmup_cycles::Int, cycle_length::Int = 100,
              seed::Union{Int, Nothing} = nothing,
              inplane::Bool = true)
    @assert n_cycles > 0
    @assert n_warmup_cycles > 0
    @assert cycle_length > 0

    # Optionally re-seed RNG
    if !isnothing(seed)
        sim.rng = MersenneTwister(seed)
    end

    # Markov chain updates
    empty!(sim.updates)
    push!(sim.updates, UpdateSingleSpin(inplane))
    update_weights = [1.0]

    @showprogress for cycle in 1:(n_warmup_cycles + n_cycles)
        for step in 1:cycle_length
            # Metropolis-Hastings criterion
            update_i = sample(sim.rng, Weights(update_weights))
            update = sim.updates[update_i]

            r = propose!(update, sim)
            sim.n_proposed[update_i] += 1

            if r > rand(sim.rng, Float64)
                accept!(update, sim)
                sim.n_accepted[update_i] += 1
            else
                reject!(update, sim)
            end
        end

        (cycle <= n_warmup_cycles) && continue

        measure!(sim)
    end
end

"Update measurement data"
function measure!(sim::MCSimulation)
    sim.measurements.count += 1
    sim.measurements.energy += sim.energy
    sim.measurements.energy2 += sim.energy^2

    # Chirality
    L1, L2 = size(sim.conf)
    L = L1 * L2
    χ = sum(eachindex(sim.conf)) do i
        hn = hexagon_nodes(i, L1, L2)
        ϕ1 = sim.conf.ϕ[hn[1]...]
        ϕ2 = sim.conf.ϕ[hn[2]...]
        ϕ3 = sim.conf.ϕ[hn[3]...]
        χi = -(clampΔφ(ϕ2 - ϕ1) + clampΔφ(ϕ3 - ϕ2) + clampΔφ(ϕ1 - ϕ3)) / (2π)
        @assert (isapprox(χi, -1, atol=1e-10) ||
                 isapprox(χi, 0, atol=1e-10) ||
                 isapprox(χi, 1, atol=1e-10))
        return χi
    end / L
    sim.measurements.chirality += χ
    sim.measurements.chirality2 += χ^2
end

mutable struct UpdateSingleSpin{Bool}
    "Integer coordinates of the spin to be updated"
    i::IntCoords
    "Azimuthal angle before update"
    ϕ_old::Float64
    "Polar angle before update"
    θ_old::Float64
    "Cartesian components of the spin before update"
    S_vec_old::Vector{Float64}
    "Contribution of the selected spin to energy's system before update"
    energy_old::Float64

    UpdateSingleSpin(inplane::Bool) = new{inplane}((0, 0), .0, .0, zeros(3), .0)
end

isinplane(::UpdateSingleSpin{inplane}) where inplane = inplane

"Propose update of a single spin in the cluster"
function propose!(update::UpdateSingleSpin, sim::MCSimulation)::Float64
    L1, L2 = size(sim.conf)
    update.i = (rand(sim.rng, 0:(L1-1)),
                rand(sim.rng, 0:(L2-1)))

    update.θ_old = sim.conf.θ[update.i...]
    update.ϕ_old = sim.conf.ϕ[update.i...]
    update.S_vec_old = sim.conf.S_vec[:, update.i...]
    update.energy_old = sim.energy_per_node[update.i...]

    θ = isinplane(update) ? (π / 2) : (π * rand(sim.rng, Float64))
    ϕ = 2π * rand(sim.rng, Float64)

    sim.conf.θ[update.i...] = θ
    sim.conf.ϕ[update.i...] = ϕ
    sim.conf.S_vec[:, update.i...] = sim.conf.S * collect(sph2cart(θ, ϕ))

    E = 0.5 * (xxz_energy_term(sim.conf, update.i, Jpar = sim.Jpar, Jperp = sim.Jperp) +
               rkky_energy_term(sim.conf, update.i, sim.V))
    sim.energy_per_node[update.i...] = E

    return exp(-sim.β * (E - update.energy_old))
end

"Accept update of a single spin in the cluster"
function accept!(update::UpdateSingleSpin, sim::MCSimulation)
    sim.energy += 0.5 * (sim.energy_per_node[update.i...] - update.energy_old)
end

"Reject update of a single spin in the cluster"
function reject!(update::UpdateSingleSpin, sim::MCSimulation)
    sim.conf.θ[update.i...] = update.θ_old
    sim.conf.ϕ[update.i...] = update.ϕ_old
    sim.conf.S_vec[:, update.i...] .= update.S_vec_old
    sim.energy_per_node[update.i...] = update.energy_old
end

"""
Extract acceptance rates of the Monte Carlo updates from a simulation structure.
"""
function acceptance_rates(sim::MCSimulation)
    rates = sim.n_accepted ./ sim.n_proposed
    return Dict(typeof(u) => r for (u, r) in zip(sim.updates, rates))
end

"Extract average energy of the system from a simulation structure."
avg_energy(sim::MCSimulation) = sim.measurements.energy / sim.measurements.count

"Extract variance of energy from a simulation structure."
function var_energy(sim::MCSimulation)
    E_avg = avg_energy(sim)
    return sqrt((sim.measurements.energy2 / sim.measurements.count) - E_avg^2)
end

"Extract average chirality from a simulation structure."
avg_chirality(sim::MCSimulation) = sim.measurements.chirality / sim.measurements.count

"Extract variance of chirality from a simulation structure."
function var_chirality(sim::MCSimulation)
    χ_avg = avg_chirality(sim)
    return sqrt((sim.measurements.chirality2 / sim.measurements.count) - χ_avg^2)
end

end # module MX2RKKY
