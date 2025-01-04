module montecarlo

using OffsetArrays: OffsetArray, Origin
using LinearAlgebra: dot, issymmetric, diag
using SpecialFunctions: besselj0, besselj1, bessely0, bessely1
using Random: AbstractRNG, MersenneTwister, rand
using Statistics: mean, std
using StatsBase: sample, Weights
using ProgressMeter: @showprogress

using MX2RKKY.utility: clamp_angle, sph2cart, range_of_nonmissing
using MX2RKKY.lattice: IntCoords, neighbors, triangle_nodes

export Configuration, MCSimulation, run!, acceptance_rates
export min_energy, avg_energy, var_energy, avg_chirality, var_chirality, specific_heat

#
# Configuration
#

"Configuration as an array of classical spins on a cluster"
struct Configuration
    "Length of the spin vectors"
    S::Float64
    "Polar angles of the spins"
    θ::OffsetArray{Float64, 2}
    "Azimuthal angles of the spins"
    ϕ::OffsetArray{Float64, 2}
    """
    Cartesian components of the spins S_{α,i1,i2}, α = x,y,z.
    i1 and i2 are integer coordinates of the spin on the cluster.
    """
    S_vec::OffsetArray{Float64, 3}
end

"""
Make an in-plane configuration of spins `S` corresponding to the 120° Néel state with a
given chirality χ (either -1 or 1).
Linear sizes of the configuration `L1` and `L2` must be multiples of 3.
"""
function Configuration(L1::Int, L2::Int, χ::Int; S::Float64 = 1.0)
    @assert L1 > 0
    @assert L2 > 0
    @assert (L1 % 3 == 0) "L1 must be a multiple of 3"
    @assert (L2 % 3 == 0) "L2 must be a multiple of 3"
    @assert χ ∈ (-1, 1)
    θ = OffsetArray(fill(π / 2, L1, L2), Origin(0))
    ϕ = OffsetArray(zeros(Float64, L1, L2), Origin(0))
    ϕ_pattern = (χ == -1) ? (π / 2, 7π / 6, 11π / 6) : (π / 2, 11π / 6, 7π / 6)
    for (i1, i2) in Base.product(0:(L1-1), 0:(L2-1))
        ϕ[i1, i2] = ϕ_pattern[(i1 + i2) % 3 + 1]
    end
    S_vec = [n[α] for α in 1:3, n in sph2cart.(S, θ, ϕ)]
    return Configuration(S, θ, ϕ, OffsetArray(S_vec, Origin(1, 0, 0)))
end

"""
Make a disordered configuration of spins `S` with given linear sizes `L1` and `L2`.
Both linear sizes must be multiples of 3.
`rng` is the random number generator used to initialize components of the spins.
z-components of all spins are set to zero, if `inplane = true`.
"""
function Configuration(L1::Int, L2::Int, rng::AbstractRNG;
                       S::Float64 = 1.0, inplane::Bool = true)
    @assert L1 > 0
    @assert L2 > 0
    @assert (L1 % 3 == 0) "L1 must be a multiple of 3"
    @assert (L2 % 3 == 0) "L2 must be a multiple of 3"
    θ = OffsetArray(inplane ? fill(π / 2, L1, L2) : (π * rand(rng, L1, L2)), Origin(0))
    ϕ = OffsetArray(2π * rand(rng, L1, L2), Origin(0))
    S_vec = [n[α] for α in 1:3, n in sph2cart.(S, θ, ϕ)]
    return Configuration(S, θ, ϕ, OffsetArray(S_vec, Origin(1, 0, 0)))
end

"Extract linear sizes (L1, L2) of the cluster from a configuration"
Base.size(conf::Configuration) = size(conf.S_vec)[2:3]

"Iterate over all integer coordinates of nodes in the cluster from a configuration"
Base.eachindex(conf::Configuration) = Base.product(axes(conf.S_vec, 2), axes(conf.S_vec, 3))

"
Update a single spin in a configuration.

`conf` - configuration to be updated.
`i` - integer coordinates of the spin.
`θ` - new polar angle of the spin.
`ϕ` - new azimuthal angle of the spin.
"
@inline function update_spin!(conf::Configuration, i::IntCoords, θ::Float64, ϕ::Float64)
    conf.θ[i...] = θ
    conf.ϕ[i...] = ϕ
    conf.S_vec[:, i...] .= collect(sph2cart(conf.S, θ, ϕ))
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
    ϕ1 = conf.ϕ[tn[1]...]
    ϕ2 = conf.ϕ[tn[2]...]
    ϕ3 = conf.ϕ[tn[3]...]
    χ = -(clamp_angle(ϕ2 - ϕ1) + clamp_angle(ϕ3 - ϕ2) + clamp_angle(ϕ1 - ϕ3)) / (2π)
    @assert (isapprox(χ, -1, atol=1e-10) ||
             isapprox(χ, 0, atol=1e-10) ||
             isapprox(χ, 1, atol=1e-10))
    return Int(round(χ))
end

#
# Heisenberg XXZ model
#

"""
Compute energy contributions from the Heisenberg XXZ interaction of a single node with its
neighbors. Add it to a given pair interaction energy array.

`conf` - configuration of the interacting spins.
`i` - integer coordinates of the node.
`pair_int_energy` - interaction energies of `i` with each node.
`Jpar` - longitudinal Heisenberg interaction constant J_{\\parallel}.
`Jperp` - transversal Heisenberg interaction constant J_{\\perp}.
"""
@inline function xxz_energy_contrib!(conf::Configuration, i::IntCoords,
                                     pair_int_energy::AbstractArray{Float64, 2};
                                     Jpar::Float64, Jperp::Float64)
    L1, L2 = size(conf)
    S_i = conf.S_vec[:, i...]
    for j in neighbors(i, L1, L2)
        S_j = conf.S_vec[:, j...]
        E = Jpar * (S_i[1] * S_j[1] + S_i[2] * S_j[2]) + Jperp * S_i[3] * S_j[3]
        pair_int_energy[j...] += E / 2
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
                              renormalize::Bool = false,
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
Compute energy contribution from the RKKY interaction of a single node with all other nodes.
Add it to a given pair interaction array.

`conf` - configuration of the interacting spins.
`i` - integer coordinates of the node.
`pair_int_energy` - interaction energies of `i` with each node.
`V` - RKKY interaction tensor.
"""
@inline function rkky_energy_contrib!(conf::Configuration, i::IntCoords,
                                      pair_int_energy::AbstractArray{Float64, 2};
                                      V::OffsetArray{Float64, 4})
    L1, L2 = size(conf)
    S_i = conf.S_vec[:, i...]
    for j in eachindex(conf)
        (i == j) && continue # No self-interaction
        S_j = conf.S_vec[:, j...]
        Δi = j .- i
        pair_int_energy[j...] += dot(S_i, V[:, :, Δi...], S_j)
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
    "Samples: Chirality"
    chirality::Vector{Union{Float64, Missing}}
    Measurements() = new(0, [], [])
end

"Clear measurement results and pre-allocate sample arrays"
function reset!(m::Measurements, n_samples::Int = 0)
    m.count = 0
    resize!(m.energy, n_samples)
    resize!(m.chirality, n_samples)
    fill!(m.energy, missing)
    fill!(m.chirality, missing)
end

"Base type of Monte Carlo updates"
abstract type AbstractUpdate end

"State of a Monte Carlo simulation"
mutable struct MCSimulation
    "Inverse temperature"
    const β::Float64
    "Heisenberg interaction parameter J_{\\parallel}"
    const Jpar::Float64
    "Heisenberg interaction parameter J_{\\perp}"
    const Jperp::Float64
    "RKKY interaction tensor"
    const V::Union{OffsetArray{Float64, 4}, Nothing}
    "Current spin configuration"
    conf::Configuration
    "Interaction energy (1/2) E[i1,i2,j1,j2] between nodes (i1, i2) and (j1, j2)"
    pair_int_energy::OffsetArray{Float64, 4}
    "Total energy of the system"
    energy::Float64
    "Chirality of each triangle in the configuration"
    chirality::OffsetArray{Int, 2}
    "Chirality summed over all nodes"
    chirality_sum::Int
    "Pseudo-random number generator"
    rng::MersenneTwister
    "Monte Carlo updates"
    updates::Vector{AbstractUpdate}
    "Number of proposed updates of each type"
    n_proposed::Vector{Int}
    "Number of accepted updates of each type"
    n_accepted::Vector{Int}
    "Taken measurements"
    const measurements::Measurements
end

"Construct a simulation structure"
function MCSimulation(init_conf::Configuration,
                      rng::AbstractRNG;
                      β::Float64,
                      Jpar::Float64, Jperp::Float64,
                      Vpar::Float64, Vperp::Float64, λ::Float64,
                      Vrenormalize::Bool = false,
                      Vmaxshells::Int = 100,
                      Vatol::Float64 = 1e-4)
    L1, L2 = size(init_conf)

    enable_rkky = !(iszero(Vpar) && iszero(Vperp))
    if enable_rkky
        V = make_rkky_int_tensor(L1, L2, Vpar, Vperp, λ,
                                 renormalize=Vrenormalize, maxshells=Vmaxshells, atol=Vatol)
    end

    pair_int_energy = OffsetArray(zeros(Float64, L1, L2, L1, L2), Origin(0))
    for i in eachindex(init_conf)
        pair_int_energy_i = @view pair_int_energy[:, :, i...]
        xxz_energy_contrib!(init_conf, i, pair_int_energy_i, Jpar=Jpar, Jperp=Jperp)
        if enable_rkky
            rkky_energy_contrib!(init_conf, i, pair_int_energy_i, V=V)
        end
    end

    # Check that pair_int_energy is symmetric and does not contain self-interaction terms
    N = L1 * L2
    pair_int_energy_mat = reshape(pair_int_energy, N, N)
    @assert isapprox(pair_int_energy_mat, pair_int_energy_mat', atol=1e-10)
    @assert iszero(diag(pair_int_energy_mat))
    energy = sum(pair_int_energy)

    χ = OffsetArray(Array{Int, 2}(undef, L1, L2), Origin(0))
    for i in eachindex(init_conf)
        χ[i...] = chirality(init_conf, i)
    end

    return MCSimulation(β, Jpar, Jperp, enable_rkky ? V : nothing,
                        init_conf,
                        pair_int_energy, energy,
                        χ, sum(χ),
                        rng,
                        [], zeros(Int, 2), zeros(Int, 2),
                        Measurements())
end

"""
Run Monte Carlo simulation.

`sim` - MC simulation structure.
`n_cycles` - Number of measurement cycles.
`n_warmup_cycles` - Number of warm-up cycles.
`cycle_length` - Number of MC updates per cycle.
`inplane` - Update only the azimuthal angles of the spins.
`update_triangle_prob` - Proposal probability of UpdateTriangle.
"""
function run!(sim::MCSimulation;
              n_cycles::Int, n_warmup_cycles::Int, cycle_length::Int = 100,
              inplane::Bool = true,
              update_triangle_prob::Float64 = .0)
    @assert n_cycles >= 0
    @assert n_warmup_cycles >= 0
    @assert cycle_length > 0

    # Markov chain updates
    L1, L2 = size(sim.conf)
    empty!(sim.updates)
    push!(sim.updates, UpdateSingleSpin(L1, L2, inplane))
    push!(sim.updates, UpdateTriangle(L1, L2))
    update_weights = [1.0 - update_triangle_prob, update_triangle_prob]

    # Initialize measurements
    reset!(sim.measurements, n_cycles)

    @showprogress desc="Running MC simulation" for cycle in 1:(n_warmup_cycles + n_cycles)
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

#
# UpdateSingleSpin
#

"Monte Carlo update: Select a random spin and change its components"
mutable struct UpdateSingleSpin{Bool} <: AbstractUpdate
    "Integer coordinates of the spin to be updated"
    i::IntCoords
    "Integer coordinates of the adjacent triangles"
    adj_triangles::Vector{IntCoords}
    "Polar angle before update"
    θ_old::Float64
    "Azimuthal angle before update"
    ϕ_old::Float64
    "Cartesian components of the spin before update"
    S_vec_old::Vector{Float64}
    "Contributions of the selected spin to system's energy after update"
    pair_int_energy_new::OffsetArray{Float64, 2}
    "Total energy change caused by the update"
    ΔE::Float64
    "Chirality of the affected triangles after update"
    χ_new::Vector{Int}

    UpdateSingleSpin(L1::Int, L2::Int, inplane::Bool) = new{inplane}(
        (0, 0), fill((0, 0), 3), .0, .0, zeros(3),
        OffsetArray(zeros(Float64, L1, L2), Origin(0)), .0,
        zeros(Int, 3)
    )
end

isinplane(::UpdateSingleSpin{inplane}) where inplane = inplane

"Propose update of a single spin in the cluster"
function propose!(update::UpdateSingleSpin, sim::MCSimulation)::Float64
    L1, L2 = size(sim.conf)
    update.i = (rand(sim.rng, 0:(L1-1)),
                rand(sim.rng, 0:(L2-1)))
    update.adj_triangles .= [update.i,
                             (update.i[1], mod(update.i[2] - 1, L2)),
                             (mod(update.i[1] - 1, L1), mod(update.i[2] - 1, L2))]

    update.θ_old = sim.conf.θ[update.i...]
    update.ϕ_old = sim.conf.ϕ[update.i...]
    update.S_vec_old = sim.conf.S_vec[:, update.i...]

    θ = isinplane(update) ? (π / 2) : (π * rand(sim.rng, Float64))
    ϕ = 2π * rand(sim.rng, Float64)

    update_spin!(sim.conf, update.i, θ, ϕ)

    update.pair_int_energy_new .= 0
    xxz_energy_contrib!(sim.conf, update.i, update.pair_int_energy_new,
                        Jpar=sim.Jpar, Jperp=sim.Jperp)
    if !isnothing(sim.V)
        rkky_energy_contrib!(sim.conf, update.i, update.pair_int_energy_new, V=sim.V)
    end

    pair_int_energy_old = @view sim.pair_int_energy[:, :, update.i...]
    update.ΔE = 2 * (sum(update.pair_int_energy_new) - sum(pair_int_energy_old))

    for (n, i) in enumerate(update.adj_triangles)
        update.χ_new[n] = chirality(sim.conf, i)
    end

    return exp(-sim.β * update.ΔE)
end

"Accept update of a single spin in the cluster"
function accept!(update::UpdateSingleSpin, sim::MCSimulation)
    sim.pair_int_energy[:, :, update.i...] .= update.pair_int_energy_new
    sim.pair_int_energy[update.i..., :, :] .= update.pair_int_energy_new
    sim.energy += update.ΔE
    for (n, i) in enumerate(update.adj_triangles)
        χ_old = sim.chirality[i...]
        sim.chirality_sum += update.χ_new[n] - χ_old
        sim.chirality[i...] = update.χ_new[n]
    end
end

"Reject update of a single spin in the cluster"
function reject!(update::UpdateSingleSpin, sim::MCSimulation)
    sim.conf.θ[update.i...] = update.θ_old
    sim.conf.ϕ[update.i...] = update.ϕ_old
    sim.conf.S_vec[:, update.i...] .= update.S_vec_old
end

#
# UpdateTriangle
#

"""
Monte Carlo update: Rotate spins on a randomly selected elementary triangle by the same
random azimuthal angle.
"""
mutable struct UpdateTriangle <: AbstractUpdate
    "Integer coordinates of the spins to be updated"
    i::Vector{IntCoords}
    "Azimuthal angles before update"
    ϕ_old::Vector{Float64}
    "Cartesian components of the spins before update"
    S_vec_old::Array{Float64, 2}
    "Contributions of the selected spins to system's energy after update"
    pair_int_energy_new::OffsetArray{Float64, 3}
    "Total energy change caused by the update"
    ΔE::Float64

    UpdateTriangle(L1::Int, L2::Int) = new(
        fill((0, 0), 3),
        zeros(Float64, 3),
        zeros(Float64, 3, 3),
        OffsetArray(zeros(Float64, L1, L2, 3), Origin(0, 0, 1)),
        .0
    )
end

"Propose update of a triangle"
function propose!(update::UpdateTriangle, sim::MCSimulation)::Float64
    L1, L2 = size(sim.conf)
    update.i .= triangle_nodes((rand(sim.rng, 0:(L1-1)), rand(sim.rng, 0:(L2-1))), L1, L2)
    Δϕ = 2π * rand(sim.rng, Float64)

    for (n, i) in enumerate(update.i)
        update.ϕ_old[n] = sim.conf.ϕ[i...]
        update.S_vec_old[:, n] .= sim.conf.S_vec[:, i...]
        ϕ = mod(sim.conf.ϕ[i...] + Δϕ, 2π)
        update_spin!(sim.conf, i, sim.conf.θ[i...], ϕ)
    end

    update.ΔE = 0
    for (n, i) in enumerate(update.i)
        pair_int_energy_new = @view update.pair_int_energy_new[:, :, n]
        pair_int_energy_new .= 0;
        xxz_energy_contrib!(sim.conf, i, pair_int_energy_new,
                            Jpar=sim.Jpar, Jperp=sim.Jperp)
        if !isnothing(sim.V)
            rkky_energy_contrib!(sim.conf, i, pair_int_energy_new, V=sim.V)
        end

        pair_int_energy_old = @view sim.pair_int_energy[i..., :, :]
        update.ΔE += 2 * (sum(pair_int_energy_new) - sum(pair_int_energy_old))
    end

    # Subtract the double counting terms stemming from the interaction between the 3
    # selected nodes
    update.ΔE -= 2 * (update.pair_int_energy_new[update.i[2]..., 1] -
                      sim.pair_int_energy[update.i[1]..., update.i[2]...])
    update.ΔE -= 2 * (update.pair_int_energy_new[update.i[3]..., 2] -
                      sim.pair_int_energy[update.i[2]..., update.i[3]...])
    update.ΔE -= 2 * (update.pair_int_energy_new[update.i[1]..., 3] -
                      sim.pair_int_energy[update.i[3]..., update.i[1]...])

    return exp(-sim.β * update.ΔE)
end

"Accept update of a triangle"
function accept!(update::UpdateTriangle, sim::MCSimulation)
    for (n, i) in enumerate(update.i)
        sim.pair_int_energy[:, :, i...] .= update.pair_int_energy_new[:, :, n]
        sim.pair_int_energy[i..., :, :] .= update.pair_int_energy_new[:, :, n]
    end
    sim.energy += update.ΔE
    # TODO: Update sim.chirality and sim.chirality_sum
end

"Reject update of a triangle"
function reject!(update::UpdateTriangle, sim::MCSimulation)
    for (n, i) in enumerate(update.i)
        sim.conf.ϕ[i...] = update.ϕ_old[n]
        sim.conf.S_vec[:, i...] .= update.S_vec_old[:, n]
    end
end

#
# Measurements
#

"Update measurement data"
function measure!(sim::MCSimulation)
    m = sim.measurements
    m.count += 1
    m.energy[m.count] = sim.energy

    # Chirality
    L1, L2 = size(sim.conf)
    m.chirality[m.count] = sim.chirality_sum / (L1 * L2)
end

"""
Extract acceptance rates of the Monte Carlo updates from a simulation structure.
"""
function acceptance_rates(sim::MCSimulation)
    rates = sim.n_accepted ./ sim.n_proposed
    return Dict(typeof(u) => r for (u, r) in zip(sim.updates, rates))
end

"Extract the lowest encountered energy from a simulation structure."
function min_energy(sim::MCSimulation,
                    sample_range::Union{AbstractRange, Nothing} = nothing)
    return minimum(range_of_nonmissing(sim.measurements.energy, sample_range))
end

"Extract average energy of the system from a simulation structure."
function avg_energy(sim::MCSimulation,
                    sample_range::Union{AbstractRange, Nothing} = nothing)
    return mean(range_of_nonmissing(sim.measurements.energy, sample_range))
end

"Extract Monte Carlo variance of energy from a simulation structure."
function var_energy(sim::MCSimulation,
                    sample_range::Union{AbstractRange, Nothing} = nothing)
    return std(range_of_nonmissing(sim.measurements.energy, sample_range))
end

"Extract average chirality from a simulation structure."
function avg_chirality(sim::MCSimulation,
                       sample_range::Union{AbstractRange, Nothing} = nothing)
    return mean(range_of_nonmissing(sim.measurements.chirality, sample_range))
end

"Extract Monte Carlo variance of chirality from a simulation structure."
function var_chirality(sim::MCSimulation,
                       sample_range::Union{AbstractRange, Nothing} = nothing)
    return std(range_of_nonmissing(sim.measurements.chirality, sample_range))
end

"Extract specific heat of the system from a simulation structure."
function specific_heat(sim::MCSimulation,
                       sample_range::Union{AbstractRange, Nothing} = nothing)
    E2 = range_of_nonmissing(sim.measurements.energy, sample_range) .^ 2
    return sim.β^2 * (mean(E2) - avg_energy(sim, sample_range) ^ 2)
end

end # module montecarlo
