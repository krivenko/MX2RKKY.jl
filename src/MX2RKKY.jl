module MX2RKKY

using LinearAlgebra: dot
using SpecialFunctions: besselj0, besselj1, bessely0, bessely1
using Random

export Configuration

#
# Triangular lattice
#

"One-based index of a node on a 2D lattice"
const Node = Tuple{Int, Int}
"Cartesian coordinates of a node on a 2D lattice"
const Coords = Tuple{Float64, Float64}

"""
Convert an index of a node on a triangular lattice to its Cartesian coordinates.
"""
i2c(i::Node, a0::Float64)::Coords = a0 .* ((i[1] - 1) - 0.5 * (i[2] - 1),
                                           (√3 / 2) * (i[2] - 1))

"""
Given a node, return indices of its neighbors on a cluster of a triangular
lattice with periodic boundary conditions.
The cluster is parametrized by two linear sizes, L1 and L2.
"""
function neighbors(i::Node, L1::Int, L2::Int)::Vector{Node}
    i1, i2 = i
    return [(mod(i1, L1) + 1, i2),
            (mod(i1 - 2, L1) + 1, i2),
            (i1, mod(i2, L2) + 1),
            (i1, mod(i2 - 2, L2) + 1),
            (mod(i1, L1) + 1, mod(i2, L2) + 1),
            (mod(i1 - 2, L1) + 1, mod(i2 - 2, L2) + 1)]
end

#
# Spins
#

"Convert polar coordinates to Cartesian components"
function polar2cartesian(ϕ::Array{Float64}, θ::Array{Float64})
    @assert size(ϕ) == size(θ)
    cart = Array{Float64}(undef, 3, size(ϕ)...)
    S_x = sin.(θ) .* cos.(ϕ)
    S_y = sin.(θ) .* sin.(ϕ)
    S_z = cos.(θ)
    return stack((S_x, S_y, S_z), dims=1)
end

"Configuration as an array of classical spins"
struct Configuration
    "Length of the spin vectors"
    S::Float64
    "Azimuthal angles of the spins"
    ϕ::Array{Float64, 2}
    "Polar angles of the spins"
    θ::Array{Float64, 2}
    "Cartesian components of the spins S_{α,ix,iy}, α = x,y,z"
    S_vec::Array{Float64, 3}
end

"""
Make a completely polarized in-plane configuration with given linear sizes L1 and L2.
"""
function Configuration(L1::Int, L2::Int; S::Float64 = 1.0)
    @assert L1 > 0
    @assert L2 > 0
    ϕ = fill(.0, L1, L2)
    θ = fill(π / 2, L1, L2)
    return Configuration(S, ϕ, θ, S * polar2cartesian(ϕ, θ))
end

"""
Make a disordered configuration of spins S with given linear sizes L1 and L2.
"""
function Configuration(L1::Int, L2::Int, rng::AbstractRNG;
                       S::Float64 = 1.0, inplane::Bool = true)
    @assert L1 > 0
    @assert L2 > 0
    ϕ = 2π * rand(rng, L1, L2)
    θ = inplane ? fill(π / 2, L1, L2) : (π * rand(rng, L1, L2))
    return Configuration(S, ϕ, θ, S * polar2cartesian(ϕ, θ))
end

#
# Heisenberg XXZ model
#

"""
Energy contribution from the Heisenberg XXZ interaction of a single node with its neighbors.
"""
@inline function xxz_energy_term(conf::Configuration, i::Node;
                                 Jpar::Float64, Jperp::Float64)
    L1, L2 = size(conf.S_vec)[2:3]
    S_i = conf.S_vec[:, i...]
    return sum(neighbors(i, L1, L2)) do j
        S_j = conf.S_vec[:, j...]
        Jpar * (S_i[1] * S_j[1] + S_i[2] * S_j[2]) + Jperp * S_i[3] * S_j[3]
    end
end

"Energy contributions from the Heisenberg XXZ terms"
function xxz_energy(conf::Configuration; Jpar::Float64, Jperp::Float64)
    L1, L2 = size(conf.S_vec)[2:3]
    return 0.5 * sum(Base.product(1:L1, 1:L2)) do i
        xxz_energy_term(conf, i, Jpar=Jpar, Jperp=Jperp)
    end
end

#
# RKKY interaction
#

"""
Bare RKKY interaction V(ΔR) as a 3x3 matrix in the space of Cartesian spin components.
λ is the interaction range parameter k_F a0.
"""
function rkky_int(Δi::Node, Vpar::Float64, Vperp::Float64, λ::Float64)
    r = sqrt(Δi[1]^2 - Δi[1] * Δi[2] + Δi[2]^2) # r = |Δi1 a1 + Δi2 a2| / a0
    ϕ = 8π / 3 * (Δi[1] - Δi[2] / 2)
    v = besselj0(λ * r) * bessely0(λ * r) + besselj1(λ * r) * bessely1(λ * r)
    return [ Vpar*cos(ϕ) Vpar*sin(ϕ) 0;
            -Vpar*sin(ϕ) Vpar*cos(ϕ) 0;
             0 0 Vperp] .* v
end

"""
Numerically compute renormalized RKKY interaction by summing over replicas of
the main cluster.
λ is the interaction range parameter k_F a0.
"""
function renorm_rkky_int(i::Node, Vpar::Float64, Vperp::Float64, λ::Float64;
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

"Make the RKKY interaction tensor V_{α1,α2,Δi1,Δi2}"
function make_rkky_int_tensor(L1::Int, L2::Int, Vpar::Float64, Vperp::Float64, λ::Float64;
                              renormalize::Bool = true,
                              maxshells::Int = 100, atol::Float64 = 1e-4)

    V = Array{Float64, 4}(undef, 3, 3, 2 * L1 - 1, 2 * L2 - 1)
    for Δi1 in (1-L1):(L1-1), Δi2 in (1-L2):(L2-1)
        Δi = (Δi1, Δi2)
        Δi == (0, 0) && continue
        if renormalize
            v = renorm_rkky_int(
                Δi, Vpar, Vperp, λ, L1=L1, L2=L2, maxshells=maxshells, atol=atol
            )
        else
            v = rkky_int(Δi, Vpar, Vperp, λ)
        end
        V[:, :, Δi1 + L1, Δi2 + L2] = v
    end
    return V
end

"""
Energy contribution from the RKKY interaction of a single node with all other nodes.
"""
@inline function rkky_energy_term(conf::Configuration, i::Node, V::Array{Float64, 4})
    L1, L2 = size(conf.S_vec)[2:3]
    S_i = conf.S_vec[:, i...]
    return sum(Base.product(1:L1, 1:L2)) do j
        (i == j) && return .0 # No self-interaction
        S_j = conf.S_vec[:, j...]
        Δi1 = i[1] - j[1] + L1
        Δi2 = i[2] - j[2] + L2
        return dot(S_i, V[:, :, Δi1, Δi2], S_j)
    end
end

"Energy contributions from the RKKY terms"
function rkky_energy(conf::Configuration, V::Array{Float64, 4})
    L1, L2 = size(conf.S_vec)[2:3]
    return 0.5 * sum(Base.product(1:L1, 1:L2)) do i
        rkky_energy_term(conf, i, V)
    end
end

end # module MX2RKKY
