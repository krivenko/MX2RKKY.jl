"Geometry and graph representation of a triangular lattice cluster"
module lattice

using OffsetArrays: OffsetArray
using StaticArrays: SVector, @SVector

"Integer coordinates of 2D lattice vectors in the basis a1 = (1, 0), a2 = (-1/2, √3/2)"
const LatCoords = CartesianIndex{2}

"Convert lattice coordinates on a triangular lattice to physical coordinates"
l2p(i1::Real, i2::Real) = (i1 - i2 / 2, √3/2 * i2)

"Convert physical coordinates to lattice coordinates on a triangular lattice"
p2l(x::Real, y::Real) = (x + y / √3, 2 * y / √3)

"""
Return relative integer coordinates of the nearest neighbors on the triangular lattice.
The neighbors are listed in the clockwise order.
"""
@inline function neighbor_shifts(::Val{1})::SVector{6, LatCoords}
    return @SVector[LatCoords(1, 0), LatCoords(0, -1), LatCoords(-1, -1),
                    LatCoords(-1, 0), LatCoords(0, 1), LatCoords(1, 1)]
end

"""
Return relative integer coordinates of the next-nearest neighbors on the triangular lattice.
The neighbors are listed in the clockwise order.
"""
@inline function neighbor_shifts(::Val{2})::SVector{6, LatCoords}
    return @SVector[LatCoords(1, -1), LatCoords(-1, -2), LatCoords(-2, -1),
                    LatCoords(-1, 1), LatCoords(1, 2), LatCoords(2, 1)]
end

"""
Return relative integer coordinates of the next-next-nearest neighbors on the triangular
lattice. The neighbors are listed in the clockwise order.
"""
@inline function neighbor_shifts(::Val{3})::SVector{6, LatCoords}
    return @SVector[LatCoords(2, 0), LatCoords(0, -2), LatCoords(-2, -2),
                    LatCoords(-2, 0), LatCoords(0, 2), LatCoords(2, 2)]
end

"""
Given coordinates of a node `i`, return its coordinates on a parallelogram-shaped cluster
of the triangular lattice with periodic boundary conditions.
The shape of the cluster is parametrized by two linear sizes, `L1` and `L2`.
"""
@inline function periodize(i::LatCoords, L1::Int, L2::Int)::LatCoords
    return LatCoords(mod(i[1], L1), mod(i[2], L2))
end

"""
Given coordinates of a node `i`, return coordinates of its neighbors on
a parallelogram-shaped cluster of the triangular lattice with periodic boundary conditions.
The neighbors are listed in the clockwise order.
The shape of the cluster is parametrized by two linear sizes, `L1` and `L2`.
"""
@inline function neighbors(i::LatCoords, L1::Int, L2::Int)::SVector{6, LatCoords}
    return @SVector[periodize(i .+ s, L1, L2) for s in neighbor_shifts(Val(1))]
end

"""
Given coordinates of a node `i`, return coordinates of the nodes belonging to the same
elementary triangle. The nodes belong to a parallelogram-shaped cluster with periodic
boundary conditions and are listed in the clockwise order.
`L1` and `L2` are linear sizes of the cluster.
"""
@inline function triangle_nodes(i::LatCoords, L1::Int, L2::Int)::SVector{3, LatCoords}
    return @SVector[i,
                    LatCoords(i[1], mod(i[2] + 1, L2)),
                    LatCoords(mod(i[1] + 1, L1), mod(i[2] + 1, L2))]
end

"""
Make a 2D-dimensional zero array with axes 0:(L1-1), 0:(L2-1), 0:(L1-1), ...
The optional argument `extra_leading_dims` can be used to prepend more axes.
"""
function LatticeArray(::Type{T}, D::Integer, L1::Integer, L2::Integer;
                      extra_leading_dims::NTuple{D0, Int} = ()) where {T, D0}
    @assert D >= 1
    dims = Iterators.cycle((L1, L2), D)
    leading_offsets = fill(0, length(extra_leading_dims))
    offsets = fill(-1, 2D)
    return OffsetArray(zeros(T, extra_leading_dims..., dims...),
                       leading_offsets..., offsets...)
end

"""
Make a 2D-dimensional array with axes 0:(L1-1), 0:(L2-1), 0:(L1-1), ... by calling a
given function of D LatCoords arguments.
The optional argument `extra_leading_dims` can be used to prepend more axes.
"""
function LatticeArray(f::Function, ::Type{T}, D::Integer, L1::Integer, L2::Integer;
                      extra_leading_dims::NTuple{D0, Int} = ()) where {T, D0}
    @assert D >= 1
    leading_ranges = (1:d for d in extra_leading_dims)
    range1 = Iterators.product(0:(L1 - 1), 0:(L2 - 1))
    a = [f(i...) for i in Iterators.product(leading_ranges..., fill(range1, D)...)]
    leading_offsets = fill(0, length(extra_leading_dims))
    offsets = fill(-1, 2D)
    return OffsetArray(a, leading_offsets..., offsets...)
end

end # module lattice
