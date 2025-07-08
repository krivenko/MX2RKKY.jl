module lattice

using OffsetArrays: OffsetArray, Origin
using StaticArrays: SVector, @SVector

"Integer coordinates of 2D lattice vectors in the basis a1 = (1, 0), a2 = (-1/2, √3/2)"
const IntCoords = Tuple{Int, Int}

"Convert lattice coordinates on a triangular lattice to Cartesian coordinates"
l2c(i1::Real, i2::Real) = (i1 - i2 / 2, √3/2 * i2)

"Convert Cartesian coordinates to lattice coordinates on a triangular lattice"
c2l(x::Real, y::Real) = (x + y / √3, 2 * y / √3)

"""
Return relative integer coordinates of the nearest neighbors on the triangular lattice.
The neighbors are listed in the clockwise order.
"""
@inline function neighbor_shifts(::Val{1})::SVector{6, IntCoords}
    return @SVector[(1, 0), (0, -1), (-1, -1), (-1, 0), (0, 1), (1, 1)]
end

"""
Return relative integer coordinates of the next-nearest neighbors on the triangular lattice.
The neighbors are listed in the clockwise order.
"""
@inline function neighbor_shifts(::Val{2})::SVector{6, IntCoords}
    return @SVector[(1, -1), (-1, -2), (-2, -1), (-1, 1), (1, 2), (2, 1)]
end

"""
Return relative integer coordinates of the next-next-nearest neighbors on the triangular
lattice. The neighbors are listed in the clockwise order.
"""
@inline function neighbor_shifts(::Val{3})::SVector{6, IntCoords}
    return @SVector[(2, 0), (0, -2), (-2, -2), (-2, 0), (0, 2), (2, 2)]
end

"""
Given coordinates of a node `i`, return its coordinates on a parallelogram-shaped cluster
of the triangular lattice with periodic boundary conditions.
The shape of the cluster is parametrized by two linear sizes, `L1` and `L2`.
"""
@inline function periodize(i::IntCoords, L1::Int, L2::Int)::IntCoords
    return (mod(i[1], L1), mod(i[2], L2))
end

"""
Given coordinates of a node `i`, return coordinates of its neighbors on
a parallelogram-shaped cluster of the triangular lattice with periodic boundary conditions.
The neighbors are listed in the clockwise order.
The shape of the cluster is parametrized by two linear sizes, `L1` and `L2`.
"""
@inline function neighbors(i::IntCoords, L1::Int, L2::Int)::SVector{6, IntCoords}
    return @SVector[periodize(i .+ s, L1, L2) for s in neighbor_shifts(Val(1))]
end

"""
Given coordinates of a node `i`, return coordinates of the nodes belonging to the same
elementary triangle. The nodes belong to a parallelogram-shaped cluster with periodic
boundary conditions and are listed in the clockwise order.
`L1` and `L2` are linear sizes of the cluster.
"""
@inline function triangle_nodes(i::IntCoords, L1::Int, L2::Int)::Vector{IntCoords}
    i1, i2 = i
    return [i, (i1, mod(i2 + 1, L2)), (mod(i1 + 1, L1), mod(i2 + 1, L2))]
end

"""
Make a 2D-dimensional zero array with axes 0:(L1-1), 0:(L2-1), 0:(L1-1), ...
"""
function LatticeArray(::Type{T}, D::Integer, L1::Integer, L2::Integer) where T
    @assert D >= 1
    dims = Iterators.cycle((L1, L2), D)
    return OffsetArray(zeros(T, dims...), Origin(0))
end

"""
Make a 2D-dimensional array with axes 0:(L1-1), 0:(L2-1), 0:(L1-1), ... by calling a
given function of D IntCoords arguments.
"""
function LatticeArray(f::Function, ::Type{T}, D::Integer, L1::Integer, L2::Integer) where T
    @assert D >= 1
    range1 = Iterators.product(0:(L1 - 1), 0:(L2 - 1))
    a = [f(i...) for i in Iterators.product(fill(range1, D)...)]
    return OffsetArray(a, Origin(0))
end

end # module lattice
