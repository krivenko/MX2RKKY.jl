module lattice

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
elementary triangle. The nodes belong to a parallelogram-shaped cluster with periodic
boundary conditions and are listed in the clockwise order.
`L1` and `L2` are linear sizes of the cluster.
"""
function triangle_nodes(i::IntCoords, L1::Int, L2::Int)::Vector{IntCoords}
    i1, i2 = i
    return [i, (i1, mod(i2 + 1, L2)), (mod(i1 + 1, L1), mod(i2 + 1, L2))]
end

end # module lattice
