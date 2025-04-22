module utility

"Convert spherical coordinates (r, θ, φ) to Cartesian components (x, y, z)"
function sph2cart(r::Float64, θ::Float64, ϕ::Float64)::Tuple{Float64, Float64, Float64}
    return (r * sin(θ) * cos(ϕ), r * sin(θ) * sin(ϕ), r * cos(θ))
end

"Cast a given angle ϕ into the range [-π; π]"
function clamp_angle(ϕ::Float64)
    if ϕ > π
        return ϕ - 2π
    elseif ϕ < -π
        return ϕ + 2π
    else
        return ϕ
    end
end

"""
Iterate along the last axis of a given array and discard all slices containing at least one
missing value. Then take a range over the last axis.
By default, all non-missing elements are taken.
"""
function range_of_nonmissing(a::AbstractArray,
                             range::Union{AbstractRange, Nothing} = nothing)
    a_nm = stack(filter(s -> !any(ismissing.(s)), eachslice(a, dims=ndims(a))))
    if isnothing(range)
        range = axes(a_nm, ndims(a_nm))
    end
    return selectdim(a_nm, ndims(a_nm), range)
end

end # module utility
