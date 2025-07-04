module utility

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
