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
Take a range of non-missing elements of a vector. By default, all non-missing elements
are taken.
"""
function range_of_nonmissing(v::Vector, range::Union{AbstractRange, Nothing} = nothing)
    v_nm = collect(skipmissing(v))
    if isnothing(range)
        range = axes(v_nm, 1)
    end
    return v_nm[range]
end

end # module utility
