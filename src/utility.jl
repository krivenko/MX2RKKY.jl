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

end # module utility
