"Visualization (plotting) functions"
module visualization

using CairoMakie
using Makie.GeometryBasics: Polygon, Point2f

using MX2RKKY.lattice: l2p, LatCoords
using MX2RKKY.montecarlo: Configuration, chirality, isoccupied, is_complete_triangle

export plot_conf

"""
Visualize a spin configuration and chirality of hexagonal cells.
Vacant sites are shown as hollow circles and cells with incomplete triangles in grey.
"""
function plot_conf(conf::Configuration; hideticks=false)
    L1, L2 = size(conf)

    xmin, ymax = l2p(0, L2-1)
    xmax, ymin = l2p(L1-1, 0)
    fig = Figure(size = 50 .* (0.5 * (xmax-xmin), 0.5 * (ymax-ymin)), fontsize = 12 * 4/3)
    ax = Axis(fig, xgridvisible = false, ygridvisible = false)
    hideticks && hidedecorations!(ax)

    # Draw hexagons
    hexagons = Polygon[]
    hexagon_facecolors = []
    for i1 in 0:(L1-2), i2 in 0:(L2-2)
        i = LatCoords(i1, i2)
        hexagon = [l2p(i1, i2),
                   l2p(i1 - 1/3, i2 + 1/3),
                   l2p(i1, i2 + 1),
                   l2p(i1 + 2/3, i2 + 4/3),
                   l2p(i1 + 1, i2 + 1),
                   l2p(i1 + 2/3, i2 + 1/3)]
        push!(hexagons, Polygon(Point2f.(hexagon)))
        if is_complete_triangle(conf, i)
            χ = Int(round(chirality(conf, i)))
            push!(hexagon_facecolors,
                  Dict(1 => "#ffb9b9", 0 => "#ffffff", -1 => "#beecff")[χ])
        else
            push!(hexagon_facecolors, "#d0d0d0")
        end
    end
    poly!(hexagons, strokecolor=:black, strokewidth=0.75, color=hexagon_facecolors)

    # Draw spins
    arrow_centers = Point2f[]
    arrow_directions = Point2f[]
    vacancies = Point2f[]
    for i in CartesianIndices(conf)
        if isoccupied(conf, i)
            push!(arrow_centers, Point2f(l2p(i[1], i[2])))
            push!(arrow_directions, Point2f(0.75 .* (cos(conf[i]), sin(conf[i]))))
        else
            push!(vacancies, Point2f(l2p(i[1], i[2])))
        end
    end
    arrows2d!(arrow_centers, arrow_directions,
              align=:center, color=:red, shaftwidth=1, tipwidth=4, tiplength=5)
    scatter!(vacancies, marker=:circle, markersize=8,
             color=:transparent, strokecolor=:black, strokewidth=1)

    # Add legend
    axislegend(ax,
               [PolyElement(color="#ffb9b9"),
                PolyElement(color="#ffffff"),
                PolyElement(color="#beecff"),
                PolyElement(color="#d0d0d0")],
               [L"\chi=1", L"\chi=0", L"\chi=-1", L"\mathrm{incomplete}"],
               patchsize=(20, 20),
               rowgap=10)

    xlims!(ax, (xmin - 1.0, xmax + 1.0))
    ylims!(ax, (ymin - 1.0, ymax + 1.0))
    fig[1, 1] = ax
    return fig
end

end # module visualization
