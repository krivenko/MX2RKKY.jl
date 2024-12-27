module visualization

using MX2RKKY.lattice: l2c
using MX2RKKY.montecarlo: Configuration, chirality

using PyPlot

const plt = PyPlot
const mpl = plt.matplotlib
plt.pygui(false); plt.svg(true);

export plot_conf

"Visualize a spin configuration and chirality of hexagonal cells"
function plot_conf(conf::Configuration)
    L1, L2 = size(conf)

    xmin, ymax = l2c(0, L2-1)
    xmax, ymin = l2c(L1-1, 0)
    fig = plt.figure(figsize=(0.5 * (xmax-xmin), 0.5 * (ymax-ymin)))

    # Draw hexagons
    hexagon_verts = []
    hexagon_facecolors = []
    for i1 in 0:(L1-2), i2 in 0:(L2-2)
        χ = Int(round(chirality(conf, (i1, i2))))
        hexagon = [l2c(i1, i2),
                   l2c(i1 - 1/3, i2 + 1/3),
                   l2c(i1, i2 + 1),
                   l2c(i1 + 2/3, i2 + 4/3),
                   l2c(i1 + 1, i2 + 1),
                   l2c(i1 + 2/3, i2 + 1/3)]
        push!(hexagon_verts, hexagon)
        push!(hexagon_facecolors, Dict(-1 => "#ffb9b9", 0 => "#ffffff", 1 => "#beecff")[χ])
    end
    hexagon = mpl.collections.PolyCollection(
        hexagon_verts, edgecolors="black", facecolors=hexagon_facecolors, linewidths=0.5
    )
    ax = plt.gca()
    ax.add_collection(hexagon)

    # Draw spins
    for i in eachindex(conf)
        Sx, Sy = 0.5 * conf.S_vec[1:2, i...] ./ conf.S
        x, y = l2c(i...)
        plt.arrow(x - Sx/2, y - Sy/2, Sx, Sy, color="red", head_width=0.1)
    end

    # Add legend
    plt.legend(handles=[
        mpl.patches.Patch(color="#ffb9b9", label=raw"$\chi=-1$"),
        mpl.patches.Patch(color="#ffffff", label=raw"$\chi=0$"),
        mpl.patches.Patch(color="#beecff", label=raw"$\chi=1$")
    ])

    ax.set_xlim(xmin - 1.0, xmax + 1.0)
    ax.set_ylim(ymin - 1.0, ymax + 1.0)
    plt.show()
    return fig
end

end # module visualization
