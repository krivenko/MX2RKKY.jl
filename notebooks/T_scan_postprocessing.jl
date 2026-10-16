# Post-processing routines for temperature scan calculations"

using LinearAlgebra: norm
using Statistics: mean, std
using HDF5
using DataFrames
using CSV

using Makie
using CairoMakie
using LaTeXStrings

"Collect T-scan data from an HDF5 archive"
function collect_from_h5(ar, name)
    T_range = ar["params/T_range"][]
    return [ar["/T$(T)/$(name)"][] for T in T_range if "T$(T)" in keys(ar)]
end

"Check whether T-scan data with a given name is present in an HDF5 archive"
function has_in_h5(ar, name)
    T_range = ar["params/T_range"][]
    T = first(T for T in T_range if "T$(T)" in keys(ar))
    return name in keys(ar["T$(T)"])
end

"""
Return the number of spins and the number of complete elementary triangles (all three
nodes occupied) of the cluster simulated in a T-scan stored in an HDF5 archive.
"""
function cluster_counts(ar)
    L = ar["params/L"][]
    # Archives written before vacancies were supported do not store the counts
    if has_in_h5(ar, "n_spins")
        T_range = ar["params/T_range"][]
        T = first(T for T in T_range if "T$(T)" in keys(ar))
        return ar["T$(T)/n_spins"][], ar["T$(T)/n_triangles"][]
    else
        return L^2, L^2
    end
end

"Copy simulation parameters into metadata of a DataFrame"
function set_params_metadata!(df, params;
                              optional_keys = ("rkky_range_cutoff",
                                               "init_chi",
                                               "n_vacancies",
                                               "disorder_seed"))
    metadata!(df, "L", params["L"], style=:note)
    metadata!(df, "V", params["V"], style=:note)
    metadata!(df, "lambda", params["lambda"], style=:note)
    for key in optional_keys
        if key in keys(params)
            metadata!(df, key, params[key], style=:note)
        end
    end
end

"Plotted DataFrame columns: (column, y-axis label, column with error bars, y-axis scale)"
const plotted_columns = [
    (:E_avg, L"$\langle E\rangle$, per atom", :E_err, identity),
    (:cV, L"$c_V$, per atom", :cV_err, identity),
    (:chi_avg, L"$\langle\chi\rangle$", :chi_err, identity),
    (:susc_chi, L"$\chi_c = \langle(\delta\chi)^2\rangle / T$, per atom",
     :susc_chi_err, identity),
    (:binder_chi, L"$g_c$", :binder_chi_err, identity),
    (:corr_length_chi, L"$\xi_c$", :corr_length_chi_err, log10),
    (:m_abs_at_K_avg, L"$\langle|\mathbf{m}(\mathbf{K})|\rangle$",
     :m_abs_at_K_err, identity),
    (:susc_spin_K, L"$\chi_s = \langle(\delta\vec{m}(\vec{K}))^2\rangle / T$",
     :susc_spin_K_err, identity),
    (:susc_norm_spin_K, L"$\chi_{|s|} = \langle\delta|\vec{m}(\vec{K})|^2\rangle / T$",
     :susc_norm_spin_K_err, identity),
    (:binder_spin_K, L"$g_s$", :binder_spin_K_err, identity),
    (:corr_length_spin_K, L"$\xi_s$", :corr_length_spin_K_err, log10),
    (:cov_chirality_norm_spin_K, L"$\langle\delta\chi\delta|\vec{m}(\vec{K})|\rangle$",
     :cov_chirality_norm_spin_K_err, identity),
    (:cov_chirality_long_spin_K,
     L"$\langle\delta\chi \delta \vec{m}(\vec{K})\frac{\vec{m}(\vec{K})}{|\vec{m}(\vec{K})|}\rangle$",
     :cov_chirality_long_spin_K_err, identity)
]

"Set plotting attributes of all columns of a T-scan DataFrame"
function set_colmetadata!(df)
    for name in propertynames(df)
        name == :T && continue
        colmetadata!(df, name, "plot", false, style=:note)
    end
    for (name, ylabel, errors, yscale) in plotted_columns
        hasproperty(df, name) || continue
        colmetadata!(df, name, "plot", true, style=:note)
        colmetadata!(df, name, "ylabel", ylabel, style=:note)
        colmetadata!(df, name, "yscale", yscale, style=:note)
        if hasproperty(df, errors)
            colmetadata!(df, name, "errors", errors, style=:note)
        end
    end
end

"""
Extract T-scan data from an HDF5 archive and convert into a DataFrame.
Extensive quantities are divided by the number of spins (chiral susceptibility by the
number of complete elementary triangles).
"""
function make_observables_df(ar)
    params = read(ar, "params")
    T_range = [T for T in params["T_range"] if "T$(T)" in keys(ar)]
    N_s, N_t = cluster_counts(ar)

    df = DataFrame("T" => T_range)
    set_params_metadata!(df, params)

    # Energy
    insertcols!(df,
        "E_min" => collect_from_h5(ar, "min_energy") / N_s,
        "E_avg" => collect_from_h5(ar, "avg_energy") / N_s,
        "E_var" => collect_from_h5(ar, "var_energy") / N_s,
        "E_err" => collect_from_h5(ar, "err_energy") / N_s
    )

    # Specific heat
    insertcols!(df, "cV" => collect_from_h5(ar, "specific_heat") / N_s)

    # Average chirality
    insertcols!(df,
        "chi_avg" => collect_from_h5(ar, "avg_chirality"),
        "chi_var" => collect_from_h5(ar, "var_chirality"),
        "chi_err" => collect_from_h5(ar, "err_chirality"),
    )

    # Chirality susceptibility
    insertcols!(df, "susc_chi" => collect_from_h5(ar, "susc_chirality") / N_t)

    # Chirality Binder cumulant
    insertcols!(df, "binder_chi" => collect_from_h5(ar, "binder_chirality"))

    # Chirality correlation length
    insertcols!(df,
        "corr_length_chi" => abs.(collect_from_h5(ar, "corr_length_chirality"))
    )

    # Average norm of the magnetization at K-point
    if has_in_h5(ar, "avg_norm_spin_K")
        insertcols!(df, "m_abs_at_K_avg" => collect_from_h5(ar, "avg_norm_spin_K"),
                        "m_abs_at_K_var" => collect_from_h5(ar, "var_spin_K"),
                        "m_abs_at_K_err" => collect_from_h5(ar, "err_spin_K")
        )
    end

    # Magnetization at K-point: susceptibility
    insertcols!(df, "susc_spin_K" => collect_from_h5(ar, "susc_spin_K") / N_s)

    # Norm of magnetization at K-point: susceptibility
    if has_in_h5(ar, "susc_norm_spin_K")
        insertcols!(df,
            "susc_norm_spin_K" => collect_from_h5(ar, "susc_norm_spin_K") / N_s
        )
    end

    # Magnetization at K-point: Binder cumulant
    insertcols!(df, "binder_spin_K" => collect_from_h5(ar, "binder_spin_K"))

    # Magnetization at K-point: Correlation length
    insertcols!(df,
        "corr_length_spin_K" => abs.(collect_from_h5(ar, "corr_length_spin_K"))
    )

    # Covariance chirality -- norm of the magnetization at K-point
    if has_in_h5(ar, "cov_chirality_norm_spin_K")
        insertcols!(df, "cov_chirality_norm_spin_K" =>
                        norm.(collect_from_h5(ar, "cov_chirality_norm_spin_K")))
    end

    # Covariance chirality -- longitudinal fluctuations of spin at K-point
    if has_in_h5(ar, "cov_chirality_long_spin_K")
        insertcols!(df, "cov_chirality_long_spin_K" =>
                        norm.(collect_from_h5(ar, "cov_chirality_long_spin_K")))
    end

    set_colmetadata!(df)
    return df
end

#
# Averaging over disorder realizations
#

"Mean of samples taken over disorder realizations and its standard error"
function disorder_mean(samples::Vector{Float64})
    n = length(samples)
    return mean(samples), (n > 1) ? std(samples) / sqrt(n) : NaN
end

"""
Evaluate `f` on means of samples taken over disorder realizations and estimate the error
of the result using the jackknife method.
"""
function disorder_jackknife(f, samples::Vector{Float64}...)
    n = length(first(samples))
    value = f(mean.(samples)...)
    n == 1 && return value, NaN
    leave_one_out = Vector{Float64}(undef, n)
    for k in 1:n
        means_k = [(sum(s) - s[k]) / (n - 1) for s in samples]
        leave_one_out[k] = f(means_k...)
    end
    loo_mean = mean(leave_one_out)
    return value, sqrt((n - 1) / n * sum((leave_one_out .- loo_mean) .^ 2))
end

"""
Extract T-scan data from several HDF5 archives, which differ only in the realization of the
vacancy pattern, average over the realizations and convert into a DataFrame.

Energy, specific heat, average chirality and magnetization, susceptibilities and
covariances are averaged over the realizations. Their errors are standard errors of the
mean computed from the spread between the realizations.

Binder cumulants and correlation lengths are computed from thermal moments averaged over
the realizations, e.g. g_s = 2 - [<m^4>] / [<m^2>]^2. Their errors are jackknife estimates
over the realizations.

With a single realization, all errors are `NaN`.
"""
function make_disorder_averaged_df(ars)
    @assert !isempty(ars)
    params = [read(ar, "params") for ar in ars]

    # Archives must differ only in the PRNG seeds and the vacancy pattern
    seed_keys = ("seed", "disorder_seed", "occupied")
    for p in params[2:end]
        @assert Set(keys(p)) == Set(keys(params[1])) "Parameter sets differ"
        for key in keys(p)
            key in seed_keys && continue
            @assert p[key] == params[1][key] "Parameter '$(key)' differs"
        end
    end
    disorder_seeds = [p["disorder_seed"] for p in params]
    @assert allunique(disorder_seeds) "Disorder realizations must be distinct"

    L = params[1]["L"]
    T_range = [T for T in params[1]["T_range"] if all("T$(T)" in keys(ar) for ar in ars)]
    n_dis = length(ars)
    N_counts = [cluster_counts(ar) for ar in ars]

    # Thermal averages for each realization: name => n_T x n_dis matrix
    samples = Dict{String, Matrix{Float64}}()
    function sample!(name, f)
        a = Matrix{Float64}(undef, length(T_range), n_dis)
        for (r, ar) in enumerate(ars)
            N_s, N_t = N_counts[r]
            for (n, T) in enumerate(T_range)
                a[n, r] = f(ar["T$(T)"], N_s, N_t)
            end
        end
        samples[name] = a
    end

    sample!("E_min", (g, N_s, N_t) -> g["min_energy"][] / N_s)
    sample!("E_avg", (g, N_s, N_t) -> g["avg_energy"][] / N_s)
    sample!("cV", (g, N_s, N_t) -> g["specific_heat"][] / N_s)
    sample!("chi_avg", (g, N_s, N_t) -> g["avg_chirality"][])
    sample!("susc_chi", (g, N_s, N_t) -> g["susc_chirality"][] / N_t)
    sample!("m_abs_at_K_avg", (g, N_s, N_t) -> g["avg_norm_spin_K"][])
    sample!("susc_spin_K", (g, N_s, N_t) -> g["susc_spin_K"][] / N_s)
    sample!("susc_norm_spin_K", (g, N_s, N_t) -> g["susc_norm_spin_K"][] / N_s)
    sample!("cov_chirality_norm_spin_K",
            (g, N_s, N_t) -> norm(g["cov_chirality_norm_spin_K"][]))
    sample!("cov_chirality_long_spin_K",
            (g, N_s, N_t) -> norm(g["cov_chirality_long_spin_K"][]))

    # Thermal moments <χ^2>, <χ^4>, <|χ(q_min)|^2>, <|m(K)|^2>, <|m(K)|^4>,
    # <|m(K + q_min)|^2>
    sample!("chi2", (g, N_s, N_t) -> g["chirality_sums"][][2] / g["count"][])
    sample!("chi4", (g, N_s, N_t) -> g["chirality_sums"][][4] / g["count"][])
    sample!("chi_q_min_2",
            (g, N_s, N_t) -> real(g["chirality_at_q_min_2_sum"][]) / g["count"][])
    sample!("m2",
            (g, N_s, N_t) -> g["magnetization_at_K_norm_sums"][][2] / g["count"][])
    sample!("m4",
            (g, N_s, N_t) -> g["magnetization_at_K_norm_sums"][][4] / g["count"][])
    sample!("m_q_min_2",
            (g, N_s, N_t) -> g["magnetization_at_K_q_min_norm_sums"][][2] /
                             g["count"][])

    df = DataFrame("T" => T_range)
    set_params_metadata!(df, params[1],
                         optional_keys = ("rkky_range_cutoff", "init_chi", "n_vacancies"))
    metadata!(df, "n_realizations", n_dis, style=:note)
    metadata!(df, "disorder_seeds", disorder_seeds, style=:note)

    insertcols!(df, "E_min" => [minimum(row) for row in eachrow(samples["E_min"])])

    for name in ("E_avg", "cV", "chi_avg", "susc_chi", "m_abs_at_K_avg",
                 "susc_spin_K", "susc_norm_spin_K",
                 "cov_chirality_norm_spin_K", "cov_chirality_long_spin_K")
        values = Vector{Float64}(undef, length(T_range))
        errors = Vector{Float64}(undef, length(T_range))
        for n in eachindex(T_range)
            values[n], errors[n] = disorder_mean(samples[name][n, :])
        end
        error_name = replace(name, r"_avg$" => "") * "_err"
        insertcols!(df, name => values, error_name => errors)
    end

    binder_chi(chi2, chi4) = 3 / 2 * (1 - chi4 / (2 * chi2 ^ 2))
    binder_spin(m2, m4) = 2.0 - m4 / m2 ^ 2
    corr_length(a2, a_q_min_2) = abs(sqrt(complex(a2 / a_q_min_2 - 1))) / (2 * sin(π / L))

    for (name, f, args) in (("binder_chi", binder_chi, ("chi2", "chi4")),
                            ("corr_length_chi", corr_length, ("chi2", "chi_q_min_2")),
                            ("binder_spin_K", binder_spin, ("m2", "m4")),
                            ("corr_length_spin_K", corr_length, ("m2", "m_q_min_2")))
        values = Vector{Float64}(undef, length(T_range))
        errors = Vector{Float64}(undef, length(T_range))
        for n in eachindex(T_range)
            arg_samples = [samples[arg][n, :] for arg in args]
            values[n], errors[n] = disorder_jackknife(f, arg_samples...)
        end
        insertcols!(df, name => values, name * "_err" => errors)
    end

    set_colmetadata!(df)
    return df
end

"Make a LaTeX legend label from metadata of a T-scan DataFrame"
function df_label(df)
    L = metadata(df, "L")
    V = metadata(df, "V")
    lambda = metadata(df, "lambda")
    R = metadata(df, "rkky_range_cutoff")
    if R == 0
        label = "V=$(V),\\ k_F a_0 = $(lambda),\\ R_V=\\infty,\\ L=$(L)"
    else
        label = "V=$(V),\\ k_F a_0 = $(lambda),\\ R_V=$(R),\\ L=$(L)"
    end
    if "init_chi" in keys(metadata(df))
        init_chi = metadata(df, "init_chi")
        label *= ",\\ \\chi_0 = $(init_chi)"
    end
    if get(metadata(df), "n_vacancies", 0) > 0
        label *= ",\\ N_{vac} = $(metadata(df, "n_vacancies"))"
        if "n_realizations" in keys(metadata(df))
            label *= ",\\ N_{dis} = $(metadata(df, "n_realizations"))"
        end
    end
    return latexstring(label)
end

"Plot T-scan data stored in a DataFrame"
function plot_observables(dfs::Vector{DataFrame}; figsize=(1000, 2000), pdfname)
    all_cols = union(keys.(eachcol.(dfs))...)
    plot_col = name -> (name != :T) && any([hasproperty(df, name) && colmetadata(df, name, "plot") for df in dfs])
    nplots = count(plot_col, all_cols)

    T = dfs[1].T

    fig = Figure(size=figsize)
    plt_i = 1
    for name in all_cols
        !plot_col(name) && continue
        ax = Axis(fig[plt_i, 1],
                  xlabel = (plt_i == nplots) ? L"$T/J$" : "",
                  ylabel = colmetadata(dfs[1], name, "ylabel"),
                  yscale = colmetadata(dfs[1], name, "yscale", identity),
                  xticklabelsvisible = (plt_i == nplots),
                  xticksvisible = (plt_i == nplots),
                  xticks = LinearTicks(20))

        ymin = Inf
        ymax = -Inf
        for df in dfs
            hasproperty(df, name) || continue
            data = df[!, name]

            lines!(ax, df.T, data, linewidth=1.0, label=df_label(df))
            if "errors" in keys(colmetadata(df, name))
                errors = df[!, colmetadata(df, name, "errors")]
                errorbars!(ax, df.T, data, errors, linewidth=1.0)
            end

            ymin = min(ymin, minimum(data))
            ymax = max(ymax, maximum(data))
        end
        xlims!(first(dfs[1].T), last(dfs[1].T))
        Δy = ymax - ymin
        ylims!(ymin, ymax + 0.1Δy)

        Legend(fig[plt_i, 2], ax)
        plt_i += 1
    end

    # General plot setup
    rowgap!(fig.layout, 0)
    if pdfname !== nothing
        save(pdfname, fig)
    end
    fig
end

"Export T-scan data stored in a DataFrame into a CSV file"
function export_observables(df::DataFrame)
    L = metadata(df, "L")
    V = metadata(df, "V")
    lambda = metadata(df, "lambda")
    R = metadata(df, "rkky_range_cutoff")
    filename = "RKKY_T_scan_V$(V)_lambda$(lambda)_R$(R)_L$(L)"
    n_vacancies = get(metadata(df), "n_vacancies", 0)
    if n_vacancies > 0
        filename *= "_nvac$(n_vacancies)"
        if "n_realizations" in keys(metadata(df))
            filename *= "_ndis$(metadata(df, "n_realizations"))"
        else
            filename *= "_dseed$(metadata(df, "disorder_seed"))"
        end
    end
    CSV.write(filename * ".dat", df, delim="    ");
end;
