using Pkg
function require_package(package_name::String)
    try
        eval(Meta.parse("using $package_name"))
        println("$package_name loaded successfully")
    catch e
        if occursin("Package $package_name not found", string(e))
            println("Installing $package_name...")
            Pkg.add(package_name)
            eval(Meta.parse("using $package_name"))
            println("$package_name installation completed")
        else
            rethrow(e)
        end
    end
end

require_package("Distributions")
require_package("LinearAlgebra")
require_package("LinearAlgebra.BLAS")
require_package("ProgressMeter")
require_package("Random")
require_package("Statistics")
require_package("QuadGK")
require_package("Plots")
require_package("StatsBase")
require_package("DataFrames")
require_package("PrettyTables")
require_package("Printf")
require_package("CSV")
require_package("XLSX")
require_package("ForwardDiff")
require_package("LaTeXStrings")

include("../OnlineCovCPD_utils.jl")

# ===================================================
# Plot the LSS difference with no change point (H0)
# ===================================================
p_dim = 100
k1, k2 = 150, 150
n_total = 2000
N_sims = 200
test_function_type = :linear

trace_diff_results = zeros(N_sims, n_total - k1 - k2);
prog = Progress(N_sims; desc="[Running H0 simulations]", barglyphs=BarGlyphs('=', '=', '>', '.', '.'))
for i in 1:N_sims
    X = randn(p_dim, n_total)
    nu4_est = calculate_kappa(X)
    mu_k_arr, sigma2_k_arr = precompute_moments(; p=p_dim, n=n_total, k1=k1, k2=k2, nu4=nu4_est, function_type=test_function_type)
    trace_diff_results[i, :] = one_step_trace_diff(X=X, k1=k1, k2=k2, mu_k_arr=mu_k_arr, sigma2_k_arr=sigma2_k_arr, function_type=test_function_type).normalized
    next!(prog)
end
finish!(prog)

alpha = 0.05;
m = k1 + k2;
L = n_total - m;
t = collect(0:1:L) / m;

W = cumsum(trace_diff_results, dims=2) / sqrt(m);
W = hcat(zeros(N_sims, 1), W);

default(fontfamily="Palatino",
    linewidth=2,
    grid=true,
    minorticks=true,
    guidefontsize=10,
    tickfontsize=8,
    legendfontsize=8)

p1 = plot(t, W',
    label="",
    xlabel="Time t",
    ylabel="Value",
    title="",
    lw=1,
    linealpha=0.1,
    color=:blue,
    legend=:topleft,
    yformatter=x -> string(round(Int, x))
)

# Boundary 1: rho_2^{-1}(t) = sqrt(t+1) * sqrt(a^2 + ln(t+1))
a = sqrt(-2 * log(alpha))
g_t = sqrt.(t .+ 1) .* sqrt.(a^2 .+ log.(t .+ 1))
plot!(p1, t, g_t,
    label=L"\pm \rho_2^{-1}(t)",
    color=:red,
    linestyle=:dash,
    lw=1.5
)
plot!(p1, t, -g_t, label="", color=:red, linestyle=:dash, lw=2)

# Plot Boundary 2: rho_{1,gamma}^{-1}(t) 
line_colors = [:green, :orange, :purple]

gamma_values = [0.0, 0.25, 0.45]
for (i, gamma) in enumerate(gamma_values)
    cv = get_critical_value(alpha, gamma)
    col = line_colors[i]
    rho_curve = cv .* (1 .+ t) .^ (1 - gamma) .* t .^ gamma

    plot!(p1, t, rho_curve,
        label=L"\pm c_{\alpha}\rho_{1,\gamma}^{-1}(t),\ \gamma = %$(gamma)",
        color=col,
        lw=1.5
    )
    plot!(p1, t, -rho_curve, label="", color=col, lw=2)

    end_y = rho_curve[end]
    annotate!(p1, [(t[end] - 0.5, end_y, text(L"\gamma = %$(gamma)", :left, 9, col))])
end

display(p1)

fig_dir = joinpath(@__DIR__, "fig")
mkpath(fig_dir)
savefig(p1, joinpath(fig_dir, "LSS_diff_BM_plot_H0.pdf"))

# ===================================================
# Plot the LSS difference with a change point (H1)
# ===================================================
p_dim = 100
k1, k2 = 150, 150
change_point = 500
n_total = 700
N_sims = 200
test_function_type = :linear

trace_diff_results = zeros(N_sims, n_total - k1 - k2);
prog = Progress(N_sims; desc="[Running H1 simulations]", barglyphs=BarGlyphs('=', '=', '>', '.', '.'))
for i in 1:N_sims
    X1 = randn(p_dim, change_point)
    X2 = randn(p_dim, n_total - change_point) * 1.2
    X = hcat(X1, X2)
    nu4_est = calculate_kappa(X1)
    mu_k_arr, sigma2_k_arr = precompute_moments(; p=p_dim, n=n_total, k1=k1, k2=k2, nu4=nu4_est, function_type=test_function_type)
    trace_diff_results[i, :] = one_step_trace_diff(X=X, k1=k1, k2=k2, mu_k_arr=mu_k_arr, sigma2_k_arr=sigma2_k_arr, function_type=test_function_type).normalized
    next!(prog)
end
finish!(prog)

alpha = 0.05;
m = k1 + k2;
L = n_total - m;
t = collect(0:1:L) / m;

W = cumsum(trace_diff_results, dims=2) / sqrt(m);
W = hcat(zeros(N_sims, 1), W);

default(fontfamily="Palatino",
    linewidth=2,
    grid=true,
    minorticks=true,
    guidefontsize=10,
    tickfontsize=8,
    legendfontsize=8)

p1 = plot(t, W',
    label="",
    xlabel="Time t",
    ylabel="Value",
    title="",
    lw=1,
    linealpha=0.1,
    color=:blue,
    legend=:topleft,
    yformatter=x -> string(round(Int, x))
)

# Boundary 1: g(t) = sqrt(t+1) * sqrt(a^2 + ln(t+1))
a = sqrt(-2 * log(alpha))
g_t = sqrt.(t .+ 1) .* sqrt.(a^2 .+ log.(t .+ 1))
plot!(p1, t, g_t,
    label=L"\pm \rho_2^{-1}(t)",
    color=:red,
    linestyle=:dash,
    lw=1.5
)
plot!(p1, t, -g_t, label="", color=:red, linestyle=:dash, lw=2)

# Plot Boundary 2: rho(t) 
line_colors = [:green, :orange, :purple]

gamma_values = [0.0, 0.25, 0.45]
for (i, gamma) in enumerate(gamma_values)
    cv = get_critical_value(alpha, gamma)
    col = line_colors[i]
    rho_curve = cv .* (1 .+ t) .^ (1 - gamma) .* t .^ gamma

    plot!(p1, t, rho_curve,
        label=L"\pm c_{\alpha}\rho_{1,\gamma}^{-1}(t),\ \gamma = %$(gamma)",
        color=col,
        lw=1.5
    )
    plot!(p1, t, -rho_curve, label="", color=col, lw=2)

    end_y = rho_curve[end]
end

# Add the true change point
vline!([(change_point-m) / m], color=:black, linestyle=:dot, label="Change point")
annotate!([( (change_point-m)/m, -5.6, Plots.text(L"t^{\star}", :center, 9, :black))])

display(p1)

fig_dir = joinpath(@__DIR__, "fig")
mkpath(fig_dir)
savefig(p1, joinpath(fig_dir, "LSS_diff_BM_plot_H1.pdf"))
