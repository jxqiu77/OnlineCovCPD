using Distributions
using LinearAlgebra
using LinearAlgebra.BLAS
using ProgressMeter
using Random
using Statistics
using QuadGK
using Plots
using StatsBase
using DataFrames
using PrettyTables
using Printf
using Plots
using CSV
using XLSX
using ForwardDiff
using Dates

cd(@__DIR__)
include("../OnlineCovCPD_utils.jl")

function select_top_volatility_stocks(df_log::DataFrame, n=30)
    # 1. Remove timestamp column
    numeric_df = select(df_log, Not(:timestamp))
    tickers = names(numeric_df)

    # 2. Compute the standard deviation (volatility) for each stock
    # Higher volatility usually means more information
    volatilities = std.(eachcol(numeric_df))

    # 3. Create a temporary table to sort
    stats = DataFrame(Ticker=tickers, Volatility=volatilities)
    sort!(stats, :Volatility, rev=true) # Sort in descending order

    # 4. Select top n stocks
    selected_tickers = stats.Ticker[1:n]

    # 5. Return filtered data (with timestamp column)
    cols_to_keep = [:timestamp; Symbol.(selected_tickers)]
    return select(df_log, cols_to_keep), selected_tickers
end

########################################################
df_log_returns = CSV.read("sp500_log_returns.csv", DataFrame);
df_log_returns_selected, _ = select_top_volatility_stocks(df_log_returns, 30);

df_full = CSV.read("sp500_full.csv", DataFrame);
size(df_full)
after_date = Date("2019-09-01")
df_log_returns_selected = filter(row -> Date(row.timestamp) >= after_date, df_log_returns_selected)
mat_log_returns = Matrix(df_log_returns_selected[:, 2:end]);

df_log_returns = DataFrame(mat_log_returns, names(df_log_returns_selected)[2:end])
df_log_returns.timestamp = df_log_returns_selected.timestamp

df_log_returns[:,end][81]

n, p = size(mat_log_returns)
k1 = 40;
k2 = 40;
function_type = :log;
X1 = mat_log_returns[1:k1, :];
nu4 = calculate_kappa(X1')

gamma = 0.00
alpha = 0.05
critical_value = get_critical_value(alpha, gamma)

mu_k_arr, sigma2_k_arr = precompute_moments(p=p, n=n, k1=k1, k2=k2, nu4=nu4, function_type=function_type)

Y = mat_log_returns[1:n, :]'
m_start = k1 + k2
L = size(Y, 2) - m_start
@assert length(mu_k_arr) == L
@assert length(sigma2_k_arr) == L

psi = 0.0
detected_time_our = -1

st = init_trace_state(Y, k1, k2, function_type)
for i in 1:L
    k = m_start + i
    m = k - k1
    xk = @view Y[:, k]

    diff_raw, _ = update_trace_diff!(st, xk)
    z = (diff_raw - mu_k_arr[i]) / sqrt(sigma2_k_arr[i])

    psi += z

    t = i / m_start
    w = (1 + t)^(gamma - 1) * (t)^(-gamma)
    stat = w * abs(psi) / sqrt(m_start)

    if detected_time_our == -1 && stat > critical_value
        detected_time_our = m_start + i
        break
    end
end

cp_idx = detected_time_our
cp_date = Date(df_log_returns_selected[cp_idx, :timestamp])
println("Detected change point index: $cp_idx")
println("Detected change point: $cp_date")

########################################################
my_cmap = cgrad(:RdBu, rev=true);
function plot_heatmap(sub_df; colorbar=false, title_str="")
    mat = Matrix(select(sub_df, Not(:timestamp)))
    mat = cov(mat)
    limit = quantile(vec(abs.(mat)), 0.99)
    
    heatmap(
        mat,
        c=my_cmap,
        clims=(-limit, limit),
        yflip=true,
        axis=nothing,
        colorbar=colorbar,
        title=title_str,
        titlefontsize=10,
        fontfamily="Palatino"
    )
end

start_data1 = Date("2019-09-01")
end_data1 = cp_date - Day(1)
duration = end_data1 - start_data1
sub_df1 = filter(row -> start_data1 <= row.timestamp <= end_data1, df_log_returns)
title_str1 = "($(Dates.format(start_data1, dateformat"yyyy-u-dd")) ~ $(Dates.format(end_data1, dateformat"yyyy-u-dd")))"
p1 = plot_heatmap(sub_df1, colorbar=false, title_str=title_str1)

start_data2 = cp_date
end_data2 = start_data2 + duration
sub_df2 = filter(row -> start_data2 <= row.timestamp <= end_data2, df_log_returns)
title_str2 = "($(Dates.format(start_data2, dateformat"yyyy-u-dd")) ~ $(Dates.format(end_data2, dateformat"yyyy-u-dd")))"
p2 = plot_heatmap(sub_df2, colorbar=true, title_str=title_str2)

separator = plot(
    [0, 0], [0, 0.4],
    linestyle = :dash, color = :red,
    linewidth = 2,
    label = "",
    framestyle = :none, axis = nothing, grid = false,
    xlims = (-1, 1), ylims = (0, 1)
)
plot!(separator,
    [0, 0], [0.6, 1.0],
    linestyle = :dash, color = :red,
    linewidth = 2,
    label = ""
)
annotate!(separator, 0, 0.5, Plots.text("⚡", 28, :gold, :center))

l = @layout [a{0.44w} b{0.02w} c{0.54w}]
final_plot = plot(
    [p1, separator, p2]..., 
    layout = l,
    size = (800, 350),
    margin = 3Plots.mm
)

display(final_plot)
savefig(final_plot, "sp500_cov_heatmap.pdf")
