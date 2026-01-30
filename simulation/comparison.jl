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
using CSV
using XLSX
using ForwardDiff
using Dates
using PrettyTables
include("../OnlineCovCPD_utils.jl")
# include("meanBootstrap.jl")

using RCall
R"""
library(parallel) 
library(covcp) # Avanesov (2019) method
library(onlineCOV) # Li & Li (2023) method

# library(doParallel)
# registerDoParallel(cores=detectCores())
# Ensure detectCores is available in global environment
# Some R packages (like covcp) may call detectCores() internally without namespace prefix
detectCores <- parallel::detectCores
"""

# ========================================================
# Helper functions
# ========================================================

# ========================================================
# Our method: OnlineCovCPD
# ========================================================
function OnlineCovCPD(X::AbstractMatrix{<:Real};
    # row of X: observations, column of X: variables
    k1::Int, k2::Int, gamma::Real, alpha::Real,
    function_type::Symbol, mu_k_arr::Vector{Float64},
    sigma2_k_arr::Vector{Float64})

    Y = X'
    n_start = k1 + k2
    L = size(Y, 2) - n_start
    @assert length(mu_k_arr) == L
    @assert length(sigma2_k_arr) == L

    st = init_trace_state(Y, k1, k2, function_type)
    critical_value = get_critical_value(alpha, gamma)
    psi = 0.0
    detected_time = -1

    for i in 1:L
        k = n_start + i
        xk = @view Y[:, k]

        diff_raw, _ = update_trace_diff!(st, xk)
        z = (diff_raw - mu_k_arr[i]) / sqrt(sigma2_k_arr[i])

        psi += z

        t = i / n_start
        w = (1 + t)^(gamma - 1) * (t)^(-gamma)
        stat = w * abs(psi) / sqrt(n_start)

        if detected_time == -1 && stat > critical_value
            detected_time = n_start + i
            return (detected=true, detect_time=detected_time)
        end
    end
    return (detected=false, detect_time=-1)
end

# ========================================================
# Avanesov (2019) method
# The code in this section are adapted from the R package "covcp"
# Reference: https://github.com/akopich/covcp
# ========================================================
# 1) prep_stream: same as before (center+scale based on stable segment)
function prep_stream(X::AbstractMatrix{<:Real},
    stable_idx::AbstractVector{<:Integer})

    @rput X stable_idx
    R"""
    Y <- covcp::vectorWiseCovariances(X) # each row is vech(x_i x_i')
    mu0 <- colMeans(Y[stable_idx, , drop = FALSE]) 
    Yc <- scale(Y, center = mu0, scale = FALSE) 

    stable_set <- Yc[stable_idx, , drop = FALSE]  
    vars <- apply(stable_set, 2, var) # variance of each column
    if (any(vars <= 0)) {
        pos <- vars[vars > 0]
        if (length(pos) == 0) stop("All coordinates have non-positive variance in stable_set.")
        vars[vars <= 0] <- min(pos)
    }
    normalizer <- 1 / sqrt(vars)
    Yn <- t(apply(Yc, 1, function(row) row * normalizer)) # each row is vech(x_i x_i') ./ sqrt(vars)
    Yn_stable <- t(apply(stable_set, 1, function(row) row * normalizer)) 
    """

    return (Y=rcopy(R"Yn"), Y_stable=rcopy(R"Yn_stable"), mu0=rcopy(R"mu0"), vars=rcopy(R"vars"))
end

# 2) calibrate_thresholds: same as before
# bootstrap based critical level: 
# x_n^b (\alpha) := inf {x: P^b (B_n^b < x) >= 1-\alpha}

inf_norm(x::Vector) = maximum(abs.(x))
no_pattern(result) = maximum(result.distances)

function calibrate_thresholds(Y_stable::AbstractMatrix{<:Real},
    window_sizes::AbstractVector{<:Integer};
    alpha::Real=0.05,
    N_monitor::Integer,
    bootstrap_iter::Integer=1000)
    crit = mean_bootstrap_based_critical_level(Y_stable, bootstrap_iter, alpha, N_monitor, window_sizes, inf_norm, no_pattern)
    return crit
    # @rput Y_stable window_sizes alpha N_monitor bootstrap_iter
    # R"""
    # crit <- covcp::meanBootstrapBasedCriticalLevel(
    #   stable = Y_stable, # \hat{Z}_i in Section 2.2
    #   iterations = bootstrap_iter,
    #   alpha = alpha,
    #   N = N_monitor,
    #   windowSizes = window_sizes, # window sizes to be tested
    #   parameterDifferenceNorm = covcp::infNorm, # infinity norm
    #   distances2statistic = covcp::noPattern
    # )
    # """
    # return rcopy(R"crit")
end

# 3) standardize full stream once using fixed (mu0, vars)
function standardize_full_stream(X::AbstractMatrix{<:Real}, mu0, vars)
    @rput X mu0 vars
    R"""
    Y_all_raw <- covcp::vectorWiseCovariances(X)
    Y_all_c   <- scale(Y_all_raw, center = mu0, scale = FALSE)
    normalizer <- 1 / sqrt(vars)
    Y_all <- t(apply(Y_all_c, 1, function(row) row * normalizer))
    """
    return rcopy(R"Y_all")
end

# 4) Precompute distance sequences for each window n ONCE using covcp,
#    then do online scanning in Julia without further R calls.
function precompute_sw_distances(Y_all::AbstractMatrix{<:Real},
    window_sizes::Vector{Int})

    @rput Y_all window_sizes

    R"""
    sw_list <- lapply(window_sizes, function(n) {
      sw <- covcp::slidingWindowsDifferenceOfMean(
        data = Y_all / sqrt(n),
        windowSize = n,
        parameterDifferenceNorm = covcp::infNorm
      )
      list(n = n,
           centralPoints = sw$centralPoints,
           distances = sw$distances)
    })
    """

    sw_list = rcopy(R"sw_list")  # Vector of Dict-like objects
    out = Dict{Int,NamedTuple{(:centralPoints, :distances),Tuple{Vector{Int},Vector{Float64}}}}()
    for item in sw_list
        n = Int(item[:n])  # Use Symbol key, not String
        cp = Vector{Int}(item[:centralPoints])
        ds = Vector{Float64}(item[:distances])
        out[n] = (centralPoints=cp, distances=ds)
    end
    return out
end

# 5) online detection:
#    - bootstrap thresholds once
#    - precompute SW distances once per window
#    - scan time m and compare latest centerpoint only
function online_cov_Avanesov2019(X::AbstractMatrix{<:Real};
    window_sizes::Vector{Int}=[20, 30, 50],
    s::Int,
    N_monitor::Int,
    alpha::Real=0.05,
    bootstrap_iter::Int=500)

    N = size(X, 1)
    N < s && error("Need at least s=$s rows, got N=$N.")

    stable_idx = collect(1:s)

    # (A) calibration
    prep = prep_stream(X[1:s, :], stable_idx)
    crit_vals = calibrate_thresholds(prep.Y_stable, window_sizes;
        alpha=alpha, N_monitor=N_monitor,
        bootstrap_iter=bootstrap_iter)

    # Convert crit_vals into a Julia vector aligned with window_sizes
    thr = Vector{Float64}(undef, length(window_sizes))
    for k in eachindex(window_sizes)
        thr[k] = Float64(crit_vals[k])  # relies on rcopy list indexing order
    end

    # (B) standardize full stream once
    Y_all = standardize_full_stream(X, prep.mu0, prep.vars)

    # (C) precompute distances once per window using covcp
    sw = precompute_sw_distances(Y_all, window_sizes)

    # (D) online scan (no more R calls)
    nmax = maximum(window_sizes)
    N < 2nmax && return (detected=false, detect_time=-1)

    for m in (2nmax):N
        for (k, n) in enumerate(window_sizes)
            m < 2n && continue

            cp = sw[n].centralPoints
            ds = sw[n].distances

            # "latest" center point at time m (right window ends at m): t = m - n + 1
            latest_t = m - n + 1

            # centralPoints in covcp correspond to feasible centers; find its index fast:
            # idx = latest_t - cp[1] + 1, if within range.
            idx = latest_t - cp[1] + 1
            if 1 <= idx <= length(cp)
                latest_stat = ds[idx]
                if latest_stat > thr[k]
                    return (detected=true,
                        detect_time=m,
                        window=n,
                        central_point=latest_t,
                        statistic=latest_stat,
                        threshold=thr[k])
                end
            end
        end
    end

    return (detected=false, detect_time=-1)
end

# ========================================================
# Li & Li (2023) method
# The code in this section are adapted from the R package "onlineCOV"
# Reference: https://cran.r-project.org/web/packages/onlineCOV/index.html
# ========================================================
function online_cov_LL23(X::AbstractMatrix{<:Real}; n_start::Int, ARL::Int, H::Int)
    data_training = X[1:n_start, :]
    @rput data_training ARL H n_start
    R"nuisance_result <- nuisance.est(data_training)"
    R"current_window_data <- data_training"

    j = n_start + 1
    decision = 0
    n, p = size(X)
    detected_time_LL23 = n + n_start

    while decision == 0 && j < n
        new_data = reshape(X[j, :], 1, p)
        @rput new_data
        r_result = R"""
        res <- stopping.rule(ARL, H, 
                            nuisance_result$mu.hat, 
                            nuisance_result$M.hat, 
                            nuisance_result$cor.hat, 
                            current_window_data, new_data)
        current_window_data <- res$old.updated
        res
        """
        decision = rcopy(Int, r_result["decision"])
        if decision == 1
            detected_time_LL23 = j
            return (detected=true, detect_time=detected_time_LL23)
        end
        j += 1
    end
    return (detected=false, detect_time=-1)
end

# ========================================================
# Main simulation
# ========================================================
Random.seed!(42)

p, n = 50, 1000
n0 = 200 # change point 
trials = 500 # number of simulation trials

### parameter for LL23 method
H = 100 # window size
ARL = 18_492 # ARL for LL23 (threshold a=3.95, see LL23 paper Sec 4.4, Thm 1)

### parameter for our method: OnlineCovCPD
k1, k2 = 60, 100
alpha, gamma = 0.05, 0.0
critical_value = get_critical_value(alpha, gamma)
function_type = :log

### parameter for Avanesov (2019) method
window_sizes = [20]
s = 100 # size of the stable segment
N_monitor = 1000 # number of monitoring samples
alpha = 0.05 # significance level
bootstrap_iter = 500 # number of bootstrap iterations

### change scenarios
change_scenarios = []
# for sigma2 in [1.1, 1.2, 1.3, 1.4, 1.5]
#     push!(change_scenarios, (type=:inflation, param=sigma2, label="sigma2=$(sigma2)"))
# end
# for rho in [0.1, 0.3, 0.5, 0.7, 0.9]
#     push!(change_scenarios, (type=:ar1, param=rho, label="rho=$(rho)"))
# end
for delta in [2.0, 2.5, 3.0, 3.5, 4.0]
    push!(change_scenarios, (type=:spike, param=delta, label="delta=$(delta)"))
end

### distribution scenarios
dist_scenarios = [
    (dist_name="Gaussian", data_gen=(p, n) -> randn(p, n)),
    # (dist_name="Uniform", data_gen=(p, n) -> rand(Uniform(-sqrt(3.0), sqrt(3.0)), p, n)),
    # (dist_name="TDist", data_gen=(p, n) -> rand(TDist(10), p, n) / sqrt(10 / 8))
]

results_df = DataFrame(
    trial=Int[],
    method=String[],
    detected_time=Int[],
    EDD=Float64[],
    ChangeType=Symbol[],
    ChangeParam=Float64[],
    Distribution=String[],
)

for dist_spec in dist_scenarios
    println("Distribution = $(dist_spec.dist_name)")
    for change_spec in change_scenarios
        println("Sigma = $(change_spec.type) ($(change_spec.param))")

        local Sigma::Matrix{Float64}
        if change_spec.type == :inflation
            Sigma = Matrix(change_spec.param * I, p, p)
        elseif change_spec.type == :ar1
            Sigma = generate_ar1_covariance(p, change_spec.param)
        elseif change_spec.type == :spike
            Sigma = generate_spike_covariance(p, change_spec.param)
        else
            error("Unknown change type")
        end
        Sigma_sqrt = sqrt(Sigma)

        detected_time_our_list = zeros(trials)
        detected_time_LL23_list = zeros(trials)
        detected_time_Avanesov_list = zeros(trials)
        EDD_our_list = zeros(trials)
        EDD_LL23_list = zeros(trials)
        EDD_Avanesov_list = zeros(trials)

        @showprogress "[Running $(change_spec.label)]" for trial in 1:trials
            X1 = dist_spec.data_gen(n0, p)
            X2 = dist_spec.data_gen(n - n0, p) * Sigma_sqrt
            X = [X1; X2] # n x p matrix

            ### our method: OnlineCovCPD
            nu4 = calculate_kappa(X1')
            mu_k_arr, sigma2_k_arr = precompute_moments_complex(
                p=p, n=n, k1=k1, k2=k2, nu4=nu4,
                function_type=function_type, rtol=1e-9
            )

            res = OnlineCovCPD(X; k1=k1, k2=k2, gamma=gamma, alpha=alpha, function_type=function_type, mu_k_arr=mu_k_arr, sigma2_k_arr=sigma2_k_arr)
            detected_time_our_list[trial] = res.detect_time
            EDD_our_list[trial] = res.detect_time - n0
            push!(results_df, (
                    trial=trial,
                    method="Our",
                    detected_time=res.detect_time,
                    EDD=res.detect_time - n0,
                    ChangeType=change_spec.type,
                    ChangeParam=change_spec.param,
                    Distribution=dist_spec.dist_name
                ); promote=true)

            ### LL23 method
            res = online_cov_LL23(X; n_start=Int(n0 / 2), ARL=ARL, H=H)
            detected_time_LL23_list[trial] = res.detect_time
            EDD_LL23_list[trial] = res.detect_time - n0
            push!(results_df, (
                    trial=trial,
                    method="LL23",
                    detected_time=res.detect_time,
                    EDD=res.detect_time - n0,
                    ChangeType=change_spec.type,
                    ChangeParam=change_spec.param,
                    Distribution=dist_spec.dist_name
                ); promote=true)

            ### Avanesov (2019) method
            res = online_cov_Avanesov2019(X;
                window_sizes=window_sizes,
                s=s,
                N_monitor=N_monitor,
                alpha=alpha,
                bootstrap_iter=bootstrap_iter)
            detected_time_Avanesov_list[trial] = res.detect_time
            EDD_Avanesov_list[trial] = res.detect_time - n0
            push!(results_df, (
                    trial=trial,
                    method="A19",
                    detected_time=res.detect_time,
                    EDD=res.detect_time - n0,
                    ChangeType=change_spec.type,
                    ChangeParam=change_spec.param,
                    Distribution=dist_spec.dist_name
                ); promote=true)
        end

        cd(@__DIR__)
        csv_filename = "compare_LL23_results_$(change_spec.type).csv"
        CSV.write(csv_filename, results_df)

        println("="^60)
        println("="^60)
        println("Power (Our): $(mean(EDD_our_list .> 0))")
        println("Power (LL23): $(mean(EDD_LL23_list .> 0))")
        println("Power (A19): $(mean(EDD_Avanesov_list .> 0))")
        println("EDD (Our): $(mean(filter(x -> x > 0, EDD_our_list)))")
        println("EDD (LL23): $(mean(filter(x -> x > 0, EDD_LL23_list)))")
        println("EDD (A19): $(mean(filter(x -> x > 0, EDD_Avanesov_list)))")
        println("="^60)
    end
end
