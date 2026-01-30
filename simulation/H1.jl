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

include("../OnlineCovCPD_utils.jl")

function run_H1_power_test(;
    p_dim::Int,
    n_total::Int,
    k1::Int,
    k2::Int,
    k_star_change::Int,
    function_type::Symbol,
    alpha::Float64,
    M_sims::Int,
    data_generator::Function,
    Sigma::Matrix{Float64},
    weight_functions::Vector{WeightFunctionConfig}
)
    # critical values per weight
    critical_values = Dict{String, Float64}()
    for wf in weight_functions
        if wf.name == "log"
            critical_values[wf.name] = 1.0
        else
            γ = get(wf.params, :gamma, 0.0)
            critical_values[wf.name] = get_critical_value(alpha, γ)
        end
    end

    all_detection_times = Dict{String, Vector{Int}}(wf.name => fill(-1, M_sims) for wf in weight_functions)

    # prog = Progress(M_sims, 1, "[Running $M_sims simulations]")

    Sigma_sqrt = sqrt(Sigma)

    m_start = k1 + k2

    prog = Progress(M_sims;
        desc="Running $M_sims simulations",
        barglyphs=BarGlyphs('=', '=', '>', '.', '.')
    )
    for rep in 1:M_sims
        Z = data_generator(p_dim, n_total)
        Y = Matrix{Float64}(undef, p_dim, n_total)
        # pre-change data
        Y[:, 1:k_star_change] = Z[:, 1:k_star_change]
        nu4 = calculate_kappa(@view Y[:, 1:m_start])
        mu_k_arr, sigma2_k_arr = precompute_moments(
            p = p_dim,
            n = n_total,
            k1 = k1,
            k2 = k2,
            nu4 = nu4,
            function_type = function_type
        )
        # post-change data
        if k_star_change < n_total
            Y[:, (k_star_change + 1):n_total] = Sigma_sqrt * Z[:, (k_star_change + 1):n_total]
        end

        # --- Online detection ---
        det = online_detect_all_weights(
            Y;
            k1 = k1,
            k2 = k2,
            mu_k_arr = mu_k_arr,
            sigma2_k_arr = sigma2_k_arr,
            function_type = function_type,
            weight_functions = weight_functions,
            critical_values = critical_values
        )

        for wf in weight_functions
            all_detection_times[wf.name][rep] = det[wf.name]
        end

        next!(prog)
    end
    finish!(prog)

    # Summaries per weight
    summary_results = Dict{String, NamedTuple}()
    for wf in weight_functions
        times = all_detection_times[wf.name]

        # Power: detected after change
        detected = filter(t -> t > 0, times)
        power = length(detected) / M_sims

        # EDD: delay conditional on detection after change
        true_detections = filter(t -> t >= k_star_change, times)
        EDD = isempty(true_detections) ? NaN : mean(true_detections .- k_star_change)

        summary_results[wf.name] = (power = power, EDD = EDD)
    end

    return summary_results
end

function main_H1_simulation()
    Random.seed!(42)

    p_dim = 100 # dimension of the data
    n_total = 1000 # sample size for the whole stream
    k1 = 150 # sample size to form S1
    k2 = 150 # sample size to form initial S2
    alpha = 0.05 # significance level
    M_sims = 2000 # trials per scenario

    weight_functions = [
        WeightFunctionConfig("gamma_0.0", Dict(:gamma => 0.0)),
        WeightFunctionConfig("gamma_0.25", Dict(:gamma => 0.25)),
        WeightFunctionConfig("gamma_0.45", Dict(:gamma => 0.45)),
        WeightFunctionConfig("log", Dict(:alpha => alpha)),
    ]

    k_star_list = [350, 450, 550]
    func_type_list = [:linear, :log, :mix]
    dist_scenarios = [
        (dist_name="Gaussian",  data_gen=(p,n)->randn(p,n)),
        (dist_name="Uniform",   data_gen=(p,n)->rand(Uniform(-sqrt(3.0), sqrt(3.0)), p, n)),
        (dist_name="TDist",     data_gen=(p,n)->rand(TDist(10),p,n)/sqrt(10/8))
    ]
    change_scenarios = []
    for sigma2 in [1.1, 1.2, 1.3, 1.4, 1.5]
        push!(change_scenarios, (type=:inflation, param=sigma2, label="sigma2=$(sigma2)"))
    end
    for rho in [0.1, 0.3, 0.5, 0.7, 0.9]
        push!(change_scenarios, (type=:ar1, param=rho, label="rho=$(rho)"))
    end
    for delta in [2.0, 2.5, 3.0, 3.5, 4.0]
        push!(change_scenarios, (type=:spike, param=delta, label="delta=$(delta)"))
    end

    results_df = DataFrame(
        Distribution = String[],
        FunctionType = Symbol[],
        WeightFunction = String[],
        K_Star = Int[],
        ChangeType = Symbol[],
        ChangeParam = Float64[],
        Power = Float64[],
        EDD = Float64[]
    )

    println("="^60)
    println("STARTING H1 SIMULATION SUITE [ONLINE]")
    println("Total scenarios to run: $(length(k_star_list) * length(func_type_list) * length(dist_scenarios) * length(change_scenarios))")
    println("="^60)

    for dist_spec in dist_scenarios
        for f_type in func_type_list
            for k_star in k_star_list
                for change_spec in change_scenarios
                    println("\n------------- RUNNING -------------")
                    println("Dist: $(dist_spec.dist_name)")
                    println("f(x) = $f_type, k* = $k_star")
                    println("Sigma = $(change_spec.type) ($(change_spec.param))")

                    local Sigma::Matrix{Float64}
                    if change_spec.type == :inflation
                        Sigma = Matrix(change_spec.param * I, p_dim, p_dim)
                    elseif change_spec.type == :ar1
                        Sigma = generate_ar1_covariance(p_dim, change_spec.param)
                    elseif change_spec.type == :spike
                        Sigma = generate_spike_covariance(p_dim, change_spec.param)
                    else
                        error("Unknown change type")
                    end

                    sim_result_dict = run_H1_power_test(
                        p_dim = p_dim,
                        n_total = n_total,
                        k1 = k1,
                        k2 = k2,
                        k_star_change = k_star,
                        function_type = f_type,
                        alpha = alpha,
                        M_sims = M_sims,
                        data_generator = dist_spec.data_gen,
                        Sigma = Sigma,
                        weight_functions = weight_functions
                    )

                    for wf in weight_functions
                        res = sim_result_dict[wf.name]
                        push!(results_df, (
                            dist_spec.dist_name,
                            f_type,
                            wf.name,
                            k_star,
                            change_spec.type,
                            change_spec.param,
                            res.power,
                            res.EDD
                        ))
                        @printf("  %-12s: Power=%.4f, EDD=%.2f\n", wf.name, res.power, res.EDD)
                    end
                end
            end
        end
    end

    println("\n" * "="^60)
    println("H1 SIMULATION SUITE COMPLETE [ONLINE]")
    println("="^60)
    
    cd(@__DIR__)
    data_dir = joinpath(@__DIR__, "data")
    mkpath(data_dir)
    csv_filename = joinpath(data_dir, "H1_results.csv")
    CSV.write(csv_filename, results_df)
    println("All scenarios' results saved to: $csv_filename")
end

println("Starting the full H1 (Power/EDD) simulation suite [ONLINE]...")
main_H1_simulation()
