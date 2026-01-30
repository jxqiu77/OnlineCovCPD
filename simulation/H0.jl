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

include("../OnlineCovCPD_utils.jl")

function run_H0_size_test(;
    p_dim::Int,
    n_total::Int,
    k1::Int,
    k2::Int,
    function_type::Symbol,
    alpha::Float64,
    M_sims::Int,
    data_generator::Function,
    weight_functions::Vector{WeightFunctionConfig}
)
    println("--- Starting H0 (Size) Simulation [ONLINE] ---")
    println("Params: p=$p_dim, N=$n_total, k1=$k1, k2=$k2, f=$function_type")
    println("M_sims=$M_sims, alpha=$alpha, weight_functions=$(length(weight_functions))")

    m_start = k1 + k2

    # Precompute critical values for each weight function
    critical_values = Dict{String, Float64}()
    for wf in weight_functions
        if wf.name == "econ"
            critical_values[wf.name] = 1.0
        else
            γ = get(wf.params, :gamma, 0.0)
            critical_values[wf.name] = get_critical_value(alpha, γ)
        end
    end

    # Store detection times for each weight function
    all_detection_times = Dict{String, Vector{Int}}(wf.name => fill(-1, M_sims) for wf in weight_functions)

    prog = Progress(M_sims, 1, "[Running H0 Sim]")
    for rep in 1:M_sims
        Y = data_generator(p_dim, n_total)

        nu4 = calculate_kappa(@view Y[:, 1:m_start])
        mu_k_arr, sigma2_k_arr = precompute_moments(
            p = p_dim,
            n = n_total,
            k1 = k1,
            k2 = k2,
            nu4 = nu4,
            function_type = function_type
        )

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

    # Compute empirical size for each weight function
    results = Dict{String, NamedTuple}()
    for wf in weight_functions
        detection_times = all_detection_times[wf.name]
        # Under H0, any detection at k>0 counts as false alarm
        false_alarms = count(t -> t > 0, detection_times)
        empirical_size = false_alarms / M_sims
        results[wf.name] = (empirical_size = empirical_size, all_detection_times = detection_times)
    end

    return results
end

# === Main Simulation Script ===
function main_H0_simulation()
    Random.seed!(42)

    p_dim = 100
    k1, k2 = 150, 150
    n_total = 1000
    alpha = 0.05
    M_sims = 2000

    weight_functions = [
        WeightFunctionConfig("gamma_0.0", Dict(:gamma => 0.0)),
        WeightFunctionConfig("gamma_0.25", Dict(:gamma => 0.25)),
        WeightFunctionConfig("gamma_0.45", Dict(:gamma => 0.45)),
        WeightFunctionConfig("econ", Dict(:alpha => alpha)),
    ]

    scenarios = [
        (dist_name="Gaussian",  data_gen=(p,n)->randn(p,n)),
        (dist_name="Uniform",   data_gen=(p,n)->rand(Uniform(-sqrt(3.0), sqrt(3.0)), p, n)),
        (dist_name="TDist",     data_gen=(p,n)->rand(TDist(10),p,n)/sqrt(10/8))
    ]
    function_types = [:linear, :log, :mix, :square]

    results_df = DataFrame(
        distribution = String[],
        function_type = String[],
        weight_function = String[],
        empirical_size = Float64[],
        valid_simulations = Int[],
        total_simulations = Int[],
        false_alarms = Int[]
    )

    data_dir = joinpath(@__DIR__, "data")
    mkpath(data_dir)
    csv_filename = joinpath(data_dir, "size.csv")

    for scenario in scenarios
        for f_type in function_types
            println("\n" * "="^80)
            println("RUNNING SCENARIO: [$(scenario.dist_name)] with [f=$(string(f_type))]")
            println("Testing $(length(weight_functions)) weight functions on the same data (ONLINE)...")
            println("="^80)

            sim_results = run_H0_size_test(
                p_dim = p_dim,
                n_total = n_total,
                k1 = k1,
                k2 = k2,
                function_type = f_type,
                alpha = alpha,
                M_sims = M_sims,
                data_generator = scenario.data_gen,
                weight_functions = weight_functions
            )

            for wf in weight_functions
                result = sim_results[wf.name]
                detected = count(t -> t > 0, result.all_detection_times)

                push!(results_df, (
                    distribution = scenario.dist_name,
                    function_type = string(f_type),
                    weight_function = wf.name,
                    empirical_size = result.empirical_size,
                    valid_simulations = M_sims,
                    total_simulations = M_sims,
                    false_alarms = detected
                ))
                println("  $(wf.name): Empirical Size = $(round(result.empirical_size, digits=4))")
            end
        end
        CSV.write(csv_filename, results_df)
    end

    cd(@__DIR__)
    data_dir = joinpath(@__DIR__, "data")
    mkpath(data_dir)
    csv_filename = joinpath(data_dir, "H0_results.csv")
    CSV.write(csv_filename, results_df)
    println("All scenarios' results saved to: $csv_filename")
end

t0 = time()
main_H0_simulation()
println("total wall time = ", round(time() - t0, digits=3), " sec")