# =================================================================
# Simulate the critical value for the online change-point detection
# =================================================================

# Weight function: rho(t) = (1+t)^(gamma - 1) * t^(-gamma)
function rho(t::Real, gamma::Real)::Real
    if t <= 0
        return Inf
    end
    return (1 + t)^(gamma - 1) * t^(-gamma)
end
# Simulate the critical value for the online change-point detection
function simulate_critical_value(; gamma::Real, p::Real, num_sims::Int=100000, T_max::Real=100.0, num_steps::Int=10000)::Real
    if !(0 <= gamma < 0.5)
        error("Parameter gamma must be in the range [0, 0.5).")
    end
    println("Parameters: gamma = $gamma, quantile = $p")
    println("Number of simulations: $num_sims, Time horizon T_max = $T_max, Time steps = $num_steps")
    # Store the supremum value from each simulation
    sup_values = Vector{Float64}(undef, num_sims)
    prog = Progress(
        num_sims;
        desc="[Simulating critical value]",
        barglyphs=BarGlyphs('=', '=', '>', '.', '.')
    )
    for i in 1:num_sims
        # 1. Define time grid and step size for this path
        dt = T_max / num_steps
        t_grid = (1:num_steps) .* dt
        # 2. Generate increments of the Brownian motion
        dW = rand(Normal(0.0, sqrt(dt)), num_steps)
        # 3. Construct the Brownian motion path W(t)
        W = cumsum(dW)
        # 4. Calculate ρ(t)|W(t)| at each time point
        process_values = rho.(t_grid, gamma) .* abs.(W)
        # 5. Find the maximum value for this path and store it
        sup_values[i] = maximum(process_values)
        # Update the progress bar
        next!(prog)
    end
    finish!(prog)
    return quantile(sup_values, p)
end
# Critical value dictionary
const critical_value_dict = Dict{Tuple{Float64,Float64},Float64}(
    # (alpha, gamma)
    # alpha = 0.01
    (0.01, 0.0) => 1.56949,
    (0.01, 0.15) => 1.81747,
    (0.01, 0.25) => 1.97581,
    (0.01, 0.35) => 2.29276,
    (0.01, 0.45) => 2.78885,
    # alpha = 0.05
    (0.05, 0.0) => 1.33027,
    (0.05, 0.15) => 1.5131,
    (0.05, 0.25) => 1.68472,
    (0.05, 0.35) => 1.93445,
    (0.05, 0.45) => 2.30402,
    # alpha = 0.10
    (0.10, 0.0) => 1.19574,
    (0.10, 0.15) => 1.35757,
    (0.10, 0.25) => 1.50264,
    (0.10, 0.35) => 1.73564,
    (0.10, 0.45) => 2.11163,
)
# get the critical value from the dictionary
function get_critical_value(alpha::Float64, gamma::Float64)
    key = (alpha, gamma)
    if haskey(critical_value_dict, key)
        return critical_value_dict[key]
    else
        error("No critical value found for alpha = $alpha, gamma = $gamma")
    end
end

# ======================================================================
# Theoretical Mean/Variance of one-step LSS difference (Proposition 2.1)
# ======================================================================
function theoretical_moments(; function_type::Symbol, p::Int, n1::Int, n2::Int, nu4::Float64)
    c1 = p / n1
    c2 = p / n2
    h = sqrt(c1 + c2 - c1 * c2)
    a = (1 - h)^2 / (1 - c1)^2
    b = (1 + h)^2 / (1 - c1)^2
    M1 = c2 / (1 - c1)
    M2 = c2 * (1 + c2 * (1 - c1)) / ((1 - c1)^3)
    M3 = (c2 * (c1^2 * c2^2 - 2 * c1 * c2^2 - 3 * c1 * c2 + c1 + c2^2 + 3 * c2 + 1)) / (1 - c1)^5
    M4 = (c2 * (-c1^3 * c2^3 + 3 * c1^2 * c2^3 + 6 * c1^2 * c2^2 - 4 * c1^2 * c2 + c1^2 - 3 * c1 * c2^3 - 12 * c1 * c2^2 - 2 * c1 * c2 + 3 * c1 + c2^3 + 6 * c2^2 + 6 * c2 + 1)) / (1 - c1)^7
    C3 = M1^4 - 3 * M1^2 * M2 + 2 * M1 * M3 + M2^2 - M4

    m_complex(z) = -(c2 * (z * (1 - c1) + 1 - c2) + 2 * z * c1 - c2 * (1 - c1) * sqrt(z - b) * sqrt(z - a)) / (2 * z * (c2 + z * c1))
    m_prime(z) = ForwardDiff.derivative(x -> m_complex(x + im * imag(z)), real(z)) 
    m = m_complex(-1.0 + 0im)
    mp = m_prime(-1.0 + 0im)
    if function_type == :linear
        mu = 0
        sigma2 = (nu4 - 3) * M1^2 / (n2 * c2) - (2 / n2) * (M1^2 - M2)
    elseif function_type == :square
        mu = -M1^2 + (nu4 - 3) * M1^2 / p + M2 / n2
        sigma2 = 4 * (nu4 - 3) * M2^2 / (n2 * c2) - 8 * C3 / n2
    elseif function_type == :log
        mu = m - 1 - log(m) - (nu4 - 3) / (2p) * (1 - m)^2 + 1 / n2 * (1 / 2 - mp * (1 / 2 - 1 / m + 1 / m^2))
        sigma2 = (nu4 - 3) * (m - 1)^2 / (n2 * c2) + 2 / n2 * (mp / m^2 - 1)
    elseif function_type == :mix
        mu = m - 1 - log(m) - (nu4 - 3) / (2p) * (1 - m)^2 + 1 / n2 * (1 / 2 - mp * (1 / 2 - 1 / m + 1 / m^2))
        sigma2 = (nu4 - 3) * (M1 + 1 - m)^2 / (n2 * c2) + 2 / n2 * (M2 - (M1 - 1)^2 + 2 - 2 / m + mp / m^2)
    end
    return mu, sigma2
end
# Precompute the theoretical mean/variance of one-step LSS difference
function precompute_moments(; p::Int, n::Int, k1::Int, k2::Int, nu4::Float64, function_type::Symbol)
    L = n - k1 - k2
    mu_all = zeros(L)
    sigma2_all = zeros(L)
    prog = Progress(L;
        desc="Precomputing all moments (p, n)=($p, $n), $function_type",
        barglyphs=BarGlyphs('=', '=', '>', '.', '.')
    )
    for s in 1:L
        mu_all[s], sigma2_all[s] = theoretical_moments(function_type=function_type, p=p, n1=k1, n2=k2 + s, nu4=nu4)
        next!(prog)
    end
    finish!(prog)
    return mu_all, sigma2_all
end

# ==========================================
# Compute the one-step LSS difference
# ==========================================
# Tr(f(F_k)) - Tr(f(F_{k-1})), (unnormalized and normalized)
function one_step_trace_diff(; X, k1, k2, mu_k_arr, sigma2_k_arr, function_type)
    p, n = size(X)
    X1 = @view X[:, 1:k1]
    X2 = @view X[:, (k1+1):(k1+k2)]
    S1_inv = inv(X1 * X1' / k1)
    S2 = X2 * X2' / k2
    Fkm1 = S1_inv * S2

    tr_diff_list_all = zeros(n - (k1 + k2))

    if function_type == :linear
        tr_f_Fkm1 = tr(Fkm1)
        for (idx, k) in enumerate(k1+k2+1:n)
            xk = @view X[:, k]
            m = k - k1
            tr_f_Fk = ((m - 1) / m) * tr_f_Fkm1 + (1 / m) * (xk' * S1_inv * xk)
            tr_diff_list_all[idx] = tr_f_Fk - tr_f_Fkm1
            tr_f_Fkm1 = tr_f_Fk
        end
    elseif function_type == :log
        epsilon = 1e-10
        tr_f_Fkm1 = logdet(Fkm1 + (1 + epsilon) * I)
        Fk_prev = Fkm1
        for (idx, k) in enumerate(k1+k2+1:n)
            m = k - k1
            xk = @view X[:, k]
            m_inv = 1.0 / m
            m1_m = (m - 1) * m_inv

            S1_inv_xk = S1_inv * xk
            Fk_prev .*= m1_m # Fk_prev = m1_m * Fk_prev
            ger!(m_inv, S1_inv_xk, xk, Fk_prev) # Fk_prev = Fk_prev + m_inv * (S1_inv_xk * xk')
            # now the Fk_prev is updated to Fk

            tr_f_Fk = logdet(Fk_prev + (1 + epsilon) * I)
            tr_diff_list_all[idx] = tr_f_Fk - tr_f_Fkm1
            tr_f_Fkm1 = tr_f_Fk
        end
    elseif function_type == :square
        tr_f_Fkm1 = tr(Fkm1 * Fkm1)
        Fk_prev = Fkm1
        for (idx, k) in enumerate(k1+k2+1:n)
            m = k - k1
            xk = @view X[:, k]
            S1_inv_xk = S1_inv * xk
            F_S1_inv_xk = Fk_prev * S1_inv_xk

            alpha_k = xk' * S1_inv_xk
            beta_k = xk' * F_S1_inv_xk

            # update tr(Fk^2)
            tr_f_Fk = ((m - 1) / m)^2 * tr_f_Fkm1 + (2 * (m - 1) / m^2) * beta_k + alpha_k^2 / m^2
            tr_diff_list_all[idx] = tr_f_Fk - tr_f_Fkm1

            # update Fk_prev = m1_m * Fk_prev + m_inv * (S1_inv_xk * xk')
            Fk_prev .*= (m - 1) / m # Fk_prev = (m-1)/m * Fk_prev
            # (use BLAS.ger! to perform rank-1 update A = A + alpha*u*v')
            # A = Fk_prev, alpha = 1/m, u = S1_inv * xk, v = xk
            ger!(1 / m, S1_inv_xk, xk, Fk_prev) # Fk = (m-1)/m * F_km1 + (S1_inv * xk * xk')/m

            # update tr_f_Fkm1
            tr_f_Fkm1 = tr_f_Fk
        end
    elseif function_type == :mix
        tr_linear_km1 = tr(Fkm1)
        tr_logdet_km1 = logdet(Fkm1 + I)
        tr_f_Fkm1 = tr_linear_km1 + tr_logdet_km1
        S2_prev = X2 * X2'
        for (idx, k) in enumerate(k1+k2+1:n)
            m = k - k1
            xk = @view X[:, k]

            # update tr(Fk) 
            S1_inv_xk = S1_inv * xk
            alpha_k = dot(xk, S1_inv_xk)
            tr_linear_k = ((m - 1) / m) * tr_linear_km1 + alpha_k / m

            # update logdet(Fk+I) 
            S2_k = S2_prev + xk * xk'
            Fk = (S1_inv * S2_k) / m
            tr_logdet_k = logdet(Fk + I)

            tr_f_Fk = tr_linear_k + tr_logdet_k
            tr_diff_list_all[idx] = tr_f_Fk - tr_f_Fkm1

            # update next iteration
            tr_f_Fkm1 = tr_f_Fk
            tr_linear_km1 = tr_linear_k # update tr() part
            S2_prev = S2_k              # update S2 matrix
        end
    else
        error("unsupported function_type")
    end

    tr_diff_normalized = (tr_diff_list_all .- mu_k_arr) ./ sqrt.(sigma2_k_arr)
    return (
        all=tr_diff_list_all,
        normalized=tr_diff_normalized
    )
end

# ==========================================
# Generate covariance matrices
# ==========================================
function generate_ar1_covariance(p::Int, rho::Float64)
    if abs(rho) >= 1
        @warn "rho is generally recommended to be in the interval (-1, 1)"
    end
    Sigma = [rho^abs(i - j) for i in 1:p, j in 1:p]
    for i in 1:p
        Sigma[i, i] = 2.0
    end
    return Sigma
end

function generate_spike_covariance(p::Int, delta::Float64)
    if delta <= 0
        @warn "delta should typically be positive to represent variance inflation"
    end
    Sigma = 1.5 * Matrix{Float64}(I, p, p)
    for i in 1:min(5, p)
        Sigma[i, i] = delta
    end
    return Sigma
end

# =====================================
# Estimate the 4th moment of the data
# =====================================
### Reference: Section 3.3 in "Bootstrapping spectral statistics in high dimensions" by Lopes, Blandino, and Aue, 2019, Biometrika.
function calculate_kappa(Y)
    # Y: p x n data matrix
    p, n = size(Y)
    S = Y * Y' / n

    tau = tr(S^2) - tr(S)^2 / n
    Y_norm = (norm.(eachcol(Y))) .^ 2
    nu = var(Y_norm)
    omega = sum((sum(abs2, Y; dims=2) ./ n) .^ 2)

    kappa = max(1.0, 3.0 + (nu - 2 * tau) / omega)
    return kappa
end

# ==========================================
# Online (Streaming) utilities
# ==========================================
struct WeightFunctionConfig
    name::String
    params::Dict
end

# scalar weight values (online)
# (1+t)^(gamma-1) * t^(-gamma),  t = i/m
@inline function weight_value(wf::WeightFunctionConfig, i::Int, n_start::Int)::Float64
    t = i / n_start
    if wf.name == "log"
        α = wf.params[:alpha]
        return (1 + t)^(-0.5) * (-2 * log(α) + log(1 + t))^(-0.5)
    else
        γ = wf.params[:gamma]
        return (1 + t)^(γ - 1) * t^(-γ)
    end
end

# === Online (streaming) LSS increment state ===
mutable struct TraceDiffOnlineState
    S1_inv::Matrix{Float64}
    m::Int                        # current denominator m = k - k1
    function_type::Symbol

    # generic caches
    tr_f_prev::Float64            # tr f(F_{k-1})
    F_prev::Matrix{Float64}       # current F_{k-1} (used by :log and :square)

    # for :mix
    tr_linear_prev::Float64
    S2_prev::Matrix{Float64}

    # constants / options
    eps_logdet::Float64
end

"""
    init_trace_state(Y, k1, k2, function_type) -> TraceDiffOnlineState

Initialize the online state at time k = k1+k2 (i.e., after the initial reference + warmup blocks).
The next update corresponds to k = k1+k2+1.
"""
function init_trace_state(Y::AbstractMatrix{<:Real}, k1::Int, k2::Int, function_type::Symbol)
    p, n = size(Y)
    @assert k1 + k2 <= n "Need at least k1+k2 columns to initialize."

    X1 = @view Y[:, 1:k1]
    X2 = @view Y[:, (k1+1):(k1+k2)]

    S1_inv = inv(Matrix(X1 * X1' / k1))
    S2 = Matrix(X2 * X2' / k2)
    F0 = S1_inv * S2
    m0 = k2 # because k = k1+k2 => m = k-k1 = k2

    eps = 1e-10

    if function_type == :linear
        tr_f0 = tr(F0)
        return TraceDiffOnlineState(S1_inv, m0, function_type, tr_f0, Matrix{Float64}(undef, 0, 0),
            0.0, Matrix{Float64}(undef, 0, 0), eps)
    elseif function_type == :log
        tr_f0 = logdet(F0 + (1 + eps) * I)
        return TraceDiffOnlineState(S1_inv, m0, function_type, tr_f0, F0,
            0.0, Matrix{Float64}(undef, 0, 0), eps)
    elseif function_type == :square
        tr_f0 = tr(F0 * F0)
        return TraceDiffOnlineState(S1_inv, m0, function_type, tr_f0, F0,
            0.0, Matrix{Float64}(undef, 0, 0), eps)
    elseif function_type == :mix
        tr_linear0 = tr(F0)
        tr_logdet0 = logdet(F0 + I)
        tr_f0 = tr_linear0 + tr_logdet0
        S2_prev = Matrix(X2 * X2')
        return TraceDiffOnlineState(S1_inv, m0, function_type, tr_f0, Matrix{Float64}(undef, 0, 0),
            tr_linear0, S2_prev, eps)
    else
        error("unsupported function_type=$function_type")
    end
end

"""
    update_trace_diff!(st, xk) -> (diff_raw, tr_f_new)

Given a new observation column xk at time k (k >= k1+k2+1),
update the state from F_{k-1} to F_k and return the raw increment:
    diff_raw = tr f(F_k) - tr f(F_{k-1}).
"""
function update_trace_diff!(st::TraceDiffOnlineState, xk::AbstractVector{<:Real})
    S1_inv = st.S1_inv
    m_prev = st.m
    m = m_prev + 1
    st.m = m

    if st.function_type == :linear
        alpha_k = dot(xk, S1_inv * xk)
        tr_f_new = ((m - 1) / m) * st.tr_f_prev + alpha_k / m
        diff = tr_f_new - st.tr_f_prev
        st.tr_f_prev = tr_f_new
        return diff, tr_f_new

    elseif st.function_type == :log
        eps = st.eps_logdet
        m_inv = 1.0 / m
        m1_m = (m - 1) * m_inv

        S1_inv_xk = S1_inv * xk
        st.F_prev .*= m1_m
        ger!(m_inv, S1_inv_xk, xk, st.F_prev)   # F <- (m-1)/m F + (S1_inv xk xk')/m

        tr_f_new = logdet(st.F_prev + (1 + eps) * I)
        diff = tr_f_new - st.tr_f_prev
        st.tr_f_prev = tr_f_new
        return diff, tr_f_new

    elseif st.function_type == :square
        F_prev = st.F_prev

        S1_inv_xk = S1_inv * xk
        F_S1_inv_xk = F_prev * S1_inv_xk

        alpha_k = dot(xk, S1_inv_xk)
        beta_k = dot(xk, F_S1_inv_xk)

        tr_f_new = ((m - 1) / m)^2 * st.tr_f_prev + (2 * (m - 1) / m^2) * beta_k + (alpha_k^2) / (m^2)
        diff = tr_f_new - st.tr_f_prev

        # update F itself (rank-1 update)
        F_prev .*= (m - 1) / m
        ger!(1 / m, S1_inv_xk, xk, F_prev)

        st.tr_f_prev = tr_f_new
        return diff, tr_f_new

    elseif st.function_type == :mix
        # update tr(Fk)
        S1_inv_xk = S1_inv * xk
        alpha_k = dot(xk, S1_inv_xk)
        tr_linear_new = ((m - 1) / m) * st.tr_linear_prev + alpha_k / m

        # update S2 and logdet(Fk+I)
        st.S2_prev .+= xk * xk'
        Fk = (S1_inv * st.S2_prev) / m
        tr_logdet_new = logdet(Fk + I)

        tr_f_new = tr_linear_new + tr_logdet_new
        diff = tr_f_new - st.tr_f_prev

        st.tr_f_prev = tr_f_new
        st.tr_linear_prev = tr_linear_new
        return diff, tr_f_new

    else
        error("unsupported function_type")
    end
end

"""
    online_detect_all_weights(Y; k1,k2, mu_k_arr, sigma2_k_arr, weight_functions, critical_values)

Process data sequentially (one new column at a time) and return detection times for each weight.
"""
function online_detect_all_weights(
    Y::AbstractMatrix{<:Real};
    k1::Int,
    k2::Int,
    mu_k_arr::AbstractVector{<:Real},
    sigma2_k_arr::AbstractVector{<:Real},
    function_type::Symbol,
    weight_functions::Vector{WeightFunctionConfig},
    critical_values::Dict{String,Float64}
)
    p, n_total = size(Y)
    n_start = k1 + k2
    st = init_trace_state(Y, k1, k2, function_type)

    # CUSUM statistics
    psi = 0.0
    detected_time = Dict(wf.name => -1 for wf in weight_functions)
    remaining = Set(wf.name for wf in weight_functions)

    min_delay = round(Int, log(n_start))

    for k in (n_start+1):n_total
        i = k - n_start  # monitoring index starting from 1
        xk = @view Y[:, k]

        diff_raw, _ = update_trace_diff!(st, xk)
        diff_normalized = (diff_raw - mu_k_arr[i]) / sqrt(sigma2_k_arr[i])

        psi += diff_normalized
        psi_abs = abs(psi)

        # update each weight function online
        for wf in weight_functions
            wf_name = wf.name
            if !in(wf_name, remaining) || i <= min_delay
                continue
            end

            w = weight_value(wf, i, n_start)
            stat = w * psi_abs / sqrt(n_start)

            if stat > critical_values[wf_name]
                detected_time[wf_name] = k
                delete!(remaining, wf_name)
            end
        end

        isempty(remaining) && break
    end

    return detected_time
end
