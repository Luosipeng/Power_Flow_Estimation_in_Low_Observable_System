using LinearAlgebra
using SparseArrays
using Statistics
using Random  # 新增

include("../src/build_observed_matrix_z.jl")
include("../src/build_noise_precision_beta.jl")
include("../src/implement_data.jl")
include("../src/matrix_completion.jl")
include("../ios/read_mat.jl")
include("../src/lindistflow.jl")
include("../src/power_flow_optimal.jl")
include("../src/likelihood_gaussian.jl")

# 简易连通性诊断：统计观测二部图的连通分量
function count_components(m::Int, n::Int, pairs::Vector{Tuple{Int,Int}})
    row_adj = [Int[] for _ in 1:m]
    col_adj = [Int[] for _ in 1:n]
    for (i,j) in pairs
        push!(row_adj[i], j)
        push!(col_adj[j], i)
    end
    visited_r = falses(m); visited_c = falses(n)
    comps = 0
    for seed_i in 1:m
        if !visited_r[seed_i] && !isempty(row_adj[seed_i])
            comps += 1
            # BFS
            rq = [seed_i]; visited_r[seed_i] = true
            while !isempty(rq)
                i = pop!(rq)
                for j in row_adj[i]
                    if !visited_c[j]
                        visited_c[j] = true
                        for ii in col_adj[j]
                            if !visited_r[ii]
                                visited_r[ii] = true
                                push!(rq, ii)
                            end
                        end
                    end
                end
            end
        end
    end
    return comps
end

function run_stage2_test()
    branch = read_topology_mat("D:/luosipeng/matpower8.1/pf_parallel_out/topology.mat")
    daily_predictions = generate_daily_predictions(result, 1, 1)
    observed_matrix_Z, observed_pairs, monitored_obs = build_observed_matrix_Z(daily_predictions; monitor_buses=Set([8, 12]))
    noise_precision_β = build_noise_precision_beta(daily_predictions)

    Random.seed!(1234)
    tolerance = 1e-6
    c = 1e-7
    d = 1e-7
    max_iter = 400

    observed_matrix_Z = Array{Float64}(observed_matrix_Z)
    noise_precision_β = Array{Float64}(noise_precision_β)

    println("observed fraction (nonzeros): ", fraction_available_data(observed_matrix_Z))
    posβ = count(!iszero, noise_precision_β)
    println("β>0 ratio: ", posβ / length(noise_precision_β))

    # 掩码与邻接
    m, n = size(observed_matrix_Z)
    O = falses(m, n)
    row_obs = [Int[] for _ in 1:m]
    col_obs = [Int[] for _ in 1:n]
    for (i, j) in observed_pairs
        O[i, j] = true
        push!(row_obs[i], j)
        push!(col_obs[j], i)
    end
    println("Observed graph components: ", count_components(m, n, observed_pairs))

    # SVD 初始化
    svd_res = svd(observed_matrix_Z)
    r = min(5, minimum(size(observed_matrix_Z)))
    U_r = svd_res.U[:, 1:r]
    Σ_r = svd_res.S[1:r]
    Vt_r = svd_res.Vt[1:r, :]

    sqrtD = Diagonal(sqrt.(Σ_r))
    A_mean = Array{Float64}(U_r * sqrtD)
    B_mean = Array{Float64}(Vt_r' * sqrtD)

    α = 1e-3
    Σa0 = α .* Matrix{Float64}(I, r, r)
    Σb0 = α .* Matrix{Float64}(I, r, r)
    Σa_list = [copy(Σa0) for _ in 1:size(A_mean, 1)]
    Σb_list = [copy(Σb0) for _ in 1:size(B_mean, 1)]
    γ = fill(1.0, r)

    X_old = Array{Float64}(A_mean * B_mean')
    latent_dim = size(A_mean, 2)

    history = Dict{Symbol, Vector{Float64}}(
        :rel_change => Float64[]
    )

    # 观测行/列集合
    obs_rows, obs_cols = Set{Int}(), Set{Int}()
    for (i,j) in observed_pairs
        push!(obs_rows, i); push!(obs_cols, j)
    end
    miss_rows = setdiff(1:size(A_mean,1), collect(obs_rows))
    miss_cols = setdiff(1:size(B_mean,1), collect(obs_cols))

    # 仅初始化：无观测行/列 = 观测均值 + 微扰
    jitter = 1e-3
    if !isempty(obs_rows)
        A_prior = vec(mean(A_mean[collect(obs_rows), :], dims=1))
        sA = max(norm(A_prior), 1.0)
        for i in miss_rows
            A_mean[i, :] .= A_prior .+ jitter*sA*randn(length(A_prior))
            Σa_list[i] = Σa0
        end
    end
    if !isempty(obs_cols)
        B_prior = vec(mean(B_mean[collect(obs_cols), :], dims=1))
        sB = max(norm(B_prior), 1.0)
        for j in miss_cols
            B_mean[j, :] .= B_prior .+ jitter*sB*randn(length(B_prior))
            Σb_list[j] = Σb0
        end
    end
    X_old = Array{Float64}(A_mean * B_mean')

    λ = 0.5
    warmup_gamma_iters = 10

    for it in 1:max_iter
        # 更新 A（仅观测行）
        for i in 1:size(A_mean, 1)
            if i in obs_rows
                βBtB = cal_beta_BTB_i(i, B_mean, Σb_list, observed_pairs, noise_precision_β, latent_dim)
                Σa_new = cal_sigma_a_i(βBtB, γ)
                a_new  = cal_a_mean_i(i, B_mean, Σa_new, observed_pairs, noise_precision_β, observed_matrix_Z)
                Σa_list[i] = Σa_new
                A_mean[i, :] = (1-λ).*A_mean[i, :] .+ λ.*a_new
            end
        end

        # 更新 B（仅观测列）
        for j in 1:size(B_mean, 1)
            if j in obs_cols
                βAtA = cal_beta_ATA_j(j, A_mean, Σa_list, observed_pairs, noise_precision_β, latent_dim)
                Σb_new = cal_sigma_b_j(βAtA, γ)
                b_new  = cal_b_mean_j(j, A_mean, Σb_new, observed_pairs, noise_precision_β, observed_matrix_Z)
                Σb_list[j] = Σb_new
                B_mean[j, :] = (1-λ).*B_mean[j, :] .+ λ.*b_new
            end
        end

        if it > warmup_gamma_iters
            for k in 1:length(γ)
                aTa = cal_aTa_i(k, A_mean, Σa_list)
                bTb = cal_bTb_j(k, B_mean, Σb_list)
                γ[k] = clamp((2c + size(A_mean,1) + size(B_mean,1)) / (aTa + bTb + 2d), 1e-6, 1e6)
            end
        end

        X_new = Array{Float64}(A_mean * B_mean')
        rel = norm(X_new - X_old) / max(norm(X_old), 1e-12)

        println("Iter $it: rel_change = $rel")
        push!(history[:rel_change], rel)
        X_old = X_new
        if rel < tolerance
            println("Converged at iter=$it, rel=$(rel)")
            break
        end
    end

    if isempty(history[:rel_change]) || history[:rel_change][end] ≥ tolerance
        @warn "Not below tolerance yet. tail(rel)=$(history[:rel_change][max(end-4,1):end])"
    end

    # 行/列偏置 + 全局均值，构造更稳的预测（避免跨分量=0）
    obs_vals = [observed_matrix_Z[i,j] for (i,j) in observed_pairs]
    μ = isempty(obs_vals) ? 0.0 : mean(obs_vals)
    r_bias = zeros(Float64, m)
    c_bias = zeros(Float64, n)
    for i in obs_rows
        js = row_obs[i]
        r_bias[i] = mean(observed_matrix_Z[i, js]) - μ
    end
    for j in obs_cols
        is = col_obs[j]
        c_bias[j] = mean(observed_matrix_Z[is, j]) - μ
    end
    # 预测矩阵：μ + r + c + AB
    X_pred = X_old .+ r_bias .* ones(1, n) .+ ones(m, 1) .* c_bias' .+ μ

    # 用预测填充未观测项：观测处保留原值
    X_completed = copy(observed_matrix_Z)
    @inbounds for i in 1:m, j in 1:n
        if !O[i, j]
            X_completed[i, j] = X_pred[i, j]
        end
    end

    # 诊断：观测行的未观测列（应非0）
    if !isempty(obs_rows)
        i0 = first(obs_rows)
        unobs_js = [j for j in 1:n if !O[i0, j]]
        println("Row $i0 unobserved cols: ", unobs_js)
        println("Predictions X_pred[$i0, unobs]: ", X_pred[i0, unobs_js])
    end
    println("Filled entries: ", count(!, O), " / ", m*n)

    elbo_result = compute_elbo_with_physics(
        X_old, A_mean, B_mean, Σa_list, Σb_list, γ,
        observed_matrix_Z, noise_precision_β, observed_pairs
    )

    return (X = X_completed, history = history,
            flows = (P = Float64[], Q = Float64[], V = Float64[]),
            injections = (P = Float64[], Q = Float64[]), elbo = elbo_result)
end

X, history, flows, injections, elbo = run_stage2_test()