using LinearAlgebra
using SparseArrays
using Statistics
using MAT

include("../src/build_observed_matrix_z.jl")
include("../src/build_noise_precision_beta.jl")
include("../src/implement_data.jl")
include("../src/matrix_completion.jl")
include("../ios/read_mat.jl")
include("../src/lindistflow.jl")
include("../src/power_flow.jl")
include("../src/get_topology.jl")

# 扩展返回内容：A_mean、Σa_list、B_mean、Σb_list、X
function run_stage2(branch, daily_predictions)
    
    observed_matrix_Z, observed_pairs = build_observed_matrix_Z(daily_predictions)
    noise_precision_β = build_noise_precision_beta(daily_predictions)

    tolerance = 1e-6
    c = 1e-4
    d = 1e-4
    max_iter = 400

    root_bus = 1
    Vref = 1.0

    observed_matrix_Z = Array{Float64}(observed_matrix_Z)
    noise_precision_β = Array{Float64}(noise_precision_β)

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

    Pij_sol = Float64[]
    Qij_sol = Float64[]
    V_sol   = Float64[]
    Pinj_sol = Float64[]
    Qinj_sol = Float64[]

    for it in 1:max_iter
        # 更新 A
        for i in 1:size(A_mean, 1)
            βBtB = cal_beta_BTB_i(i, B_mean, Σb_list, observed_pairs, noise_precision_β, latent_dim)
            Σa_list[i] = cal_sigma_a_i(βBtB, γ)
            A_mean[i, :] = cal_a_mean_i(i, B_mean, Σa_list[i], observed_pairs, noise_precision_β, observed_matrix_Z)
        end
        # 更新 B
        for j in 1:size(B_mean, 1)
            βAtA = cal_beta_ATA_j(j, A_mean, Σa_list, observed_pairs, noise_precision_β, latent_dim)
            Σb_list[j] = cal_sigma_b_j(βAtA, γ)
            B_mean[j, :] = cal_b_mean_j(j, A_mean, Σb_list[j], observed_pairs, noise_precision_β, observed_matrix_Z)
        end
        # 更新 γ
        for k in 1:length(γ)
            aTa = cal_aTa_i(k, A_mean, Σa_list)
            bTb = cal_bTb_j(k, B_mean, Σb_list)
            γ[k] = clamp((2c + size(A_mean,1) + size(B_mean,1)) / (aTa + bTb + 2d), 1e-6, 1e6)
        end

        # 构造并调用潮流
        X_new = Array{Float64}(A_mean * B_mean')
        P_inj = X_new[:, 1]./10
        Q_inj = X_new[:, 2]./10
        Vb    = X_new[:, 5]

        # AC 潮流
        V_sol, θ_sol, Pinj_sol, Qinj_sol, Vr_sol, Vi_sol =
            ac_nodal_injection(P_inj, Q_inj, Vb, branch, root_bus, Vref, 0.0, observed_pairs, verbose=false)

        # 回填
        X_new[:, 5] .= V_sol
        X_new[:, 1] .= Pinj_sol .* 10
        X_new[:, 2] .= Qinj_sol .* 10
        X_new[:, 3] .= Vr_sol
        X_new[:, 4] .= Vi_sol

        # 收敛监控
        numerator = norm(X_new - X_old)
        denominator = max(norm(X_old), 1e-12)
        rel = numerator / denominator
        push!(history[:rel_change], rel)
        X_old = X_new

        if rel < tolerance
            # println("Converged at iter=$it, rel=$(rel)")
            break
        end
    end

    if isempty(history[:rel_change]) || history[:rel_change][end] ≥ tolerance
        @warn "Not below tolerance yet. tail(rel)=$(history[:rel_change][max(end-4,1):end])"
    end

    return (
        X = X_old,
        history = history,
        flows = (P = Pij_sol, Q = Qij_sol, V = V_sol),
        injections = (P = Pinj_sol, Q = Qinj_sol),
        A_mean = A_mean,
        Σa_list = Σa_list,
        B_mean = B_mean,
        Σb_list = Σb_list
    )
end

mutable struct RunSuccessRecord
    idx::Int
    prob::Float64
    A_mean::Matrix{Float64}
    Σa_list::Vector{Matrix{Float64}}
    B_mean::Matrix{Float64}
    Σb_list::Vector{Matrix{Float64}}
    X_old::Matrix{Float64}
end

function run_all_branches(result, branch_list, prob_list; tol=1e-6)
    success_records = RunSuccessRecord[]
    failed_indices = Int[]
    success_count = 0
    daily_predictions = generate_daily_predictions(result, 1, 1)

    for (idx, brch) in enumerate(branch_list)
        try
            res = run_stage2(brch, daily_predictions)
            hist = res.history
            ok = !isempty(hist[:rel_change]) && last(hist[:rel_change]) ≤ tol
            if ok
                success_count += 1
                push!(success_records, RunSuccessRecord(
                    idx,
                    prob_list[idx],
                    res.A_mean,
                    res.Σa_list,
                    res.B_mean,
                    res.Σb_list,
                    res.X
                ))
                @info "Branch idx=$idx converged successfully."
            else
                push!(failed_indices, idx)
                @warn "Branch idx=$idx did not converge to tolerance; last rel=$(isempty(hist[:rel_change]) ? NaN : last(hist[:rel_change]))"
            end
        catch e
            push!(failed_indices, idx)
            @warn "Branch idx=$idx failed with error: $e"
        end
    end

    return (success_count=success_count,
            success_records=success_records,
            failed_indices=failed_indices)
end

# 顶层只做数据准备和一次调用，避免全局可变状态
branch_based = read_topology_mat("D:/luosipeng/matpower8.1/pf_parallel_out/topology.mat")
branch_list, prob_list = generate_branch_list_with_prior(
    branch_based;
    param_sets = nothing,
    param_source_rows = (35,5,1),
    per_line_cartesian = true
)

res_all = run_all_branches(result, branch_list, prob_list; tol=1e-6)
println("Run finished. success=$(res_all.success_count), failed=$(length(res_all.failed_indices))")
if !isempty(res_all.success_records)
    r1 = res_all.success_records[1]
    println("First success: idx=$(r1.idx), prior_prob=$(r1.prob), size(A_mean)=$(size(r1.A_mean)), size(B_mean)=$(size(r1.B_mean))")
end
