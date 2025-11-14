using LinearAlgebra
using SparseArrays
using Statistics
using MAT
using ProgressMeter

include("../src/build_observed_matrix_z.jl")
include("../src/build_noise_precision_beta.jl")
include("../src/implement_data.jl")
include("../src/matrix_completion.jl")
include("../ios/read_mat.jl")
include("../src/ac_dc_power_flow.jl")
include("../src/power_flow.jl")
include("../src/get_topology.jl")

# 从 result 获取总时间长度的辅助函数（请按你的 result 结构实现）
function get_total_len(result)::Int
    # 下面是示意，你需要用自己数据结构返回总时长（样本个数）
    # 例如：return length(result["time"])
    return result["total_len"]
end

# 扩展返回内容：返回该次采样的监测损失 loss_sample，同时允许传入 monitor_buses
function run_stage2(branch, branchDC, daily_predictions; monitor_buses=Set([8, 12]))
    observed_matrix_Z, observed_pairs, monitored_obs =
        build_observed_matrix_Z(daily_predictions; monitor_buses=monitor_buses)
    noise_precision_β = build_noise_precision_beta(daily_predictions)

    tolerance = 1e-6
    c = 1e-7
    d = 1e-7
    max_iter = 400

    root_bus = 1
    Vref = 1.0
    inv_bus = 18
    rec_bus = 1
    eta = 0.9
    nac = 33
    ndc = 4

    observed_matrix_Z = Array{Float64}(observed_matrix_Z)
    noise_precision_β = Array{Float64}(noise_precision_β)

    svd_res = svd(observed_matrix_Z)
    r = 5
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
        P_inj = X_new[:, 1] ./ 100
        Q_inj = X_new[:, 2] ./ 100
        Vb    = X_new[:, 5]
        Vbr   = X_new[:, 3]
        Vbi   = X_new[:, 4]

        # 注意：这里的 ac_dc_power_flow 参数顺序遵循你的新版调用
        Vr_ac_sol, Vi_ac_sol, V_ac_sol, Pinj_ac_sol, Qinj_ac_sol, V_dc_sol, Pinj_dc_sol =
            ac_dc_power_flow(
                branch, branchDC, nac, ndc,
                P_inj, Q_inj, Vb, Vbr, Vbi,
                root_bus, 1, 1.00833,  # 你传入的中间三个参数
                inv_bus, rec_bus, eta, Vref,
                observed_pairs, true
            )

        # 回填
        X_new[:, 5] .= vcat(V_ac_sol, V_dc_sol)
        X_new[:, 1] .= vcat(Pinj_ac_sol .* 100, Pinj_dc_sol .* 100)
        X_new[:, 2] .= vcat(Qinj_ac_sol .* 100, zeros(length(Pinj_dc_sol)) .* 100)
        X_new[:, 3] .= vcat(Vr_ac_sol, V_dc_sol)
        X_new[:, 4] .= vcat(Vi_ac_sol, zeros(length(V_dc_sol)))

        # 收敛监控
        numerator = norm(X_new - X_old)
        denominator = max(norm(X_old), 1e-12)
        rel = numerator / denominator
        push!(history[:rel_change], rel)
        X_old = X_new

        if rel < tolerance
            break
        end
    end

    if isempty(history[:rel_change]) || history[:rel_change][end] ≥ tolerance
        @warn "Not below tolerance yet. tail(rel)=$(history[:rel_change][max(end-4,1):end])"
    end

    # Loss：bus 12 与 bus 8 的 vmag（列5）百分比误差
    function loss_for_bus(monitored_obs, X, bus::Int, col::Int)
        if haskey(monitored_obs, bus) && haskey(monitored_obs[bus], col)
            z = monitored_obs[bus][col]
            x = X[bus, col]
            if abs(z) > 0
                return abs(x - z) / abs(z) * 100.0
            else
                return abs(x - z) * 100.0
            end
        else
            return missing
        end
    end

    l12 = loss_for_bus(monitored_obs, X_old, 12, 5)
    l08 = loss_for_bus(monitored_obs, X_old, 8, 5)

    loss_sample = begin
        if l12 !== missing && l08 !== missing
            l12 + l08
        elseif l12 !== missing
            l12
        elseif l08 !== missing
            l08
        else
            missing
        end
    end

    return (
        X = X_old,
        history = history,
        flows = (P = Pij_sol, Q = Qij_sol, V = V_sol),
        injections = (P = Pinj_sol, Q = Qinj_sol),
        A_mean = A_mean,
        Σa_list = Σa_list,
        B_mean = B_mean,
        Σb_list = Σb_list,
        loss = loss_sample,
        loss_bus8 = l08,
        loss_bus12 = l12
    )
end


mutable struct RunSuccessRecord
    idx::Int
    prob::Float64
    total_loss::Float64
    count_used::Int
end

# 主流程：按步长 step 采样，遍历所有 branch，对每个 branch 累计所有采样的 Loss
function evaluate_branches_over_samples(result, branch_list, branchDC, prob_list; step=1, tol=1e-8, monitor_buses=Set([8,12]))
    # total_len = get_total_len(result)  # 如需按真实长度采样可启用
    total_len = 1
    start_counts = collect(1:step:total_len)

    nB = length(branch_list)
    total_loss = fill(0.0, nB)
    total_loss_bus8 = fill(0.0, nB)
    total_loss_bus12 = fill(0.0, nB)
    used_count = fill(0, nB)  # 记录有效样本数（有loss）

    # 外层遍历采样时间
    for (si, start_count) in enumerate(start_counts)
        # 生成该采样时间的日预测
        daily_predictions = generate_daily_predictions(result, start_count, 1)

        # 遍历所有备选拓扑（branch）
        @showprogress for (idx, brch) in enumerate(branch_list)
            try
                res = run_stage2(brch, branchDC, daily_predictions; monitor_buses=monitor_buses)
                hist = res.history
                ok = !isempty(hist[:rel_change]) && last(hist[:rel_change]) ≤ tol
                if !ok
                    @debug "Branch idx=$idx sample si=$si not under tol, last_rel=$(isempty(hist[:rel_change]) ? NaN : last(hist[:rel_change]))"
                end
                if res.loss !== missing
                    total_loss[idx] += Float64(res.loss)
                    used_count[idx] += 1
                else
                    @debug "Branch idx=$idx sample si=$si has no monitored loss (missing)."
                end
                if res.loss_bus8 !== missing
                    total_loss_bus8[idx] += Float64(res.loss_bus8)
                end
                if res.loss_bus12 !== missing
                    total_loss_bus12[idx] += Float64(res.loss_bus12)
                end
            catch e
                @warn "Branch idx=$idx sample si=$si failed with error: $e"
                # 失败则本样本对该 branch 不计分
            end
        end
        @info "Finished sample si=$si (start=$start_count)"
    end

    # 归并结果
    records = RunSuccessRecord[]
    for i in 1:nB
        push!(records, RunSuccessRecord(i, prob_list[i], total_loss[i], used_count[i]))
    end

    # 仅对 used_count>0 的进行排序；若需要对0样本的给高惩罚，可自定义
    valid = filter(r -> r.count_used > 0, records)
    sort!(valid, by = r -> r.total_loss, rev = false)

    # 取前五
    topk = first(valid, min(5, length(valid)))

    return (
        records = records,
        top5 = topk,
        start_counts = start_counts,
        step = step,
        total_loss = total_loss,
        total_loss_bus8 = total_loss_bus8,
        total_loss_bus12 = total_loss_bus12,
        used_count = used_count
    )
end

# 顶层：准备 branch_list，并调用评估
branchAC, branchDC = read_topology_mat("C:/Users/PC/Desktop/paper_case/topology_results.mat")
branch_list, prob_list = generate_branch_list_with_prior(
    branchAC;
    param_sets = nothing,
    param_source_rows = (35,5,1),
    per_line_cartesian = true
)

res_eval = evaluate_branches_over_samples(result, branch_list, branchDC, prob_list; step=1, tol=1e-8, monitor_buses=Set([8, 12]))

println("Evaluation finished. total branches=$(length(branch_list))")
println("Valid branches with at least 1 sample used = $(length(filter(r -> r.count_used>0, res_eval.records)))")

println("Top-5 branches (by cumulative loss):")
for (rank, r) in enumerate(res_eval.top5)
    idx = r.idx
    lb8 = res_eval.total_loss_bus8[idx]
    lb12 = res_eval.total_loss_bus12[idx]
    ltot = res_eval.total_loss[idx]
    used = res_eval.used_count[idx]
    println("[$rank] idx=$(idx), prior_prob=$(r.prob), loss_bus8=$(lb8), loss_bus12=$(lb12), total_loss=$(ltot), used_samples=$(used)")
end

all_total_loss = res_eval.total_loss
all_used = res_eval.used_count
nB = length(all_total_loss)
idxs = collect(1:nB)

# 可选：若你希望忽略 used_count=0 的拓扑，可屏蔽它们（设为 NaN，绘图会断开）
plot_loss = copy(all_total_loss)
for i in 1:nB
    if all_used[i] == 0
        plot_loss[i] = NaN
    end
end

# 2) 找到最小损失的 idx（仅在 used_count>0 的集合中）
valid_idx = [i for i in 1:nB if all_used[i] > 0 && isfinite(plot_loss[i])]
if isempty(valid_idx)
    @warn "没有有效的样本用于评估，无法确定最小损失的 idx。"
else
    min_idx = argmin(plot_loss[valid_idx])
    best_idx = valid_idx[min_idx]
    best_loss = plot_loss[best_idx]
    best_prob = res_eval.records[best_idx].prob
    println("Best branch by cumulative loss: idx=$(best_idx), prior_prob=$(best_prob), total_loss=$(best_loss), used_samples=$(all_used[best_idx])")
end

# 3) 绘制曲线并保存
plt = plot(
    idxs, plot_loss;
    seriestype = :line,
    linewidth = 2,
    color = :steelblue,
    legend = :topright,
    label = "total_loss (used_count>0)",
    xlabel = "idx",
    ylabel = "cumulative loss",
    title = "Cumulative Loss over Branch idx",
    grid = true,
)

# 标注最优点（如果存在）
if !isempty(valid_idx)
    scatter!(plt, [best_idx], [best_loss], markersize=6, color=:red, label="best idx=$(best_idx)")
    annotate!(plt, best_idx, best_loss, text("best idx=$(best_idx)", :red, 8, :bold))
end

savefig(plt, "loss_curve.png")
println("Loss curve saved to loss_curve.png")