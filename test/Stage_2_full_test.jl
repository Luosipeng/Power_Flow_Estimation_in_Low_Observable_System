using LinearAlgebra
using SparseArrays
using Statistics
using ProgressMeter
using Printf

# 你的项目已有
include("../src/build_observed_matrix_z.jl")
include("../src/build_noise_precision_beta.jl")
include("../src/implement_data.jl")
include("../src/matrix_completion.jl")
include("../ios/read_mat.jl")
include("../src/lindistflow.jl")
include("../src/power_flow_optimal.jl")
include("../src/get_topology.jl")

# ========== 实用函数 ==========

# 权重化观测位置的 β：beta_weighted[i,j] = KAPPA * beta[i,j]
function weight_observed_beta!(beta::AbstractMatrix{<:Real},
                               observed_pairs::AbstractVector{<:Tuple{Int,Int}},
                               KAPPA::Real)
  @assert KAPPA >= 1.0 "KAPPA 应≥1；若<1则降低观测权重"
  @inbounds for (i,j) in observed_pairs
    beta[i,j] = float(beta[i,j]) * KAPPA
  end
  return beta
end

# 观测位置 MAE/RMSE
function obs_metrics(Z::AbstractMatrix{<:Real},
                     X::AbstractMatrix{<:Real},
                     observed_pairs::AbstractVector{<:Tuple{Int,Int}})
  res = Vector{Float64}(undef, length(observed_pairs))
  @inbounds for (k, (i,j)) in enumerate(observed_pairs)
    res[k] = abs(float(Z[i,j]) - float(X[i,j]))
  end
  mae = mean(res)
  rmse = sqrt(mean(x->x^2, res))
  return mae, rmse
end

# 简化 ELBO 计算（γ 点估计；不使用 module）
const LOG2PI = log(2π)
loggamma_fallback(x::Real) = log(gamma(float(x)))

function elbo_point_gamma(
  Z::AbstractMatrix{<:Real},
  beta::AbstractMatrix{<:Real},
  observed_pairs::AbstractVector{<:Tuple{Int,Int}},
  A_mean::AbstractMatrix{<:Real},
  B_mean::AbstractMatrix{<:Real},
  SigmaA::AbstractVector{<:AbstractMatrix{<:Real}},
  SigmaB::AbstractVector{<:AbstractMatrix{<:Real}},
  gamma::AbstractVector{<:Real},
  c::Real, d::Real
)
  m, r = size(A_mean)
  n, rB = size(B_mean)
  @assert r == rB "latent dimension mismatch"
  @assert length(SigmaA) == m
  @assert length(SigmaB) == n
  @assert length(gamma) == r

  # 1) Likelihood
  like = 0.0
  @inbounds for (i,j) in observed_pairs
    β = max(float(beta[i,j]), 1e-12)
    ai = @view A_mean[i, :]
    bj = @view B_mean[j, :]
    Σai = SigmaA[i]
    Σbj = SigmaB[j]
    μ = dot(ai, bj)
    resid2_mean = (float(Z[i,j]) - μ)^2 + tr(Σai*Σbj) + dot(ai, Σbj*ai) + dot(bj, Σai*bj)
    like += 0.5*(log(β) - LOG2PI) - 0.5*β*resid2_mean
  end

  # 2) Priors A|γ, B|γ
  priorA = 0.0
  @inbounds for i in 1:m
    Σai = SigmaA[i]
    ai = @view A_mean[i, :]
    for k in 1:r
      γk = max(float(gamma[k]), 1e-12)
      priorA += 0.5*(log(γk) - LOG2PI) - 0.5*γk*(Σai[k,k] + ai[k]^2)
    end
  end

  priorB = 0.0
  @inbounds for j in 1:n
    Σbj = SigmaB[j]
    bj = @view B_mean[j, :]
    for k in 1:r
      γk = max(float(gamma[k]), 1e-12)
      priorB += 0.5*(log(γk) - LOG2PI) - 0.5*γk*(Σbj[k,k] + bj[k]^2)
    end
  end

  # 3) Prior γ（点值）
  priorGamma = 0.0
  @inbounds for k in 1:r
    γk = max(float(gamma[k]), 1e-12)
    priorGamma += (float(c)-1)*log(γk) - γk/float(d) - float(c)*log(float(d)) - loggamma_fallback(float(c))
  end

  # 4) Entropy
  HA = 0.0
  @inbounds for i in 1:m
    F = cholesky(Symmetric(SigmaA[i]))
    HA += 0.5*( r*(1 + LOG2PI) + 2*sum(log, diag(F.U)) )
  end
  HB = 0.0
  @inbounds for j in 1:n
    F = cholesky(Symmetric(SigmaB[j]))
    HB += 0.5*( r*(1 + LOG2PI) + 2*sum(log, diag(F.U)) )
  end

  elbo = like + priorA + priorB + priorGamma + HA + HB
  return elbo, (like=like, priorA=priorA, priorB=priorB, priorGamma=priorGamma, entropyA=HA, entropyB=HB)
end

# ========== 主函数：无覆盖训练 ==========

"""
无覆盖版本的变分矩阵补全（观测加权）。
参数：
  - KAPPA: 观测位置权重倍数（β放大倍数），10~50 常用，越大越“贴合”
  - c,d: γ 的先验超参；增大 d 可减弱收缩，便于拟合
  - eta: 更新阻尼（0.7~1.0），KAPPA 大时建议 0.8~0.95 提升稳定性
  - compute_elbo: 计算并记录 ELBO（用加权后的 beta）
  - eval_powerflow: 收敛后是否调用潮流进行“评估”（不覆盖 X）
"""
function run_stage2_no_overwrite(branch, daily_predictions;
                                 monitor_buses=Set([8, 12]),
                                 KAPPA::Float64=20.0,
                                 c::Float64=1e-4, d::Float64=1e-3,
                                 max_iter::Int=400,
                                 tolerance::Float64=1e-6,
                                 eta::Float64=0.9,
                                 print_every::Int=20,
                                 compute_elbo::Bool=true,
                                 eval_powerflow::Bool=false,
                                 root_bus::Int=1, Vref::Float64=1.0)

  # 构造观测与β
  observed_matrix_Z, observed_pairs, monitored_obs =
    build_observed_matrix_Z(daily_predictions; monitor_buses=monitor_buses)
  noise_precision_β = build_noise_precision_beta(daily_predictions)

  Z = Array{Float64}(observed_matrix_Z)
  beta = Array{Float64}(noise_precision_β)

  beta_weighted = copy(beta)
  weight_observed_beta!(beta_weighted, observed_pairs, KAPPA)

  # SVD 初始化
  svd_res = svd(Z)
  r = min(5, minimum(size(Z)))
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
    :rel_change => Float64[],
    :elbo => Float64[],
    :mae_obs => Float64[],
    :rmse_obs => Float64[]
  )

  # 迭代
  for it in 1:max_iter
    # 更新 A
    for i in 1:size(A_mean, 1)
      βBtB = cal_beta_BTB_i(i, B_mean, Σb_list, observed_pairs, beta_weighted, latent_dim)
      Σa_new = cal_sigma_a_i(βBtB, γ)
      a_new = cal_a_mean_i(i, B_mean, Σa_new, observed_pairs, beta_weighted, Z)
      Σa_list[i] = eta*Σa_new + (1-eta)*Σa_list[i]
      @inbounds @views A_mean[i, :] .= eta*a_new .+ (1-eta).*A_mean[i, :]
    end

    # 更新 B
    for j in 1:size(B_mean, 1)
      βAtA = cal_beta_ATA_j(j, A_mean, Σa_list, observed_pairs, beta_weighted, latent_dim)
      Σb_new = cal_sigma_b_j(βAtA, γ)
      b_new = cal_b_mean_j(j, A_mean, Σb_new, observed_pairs, beta_weighted, Z)
      Σb_list[j] = eta*Σb_new + (1-eta)*Σb_list[j]
      @inbounds @views B_mean[j, :] .= eta*b_new .+ (1-eta).*B_mean[j, :]
    end

    # 更新 γ（点估计）
    for k in 1:length(γ)
      aTa = cal_aTa_i(k, A_mean, Σa_list)
      bTb = cal_bTb_j(k, B_mean, Σb_list)
      γ[k] = clamp((2c + size(A_mean,1) + size(B_mean,1)) / (aTa + bTb + 2d), 1e-8, 1e8)
    end

    # 收敛监控
    X_new = Array{Float64}(A_mean * B_mean')
    numerator = norm(X_new - X_old)
    denominator = max(norm(X_old), 1e-12)
    rel = numerator / denominator
    push!(history[:rel_change], rel)

    mae_obs, rmse_obs = obs_metrics(Z, X_new, observed_pairs)
    push!(history[:mae_obs], mae_obs)
    push!(history[:rmse_obs], rmse_obs)

    if compute_elbo
      elbo, _ = elbo_point_gamma(Z, beta_weighted, observed_pairs, A_mean, B_mean, Σa_list, Σb_list, γ, c, d)
      push!(history[:elbo], elbo)
    end

    if (it % print_every == 0) || (rel < tolerance)
      @printf "it=%4d  rel=%.3e  mae_obs=%.4g  rmse_obs=%.4g" it rel mae_obs rmse_obs
      if compute_elbo
        @printf "  ELBO=%.6e" last(history[:elbo])
      end
      println()
    end

    X_old = X_new
    if rel < tolerance
      println("Converged at iter=$it (rel=$(rel))")
      break
    end
  end

  # 收敛后可选：仅评估潮流，不覆盖
  pf_eval = nothing
  if eval_powerflow
    P_inj = X_old[:, 1] ./ 10
    Q_inj = X_old[:, 2] ./ 10
    Vb    = X_old[:, 5]
    V_sol, θ_sol, Pinj_sol, Qinj_sol, Vr_sol, Vi_sol =
      ac_nodal_injection(P_inj, Q_inj, Vb, branch, 1, 1.0, 0.0, observed_pairs; verbose=false)
    pf_eval = (V=V_sol, θ=θ_sol, Pinj=Pinj_sol, Qinj=Qinj_sol, Vr=Vr_sol, Vi=Vi_sol)
  end

  return (
    X = X_old,
    history = history,
    A_mean = A_mean, Σa_list = Σa_list,
    B_mean = B_mean, Σb_list = Σb_list,
    gamma = γ,
    beta_weighted = beta_weighted,
    observed_pairs = observed_pairs,
    pf_eval = pf_eval
  )
end

# ========== 评估所有拓扑（不覆盖版本） ==========

mutable struct RunSuccessRecord
  idx::Int
  prob::Float64
  total_loss::Float64
  count_used::Int
end

function evaluate_branches_over_samples_no_overwrite(result, branch_list, prob_list;
                                                     step::Int=43100,
                                                     total_len::Int=90000,
                                                     tol::Float64=1e-6,
                                                     monitor_buses=Set([8,12]),
                                                     KAPPA::Float64=20.0,
                                                     c::Float64=1e-4, d::Float64=1e-3,
                                                     eta::Float64=0.9)
  start_counts = collect(90000:step:total_len)
  nB = length(branch_list)
  total_loss = fill(0.0, nB)
  used_count = fill(0, nB)

  for (si, start_count) in enumerate(start_counts)
    daily_predictions = generate_daily_predictions(result, start_count, 1)
    @showprogress for (idx, brch) in enumerate(branch_list)
      try
        res = run_stage2_no_overwrite(brch, daily_predictions;
                  monitor_buses=monitor_buses,
                  KAPPA=KAPPA, c=c, d=d,
                  max_iter=600, tolerance=tol,
                  eta=eta, print_every=50,
                  compute_elbo=false,
                  eval_powerflow=false)
        # 采用观测 MAE+RMSE 作为损失汇总（你也可只用 RMSE）
        loss = last(res.history[:mae_obs]) + last(res.history[:rmse_obs])
        total_loss[idx] += loss
        used_count[idx] += 1
      catch e
        @warn "Branch idx=$idx sample si=$si failed with error: $e"
      end
    end
    @info "Finished sample si=$si (start=$start_count)"
  end

  records = RunSuccessRecord[]
  for i in 1:nB
    push!(records, RunSuccessRecord(i, prob_list[i], total_loss[i], used_count[i]))
  end
  valid = filter(r -> r.count_used > 0, records)
  sort!(valid, by = r -> r.total_loss, rev = false)
  topk = first(valid, min(5, length(valid)))

  return (records=records, top5=topk, start_counts=start_counts, step=step)
end

# ========== 顶层入口示例 ==========

function main_no_overwrite(result; monitor_buses=Set([8,12]),
                           KAPPA::Float64=20.0,
                           c::Float64=1e-4, d::Float64=1e-3,
                           eta::Float64=0.9)
  branch_based = read_topology_mat("D:/luosipeng/matpower8.1/pf_parallel_out/topology.mat")
  branch_list, prob_list = generate_branch_list_with_prior(
    branch_based; param_sets=nothing, param_source_rows=(35,5,1), per_line_cartesian=true
  )
  res_eval = evaluate_branches_over_samples_no_overwrite(result, branch_list, prob_list;
               step=43100, tol=1e-6, monitor_buses=monitor_buses,
               KAPPA=KAPPA, c=c, d=d, eta=eta)
  println("Evaluation finished. total branches=$(length(branch_list))")
  println("Valid branches with at least 1 sample used = $(length(filter(r -> r.count_used>0, res_eval.records)))")
  println("Top-5 branches (by cumulative loss):")
  for (rank, r) in enumerate(res_eval.top5)
    println("[$rank] idx=$(r.idx), prior_prob=$(r.prob), total_loss=$(r.total_loss), used_samples=$(r.count_used)")
  end
  return res_eval
end

main_no_overwrite(result; monitor_buses=Set([8,12]), KAPPA=20.0, c=1e-4, d=1e-3, eta=0.9)