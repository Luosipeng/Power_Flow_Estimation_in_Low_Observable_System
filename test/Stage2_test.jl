"""
This part is used to perform sparse Bayesian matrix completion (Stage 2) with
a LinDistFlow-based soft physics projection.
"""
using LinearAlgebra
using SparseArrays

include("../src/build_observed_matrix_z.jl")
include("../src/build_noise_precision_beta.jl")
include("../src/implement_data.jl")
include("../src/matrix_completion.jl")
include("../ios/read_mat.jl")

# -----------------------
# Data
# -----------------------
branch = read_topology_mat("D:/luosipeng/matpower8.1/pf_parallel_out/topology.mat")
daily_predictions = generate_daily_predictions(result, 1, 1)
observed_matrix_Z, observed_pairs = build_observed_matrix_Z(daily_predictions)
observed_matrix_Z[:,1] = -observed_matrix_Z[:,1]/100  # P 取负号
observed_matrix_Z[:,2] = -observed_matrix_Z[:,2]/100  # Q 取负号
noise_precision_β = build_noise_precision_beta(daily_predictions)

# 可选：如果你明确观测到根节点电压为 1.0，可保留这一观测
# observed_matrix_Z[1, 5] = 1.0

# -----------------------
# Hyper-params
# -----------------------
tolerance = 1e-6
c = 1e-4
d = 1e-4
max_iter = 400

# Physics projection controls
η = 1.0         # V 更新步长（可调 0.2~0.5）
λ_reg = 1e-3      # 正规方程脊回归（必要时升至 1e-2）
w_anchor = 500.0  # 根节点锚定强度（可调 80~200）
root_bus = 1
Vref = 1.0
v0 = 0.0          # 统一使用 0.0 基准；root 通过 anchor 拉至 1.0

# -----------------------
# SVD init
# -----------------------
observed_matrix_Z = Array{Float64}(observed_matrix_Z)
noise_precision_β = Array{Float64}(noise_precision_β)

svd_res = svd(observed_matrix_Z)
r = min(5, minimum(size(observed_matrix_Z)))
U_r = svd_res.U[:, 1:r]
Σ_r = svd_res.S[1:r]
Vt_r = svd_res.Vt[1:r, :]

sqrtD = Diagonal(sqrt.(Σ_r))
A_mean = Array{Float64}(U_r * sqrtD)      # m×r
B_mean = Array{Float64}(Vt_r' * sqrtD)    # n×r

# Priors
α = 1e-3
Σa0 = α .* Matrix{Float64}(I, r, r)
Σb0 = α .* Matrix{Float64}(I, r, r)
Σa_list = [copy(Σa0) for _ in 1:size(A_mean, 1)]
Σb_list = [copy(Σb0) for _ in 1:size(B_mean, 1)]
γ = fill(1.0, r)

X_old = Array{Float64}(A_mean * B_mean')
latent_dim = size(A_mean, 2)

M = build_M_from_branch(branch; root=root_bus)
Mv, Mp, Mq = M.Mv, M.Mp, M.Mq
history = Dict{Symbol, Any}(:rel_change=>Float64[], :obj_phys=>Float64[])

history = Dict{Symbol, Any}(
    :rel_change => Float64[],
    :obj_phys   => Float64[]
)
for it in 1:max_iter
    # VI updates
    for i in 1:size(A_mean, 1)
        βBtB = cal_beta_BTB_i(i, B_mean, Σb_list, observed_pairs, noise_precision_β, latent_dim)
        Σa_list[i] = cal_sigma_a_i(βBtB, γ)
        A_mean[i, :] = cal_a_mean_i(i, B_mean, Σa_list[i], observed_pairs, noise_precision_β, observed_matrix_Z)
    end
    for j in 1:size(B_mean, 1)
        βAtA = cal_beta_ATA_j(j, A_mean, Σa_list, observed_pairs, noise_precision_β, latent_dim)
        Σb_list[j] = cal_sigma_b_j(βAtA, γ)
        B_mean[j, :] = cal_b_mean_j(j, A_mean, Σb_list[j], observed_pairs, noise_precision_β, observed_matrix_Z)
    end
    # γ update with clamp
    for k in 1:length(γ)
        aTa = cal_aTa_i(k, A_mean, Σa_list)
        bTb = cal_bTb_j(k, B_mean, Σb_list)
        γ[k] = clamp((2c + size(A_mean,1) + size(B_mean,1)) / (aTa + bTb + 2d), 1e-6, 1e6)
    end

    # reconstruction
    X_new = Array{Float64}(A_mean * B_mean')
    PD = X_new[:, 1]; QD = X_new[:, 2]; Vb = X_new[:, 5]
    # monitor with v0=0.0 (consistent)
    res_phys = Mv*Vb + Mp*PD + Mq*QD .- v0
    push!(history[:obj_phys], norm(res_phys))

    # physics projection (soft, only V)
    X_new = project_X_with_linphys(X_new, 1, 2, 5, Mv, Mp, Mq, v0;
        λ_reg=λ_reg, root=root_bus, Vref=Vref, anchor=true, w_anchor=w_anchor, η=η)

    # rel change
    numerator = norm(X_new - X_old)
    denominator = max(norm(X_old), 1e-12)
    rel = numerator / denominator
    push!(history[:rel_change], rel)
    X_old = X_new

    # logging
    if it % 10 == 0
        println("it=$it, rel=$(round(rel, sigdigits=4)), phys=$(round(history[:obj_phys][end], sigdigits=4)), η=$(η), λ=$(λ_reg), w=$(w_anchor)")
    end

    # simple convergence
    if rel < tolerance
        println("Converged at iter=$it, rel=$(rel), phys=$(history[:obj_phys][end])")
        break
    end
end

if history[:rel_change][end] ≥ tolerance
    @warn "Not below tolerance yet. tail(rel)=$(history[:rel_change][max(end-4,1):end])"
end

println("Final phys residual norm: ", history[:obj_phys][end])
println("V column head (expected ~1 then decreasing): ", X_old[1:min(end,8), 5])

"""
LinDistFlow 方程（配电网线性化模型）:
A' * Pij + PD = 0
A' * Qij + QD = 0
A * V - v0 * ones(1, nbus) = 2 * Dr * Pij + 2 * Di * Qij
"""