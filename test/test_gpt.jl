using LinearAlgebra
using SparseArrays
using Statistics

include("../src/build_observed_matrix_z.jl")
include("../src/build_noise_precision_beta.jl")
include("../src/implement_data.jl")
include("../src/matrix_completion.jl")
include("../ios/read_mat.jl")
include("../src/lindistflow.jl")             # 如果已有，可继续使用（可选）
include("../src/power_flow_optimal.jl")      # 旧OPF（可选保留）
include("../src/likelihood_gaussian.jl")
include("../src/physic_constraint_opf.jl")   # 旧OPF（可选保留）
include("../gpt_src/physics_factor.jl")          # 新增：概率物理因子更新

function run_stage2_test(; mode::Symbol = :ac, λ_pf::Float64 = 1e7, η::Float64 = 0.3)
    branch = read_topology_mat("D:/luosipeng/matpower8.1/pf_parallel_out/topology.mat")
    daily_predictions = generate_daily_predictions(result, 1, 1)
    observed_matrix_Z, observed_pairs, monitored_obs = build_observed_matrix_Z(daily_predictions; monitor_buses=Set([8, 12]))
    noise_precision_β = build_noise_precision_beta(daily_predictions)

    tolerance = 1e-6
    c = 1e-7
    d = 1e-7
    max_iter = 400

    root_bus = 1
    Vref = 1.0

    observed_matrix_Z = Array{Float64}(observed_matrix_Z)
    noise_precision_β = Array{Float64}(noise_precision_β)

    X0 = copy(observed_matrix_Z)
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

    history = Dict{Symbol, Vector{Float64}}(:rel_change => Float64[])

    for it in 1:max_iter
        # 低秩更新
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
        for k in 1:length(γ)
            aTa = cal_aTa_i(k, A_mean, Σa_list)
            bTb = cal_bTb_j(k, B_mean, Σb_list)
            γ[k] = clamp((2c + size(A_mean,1) + size(B_mean,1)) / (aTa + bTb + 2d), 1e-6, 1e6)
        end

        X_mc = Array{Float64}(A_mean * B_mean')

        # ===== 概率物理约束（软约束，高斯因子，保留UQ）=====
        if mode == :lin
            # 单相 LinDistFlow：y=[P,Q,V]，一次高斯更新
            X_phys, _, _ = physics_update_lindistflow(
                X_mc, A_mean, Σa_list, B_mean, Σb_list, branch, root_bus;
                Vref=Vref, λ_pf=λ_pf, colmap=(P=1,Q=2,V=5), cols_branch=(1,2,3,4,11)
            )
        else
            # AC 平衡：y=[P,Q,Vr,Vi]，迭代线性化 + 高斯更新
            X_phys, _, _ = physics_update_ac_linearized(
                X_mc, A_mean, Σa_list, B_mean, Σb_list, branch, root_bus;
                Vref=Vref, λ_pf=λ_pf, iters=3,
                colmap=(P=1,Q=2,Vr=3,Vi=4,V=5), cols_branch=(1,2,3,4,11)
            )
        end

        # 与MC结果做阻尼融合（稳定）
        X_new = η * X_phys + (1-η) * X_mc

        rel = norm(X_new - X_old) / max(norm(X_old), 1e-12)
        push!(history[:rel_change], rel)
        X_old = X_new
        η = min(η * 1.3, 0.8)  # 逐步加大物理权重
        if rel < tolerance
            println("Converged at iter=$it, rel=$(rel)")
            break
        end
    end

    # 计算ELBO（你的实现会用 X_old 等）
    elbo_result = compute_elbo_with_physics(
        X_old, A_mean, B_mean, Σa_list, Σb_list, γ,
        observed_matrix_Z, noise_precision_β, observed_pairs
    )

    return (X = X_old, history = history,
            flows = (P = Float64[], Q = Float64[], V = Float64[]),
            injections = (P = Float64[], Q = Float64[]), elbo = elbo_result)
end

X, history, flows, injections, elbo = run_stage2_test()