using LinearAlgebra
using SparseArrays
using Statistics

include("../src/build_observed_matrix_z.jl")
include("../src/build_noise_precision_beta.jl")
include("../src/implement_data.jl")
include("../src/matrix_completion.jl")
include("../ios/read_mat.jl")
include("../src/build_admittance_matrix.jl")
include("../src/physic_constraint.jl")
include("../src/calculate_matrix_uncertainty.jl")

function run_stage2_test_corrected()
    # ================= 1. 数据与模型初始化 =================
    branch = read_topology_mat("D:/luosipeng/matpower8.1/pf_parallel_out/topology.mat")
    Y_bus = build_three_phase_admittance_matrix(branch)
    w = build_w_matrix(Y_bus)
    abs_w = abs.(w)
    M = build_matrix_M(Y_bus, w)
    K_mat = build_matrix_K(M, w)
    
    # 生成数据
    daily_predictions = generate_daily_predictions(result, 1, 1)
    observed_matrix_Z, observed_pairs, monitored_obs = build_observed_matrix_Z(daily_predictions)
    noise_precision_β = build_noise_precision_beta(daily_predictions)

    # 参数设置
    tolerance = 1e-6
    c_param = 1e-7
    d_param = 1e-7
    max_iter = 400
    baseMVA = 10.0
    
    # 定义列索引
    idx_P, idx_Q = 1, 2
    idx_Vr, idx_Vi, idx_V = 3, 4, 5

    # ================= 2. SVD 初始化 =================
    svd_res = svd(observed_matrix_Z)
    r = 5 # Rank
    U_r = svd_res.U[:, 1:r]
    Σ_r = svd_res.S[1:r]
    Vt_r = svd_res.Vt[1:r, :]

    sqrtD = Diagonal(sqrt.(Σ_r))
    A_mean = Array{Float64}(U_r * sqrtD)
    B_mean = Array{Float64}(Vt_r' * sqrtD)

    # 初始化协方差与超参数
    α = 1e-3
    Σa0 = α .* Matrix{Float64}(I, r, r)
    Σb0 = α .* Matrix{Float64}(I, r, r)
    Σa_list = [copy(Σa0) for _ in 1:size(A_mean, 1)]
    Σb_list = [copy(Σb0) for _ in 1:size(B_mean, 1)]
    γ = fill(1.0, r)

    # X_old 初始值 (SVD 结果)
    X_old = Array{Float64}(A_mean * B_mean')
    latent_dim = size(A_mean, 2)

    history = Dict{Symbol, Vector{Float64}}(:rel_change => Float64[])

    # ================= 3. 构建反馈闭环结构 (关键修复) =================
    
    # 3.1 识别哪些位置是真实观测的 (用于保护数据)
    observed_rows_Vr = Set{Int}()
    observed_rows_Vi = Set{Int}()
    observed_rows_V  = Set{Int}()
    
    for (r, col) in observed_pairs
        if col == idx_Vr push!(observed_rows_Vr, r)
        elseif col == idx_Vi push!(observed_rows_Vi, r)
        elseif col == idx_V  push!(observed_rows_V, r)
        end
    end
    
    println("Real Observations - Real: $(length(observed_rows_Vr)), Imag: $(length(observed_rows_Vi)), Mag: $(length(observed_rows_V))")

    # 3.2 构建“增强”观测集 (Augmented Sets)
    # 我们将把物理推导出的电压作为“伪量测”加入到这些集合中
    augmented_pairs = copy(observed_pairs)
    augmented_Z = copy(observed_matrix_Z) 
    augmented_beta = copy(noise_precision_β)
    
    # 记录哪些位置是缺失的，需要靠物理方程填补
    missing_indices = Vector{Tuple{Int, Int}}()
    
    # 物理约束的置信度 (Precision)。
    # 1e4 表示我们相当信任物理方程 (根据实际噪声水平调整)
    beta_phys = 1e4 

    for i in 1:size(observed_matrix_Z, 1)
        # 检查电压实部
        if !(i in observed_rows_Vr)
            push!(augmented_pairs, (i, idx_Vr))
            push!(missing_indices, (i, idx_Vr))
            augmented_beta[i, idx_Vr] = beta_phys
            # 初始填充可以用 SVD 的结果，防止零启动
            augmented_Z[i, idx_Vr] = X_old[i, idx_Vr]
        end
        # 检查电压虚部
        if !(i in observed_rows_Vi)
            push!(augmented_pairs, (i, idx_Vi))
            push!(missing_indices, (i, idx_Vi))
            augmented_beta[i, idx_Vi] = beta_phys
            augmented_Z[i, idx_Vi] = X_old[i, idx_Vi]
        end
        # 检查电压幅值
        if !(i in observed_rows_V)
            push!(augmented_pairs, (i, idx_V))
            push!(missing_indices, (i, idx_V))
            augmented_beta[i, idx_V] = beta_phys
            augmented_Z[i, idx_V] = X_old[i, idx_V]
        end
    end
    
    println("Augmented pairs size: $(length(augmented_pairs)) (Original: $(length(observed_pairs)))")

    # ================= 4. 主迭代循环 =================
    for it in 1:max_iter
        
        # --- [Step A] 反馈步骤 (Feedback Step) ---
        # 将上一轮经过物理约束修正后的 X_old 中的电压值，填入 augmented_Z
        # 这样 A 和 B 在更新时就能“看到”物理约束的结果
        if it > 1
            for (r, c) in missing_indices
                augmented_Z[r, c] = X_old[r, c]
            end
        end

        # --- [Step B] 贝叶斯更新 A (使用 augmented 数据) ---
        for i in 1:size(A_mean, 1)
            # 注意：传入 augmented_pairs, augmented_beta, augmented_Z
            βBtB = cal_beta_BTB_i(i, B_mean, Σb_list, augmented_pairs, augmented_beta, latent_dim)
            Σa_list[i] = cal_sigma_a_i(βBtB, γ)
            A_mean[i, :] = cal_a_mean_i(i, B_mean, Σa_list[i], augmented_pairs, augmented_beta, augmented_Z)
        end

        # --- [Step C] 贝叶斯更新 B (使用 augmented 数据) ---
        for j in 1:size(B_mean, 1)
            βAtA = cal_beta_ATA_j(j, A_mean, Σa_list, augmented_pairs, augmented_beta, latent_dim)
            Σb_list[j] = cal_sigma_b_j(βAtA, γ)
            B_mean[j, :] = cal_b_mean_j(j, A_mean, Σb_list[j], augmented_pairs, augmented_beta, augmented_Z)
        end

        # --- [Step D] 更新超参数 Gamma ---
        for k in 1:length(γ)
            aTa = cal_aTa_i(k, A_mean, Σa_list)
            bTb = cal_bTb_j(k, B_mean, Σb_list)
            γ[k] = clamp((2 * c_param + size(A_mean,1) + size(B_mean,1)) / (aTa + bTb + 2 * d_param), 1e-6, 1e6)
        end

        # --- [Step E] 重构矩阵 ---
        X_new = Array{Float64}(A_mean * B_mean')

        # --- [Step F] 应用物理约束 (Physical Projection) ---
        try
            P_est = X_new[:, idx_P] ./ baseMVA
            Q_est = X_new[:, idx_Q] ./ baseMVA
            
            if length(P_est) * 2 == size(K_mat, 2)
                 PQ_vec = vcat(P_est, Q_est)
                 
                 # 1. 计算物理一致的电压
                 v_complex_phys = w + M * PQ_vec       # Eq 14
                 v_mag_phys     = abs_w + K_mat * PQ_vec # Eq 15
                 
                 # 2. 强制覆盖未观测位置 (Projection)
                 for i in 1:size(X_new, 1)
                     # 仅更新那些在 missing_indices 里的位置
                     # (或者用之前的 Set 判断，效果一样)
                     
                     if !(i in observed_rows_Vr)
                         X_new[i, idx_Vr] = real(v_complex_phys[i])
                     end
                     
                     if !(i in observed_rows_Vi)
                         X_new[i, idx_Vi] = imag(v_complex_phys[i])
                     end
                     
                     if !(i in observed_rows_V)
                         X_new[i, idx_V] = v_mag_phys[i]
                     end
                 end
            else
                @warn "Dimension mismatch in physical constraint."
            end
        catch e
            println("Physical constraint error: $e")
        end

        # --- [Step G] 收敛性检查 ---
        numerator = norm(X_new - X_old)
        denominator = max(norm(X_old), 1e-12)
        rel = numerator / denominator

        if it % 10 == 0 || it == 1
            println("Iter $it: rel_change = $rel")
        end
        push!(history[:rel_change], rel)
        
        X_old = copy(X_new) # 更新 X_old 供下一轮反馈使用

        if rel < tolerance
            println("Converged at iter=$it, rel=$(rel)")
            break
        end
    end

    if isempty(history[:rel_change]) || history[:rel_change][end] ≥ tolerance
        @warn "Not below tolerance yet."
    end

    # ================= [新增] 计算概率信息 =================
    println("Calculating uncertainty quantification...")
    var_X, std_X = calculate_matrix_uncertainty(A_mean, B_mean, Σa_list, Σb_list)
    # 提取最终结果
    Pij_sol = X_old[:, idx_P]
    Qij_sol = X_old[:, idx_Q]
    V_sol   = X_old[:, idx_V]

    # 提取最终结果 (标准差/不确定性)
    # 这告诉我们估计结果的“可信度”
    Pij_std = std_X[:, idx_P]
    Qij_std = std_X[:, idx_Q]
    V_std   = std_X[:, idx_V]

    return (
        X_mean = X_old,          # 估计值的均值矩阵
        X_std  = std_X,          # 估计值的标准差矩阵 (概率信息)
        history = history,
        flows = (P = Pij_sol, Q = Qij_sol, V = V_sol),
        uncertainty = (P_std = Pij_std, Q_std = Qij_std, V_std = V_std)
    )
end

# 运行
result_bmc = run_stage2_test_corrected()
