include("../src/build_observed_matrix_z.jl")
include("../src/build_noise_precision_beta.jl")
include("../src/implement_data.jl")
include("../src/matrix_completion.jl")
include("../ios/read_mat.jl")
include("../src/build_admittance_matrix.jl")
include("../src/physic_constraint.jl")
include("../src/calculate_matrix_uncertainty.jl")
include("../src/get_aligned_physics_matrix.jl")
include("../src/build_ground_truth_vector.jl")


function mtgpbmc(mtgp_result, noise_level, lack_a, lack_b, lack_c; confidence_level=0.95)
    bmc_results, phase_info = run_stage2_test_corrected(mtgp_result, noise_level, lack_a, lack_b, lack_c)
    X_mean = bmc_results.X_mean
    X_std = bmc_results.X_std  # ✅ 获取不确定度
    
    Pij_sol = X_mean[:, 1]
    Qij_sol = X_mean[:, 2]
    Vr_sol = X_mean[:, 3]
    Vi_sol = X_mean[:, 4]
    V_angle = atan.(Vi_sol, Vr_sol)
    V_mag = X_mean[:, 5]
    
    # ✅ 获取标准差
    P_std = X_std[:, 1]
    Q_std = X_std[:, 2]
    Vr_std = X_std[:, 3]
    Vi_std = X_std[:, 4]
    V_mag_std = X_std[:, 5]
    
    # ✅ 计算角度的不确定度（误差传播）
    # θ = atan(Vi, Vr)，使用一阶泰勒展开
    V_angle_std = sqrt.((Vr_sol .* Vi_std).^2 + (Vi_sol .* Vr_std).^2) ./ (Vr_sol.^2 + Vi_sol.^2)
    
    # 读取真实值
    (voltage_mag_a, voltage_mag_b, voltage_mag_c,
     voltage_ang_a, voltage_ang_b, voltage_ang_c,
     power_p_a, power_p_b, power_p_c,
     power_q_a, power_q_b, power_q_c) = read_all_opendss_data()
    
    # ✅ 使用统一的构造函数
    origin_value_power = build_ground_truth_vector(
        power_p_a, power_p_b, power_p_c, 
        lack_a, lack_b, lack_c
    )
    
    origin_value_theta = build_ground_truth_vector(
        voltage_ang_a .* (π/180), 
        voltage_ang_b .* (π/180), 
        voltage_ang_c .* (π/180),
        lack_a, lack_b, lack_c
    )
    
    origin_value_magnitude = build_ground_truth_vector(
        voltage_mag_a, voltage_mag_b, voltage_mag_c,
        lack_a, lack_b, lack_c
    )

    imputed_value_power = Pij_sol
    imputed_value_theta = V_angle
    imputed_value_magnitude = V_mag

    # ================= 计算点估计误差 =================
    miae_power = sum(abs.(origin_value_power - imputed_value_power)) / length(imputed_value_power) * 100
    miae_theta = sum(abs.(origin_value_theta - imputed_value_theta)) / length(imputed_value_theta) * 100
    mape_voltage = sum(abs.(origin_value_magnitude - imputed_value_magnitude) ./ origin_value_magnitude) / length(imputed_value_magnitude) * 100
    
    # ================= 计算 PICP 和 MPIW =================
    # 根据置信水平计算 z-score（假设正态分布）
    z_score = quantile(Normal(), 0.5 + confidence_level / 2)
    
    # --- 功率 P 的 PICP 和 MPIW ---
    picp_power, mpiw_power = calculate_picp_mpiw(
        origin_value_power, imputed_value_power, P_std, z_score
    )
    
    # --- 功率 Q 的 PICP 和 MPIW ---
    picp_q, mpiw_q = calculate_picp_mpiw(
        build_ground_truth_vector(power_q_a, power_q_b, power_q_c, lack_a, lack_b, lack_c),
        Qij_sol, Q_std, z_score
    )
    
    # --- 电压角度的 PICP 和 MPIW ---
    picp_theta, mpiw_theta = calculate_picp_mpiw(
        origin_value_theta, imputed_value_theta, V_angle_std, z_score
    )
    
    # --- 电压幅值的 PICP 和 MPIW ---
    picp_voltage, mpiw_voltage = calculate_picp_mpiw(
        origin_value_magnitude, imputed_value_magnitude, V_mag_std, z_score
    )
    
    # ================= 返回结果 =================
    return (
        # 点估计误差
        miae_power = miae_power,
        miae_theta = miae_theta,
        mape_voltage = mape_voltage,
        
        # PICP（覆盖概率，越接近 confidence_level 越好）
        # picp_power = picp_power,
        # picp_q = picp_q,
        picp_theta = picp_theta,
        picp_voltage = picp_voltage,
        
        # MPIW（区间宽度，越小越好）
        # mpiw_power = mpiw_power,
        # mpiw_q = mpiw_q,
        mpiw_theta = mpiw_theta,
        mpiw_voltage = mpiw_voltage
    )
end

# ================= 辅助函数：计算 PICP 和 MPIW =================
"""
计算预测区间覆盖概率 (PICP) 和平均预测区间宽度 (MPIW)

参数:
- true_values: 真实值向量
- predicted_values: 预测值向量
- std_values: 标准差向量
- z_score: 置信水平对应的 z-score（如 95% 对应 1.96）

返回:
- picp: 预测区间覆盖概率（0-100%）
- mpiw: 平均预测区间宽度
"""
function calculate_picp_mpiw(true_values, predicted_values, std_values, z_score)
    n = length(true_values)
    
    # 计算预测区间
    lower_bound = predicted_values .- z_score .* std_values
    upper_bound = predicted_values .+ z_score .* std_values
    
    # PICP: 真实值落在预测区间内的比例
    coverage = sum((true_values .>= lower_bound) .& (true_values .<= upper_bound))
    picp = (coverage / n) * 100  # 转换为百分比
    
    # MPIW: 预测区间的平均宽度
    interval_widths = upper_bound .- lower_bound
    mpiw = mean(interval_widths)
    
    return picp, mpiw
end


function adaptive_rank_selection(observed_matrix_Z::Matrix{Float64}, noise_level::Float64)
    """
    优化后的自适应秩选择：
    结合 '能量占比' 和 '相对阈值' 双重标准，防止高噪下的过拟合。
    """
    
    svd_res = svd(observed_matrix_Z)
    S = svd_res.S
    max_S = maximum(S)
    
    # --- 策略 1: 相对幅度阈值 (Relative Magnitude Threshold) ---
    # 逻辑：如果一个特征值比最大特征值小太多，它大概率是噪声。
    # 修改点：直接使用 noise_level 作为比例，不再乘以 0.1
    # 例如：10% 噪声下，任何小于主成分 10% 的细节都可能是噪声，直接丢弃。
    # 我们设置一个基底 0.02 (2%) 防止在低噪下截断过度。
    magnitude_threshold = max(noise_level, 0.02) * max_S
    r_magnitude = sum(S .> magnitude_threshold)
    
    # --- 策略 2: 累积能量截断 (Cumulative Energy Preservation) ---
    # 逻辑：保留 99% 的能量通常就够了，剩下的 1% 往往是噪声。
    # 在高噪 (10%) 下，我们甚至只保留 95% 的能量，主动丢弃噪声。
    total_energy = sum(S .^ 2)
    target_energy_ratio = noise_level >= 0.1 ? 0.95 : 0.99
    
    current_energy = 0.0
    r_energy = 0
    for (k, val) in enumerate(S)
        current_energy += val^2
        r_energy = k
        if current_energy / total_energy >= target_energy_ratio
            break
        end
    end

    # --- 综合决策 ---
    # 取两者的最小值 (Conservative Principle)，宁可欠拟合也不要过拟合噪声
    r_final = min(r_magnitude, r_energy)
    
    # --- 物理硬约束 ---
    # [关键修改] 强制最大秩为 4 
    # 无论 FAD 多大，我们只允许 2, 3, 4 这三个档位。
    # Rank 5+ 在配电网中通常都是拟合噪声。
    r_final = clamp(r_final, 2, 4)
    
    println("������ Adaptive rank selection (Optimized):")
    println("  Noise level: $(noise_level)")
    println("  Max Singular Val: $(round(max_S, digits=4))")
    println("  Threshold (Mag): $(round(magnitude_threshold, digits=4)) -> Rank: $r_magnitude")
    println("  Threshold (Eng): $(target_energy_ratio*100)% -> Rank: $r_energy")
    println("  Selected Final Rank: $r_final")
    println("  Top 8 Singular values: $(round.(S[1:min(8, length(S))], digits=4))")
    
    return r_final
end


function run_stage2_test_corrected(mtgp_result, noise_level, lack_a, lack_b, lack_c)
    # ================= 1. 数据准备 =================
    baseMVA = 10.0
    
    daily_predictions = generate_daily_predictions(mtgp_result, 432000, 1)
    observed_matrix_Z, observed_pairs, monitored_obs, phase_info = build_observed_matrix_Z(daily_predictions, lack_a, lack_b, lack_c, baseMVA)
    noise_precision_β = build_noise_precision_beta(daily_predictions, phase_info)

    observed_matrix_Z = Float64.(observed_matrix_Z)
    noise_precision_β = Float64.(noise_precision_β)

    # 🔧 关键调参1：数据权重封顶 - 防止过度信任噪声观测
    n_rows, n_cols = size(observed_matrix_Z)
    obs_ratio = length(observed_pairs) / (n_rows * n_cols)
    
    # FAD越高，观测越多，噪声累积越严重，需要调整beta上限
    # 🔧 2026-01-31 优化：FAD=20%时最优参数 data_beta_cap=250
    if obs_ratio >= 0.15
        data_beta_cap = 250.0   # FAD>=15%: 优化后的参数
    else
        data_beta_cap = 500.0  # FAD<15%: 宽松限制
    end
    replace!(noise_precision_β, Inf => data_beta_cap, NaN => 1.0)
    noise_precision_β .= clamp.(noise_precision_β, 0.1, data_beta_cap)
    println("📊 Observation ratio: $(round(obs_ratio*100, digits=1))%, Beta cap: $data_beta_cap")

    w, M, K_mat = get_aligned_physics_matrices(phase_info, baseMVA)
    abs_w = abs.(w)

    tolerance = 1e-5
    c_param = 1e-7
    d_param = 1e-7
    max_iter = 20000
    
    idx_P, idx_Q = 1, 2
    idx_Vr, idx_Vi, idx_V = 3, 4, 5

    # ================= 2. 初始化 =================
    init_matrix = build_full_matrix_from_predictions(daily_predictions, phase_info)
    
    svd_res = svd(init_matrix)
    r = 4
    r = clamp(r, 4, min(size(init_matrix)...))

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

    X_old = copy(init_matrix)
    latent_dim = size(A_mean, 2)
    history = Dict{Symbol, Vector{Float64}}(:rel_change => Float64[])

    # ================= 3. 识别观测位置 =================
    observed_rows_Vr = Set{Int}()
    observed_rows_Vi = Set{Int}()
    observed_rows_V  = Set{Int}()
    observed_rows_P  = Set{Int}()
    observed_rows_Q  = Set{Int}()
    
    for (row, col) in observed_pairs
        if col == idx_P  push!(observed_rows_P, row)
        elseif col == idx_Q  push!(observed_rows_Q, row)
        elseif col == idx_Vr push!(observed_rows_Vr, row)
        elseif col == idx_Vi push!(observed_rows_Vi, row)
        elseif col == idx_V  push!(observed_rows_V, row)
        end
    end
    
    n_voltage_obs = length(observed_rows_V) + length(observed_rows_Vr) + length(observed_rows_Vi)
    println("🔌 Voltage observations: $n_voltage_obs")

    # ================= 4. 构建增强观测集 =================
    augmented_pairs = copy(observed_pairs)
    augmented_Z = copy(observed_matrix_Z) 
    augmented_beta = copy(noise_precision_β)
    missing_indices = Vector{Tuple{Int, Int}}()
    
    # 🔧 关键调参2：物理权重 - 需要与data_beta平衡
    # 🔧 2026-01-31 优化：FAD=20%时最优参数 beta_phys=50000
    if obs_ratio >= 0.15
        beta_phys = 50000.0   # FAD>=15%: 强物理约束
    else
        beta_phys = clamp(8000.0 / (sqrt(noise_level) + 0.01), 3000.0, 90000.0)
    end
    println("⚖️  Physics Beta: $(round(beta_phys, digits=1))")

    for i in 1:size(observed_matrix_Z, 1)
        for (col, obs_set) in [(idx_Vr, observed_rows_Vr), 
                               (idx_Vi, observed_rows_Vi), 
                               (idx_V, observed_rows_V)]
            if !(i in obs_set)
                push!(augmented_pairs, (i, col))
                push!(missing_indices, (i, col))
                augmented_beta[i, col] = beta_phys
                augmented_Z[i, col] = X_old[i, col]
            end
        end
    end

    # ================= 6. 主迭代循环 =================
    for it in 1:max_iter
        
        if it > 1
            for (r_idx, c_idx) in missing_indices
                augmented_Z[r_idx, c_idx] = X_old[r_idx, c_idx]
            end
        end

        # 贝叶斯更新 A
        for i in 1:size(A_mean, 1)
            βBtB = cal_beta_BTB_i(i, B_mean, Σb_list, augmented_pairs, augmented_beta, latent_dim)
            Σa_list[i] = cal_sigma_a_i(βBtB, γ)
            A_mean[i, :] = cal_a_mean_i(i, B_mean, Σa_list[i], augmented_pairs, augmented_beta, augmented_Z)
        end

        # 贝叶斯更新 B
        for j in 1:size(B_mean, 1)
            βAtA = cal_beta_ATA_j(j, A_mean, Σa_list, augmented_pairs, augmented_beta, latent_dim)
            Σb_list[j] = cal_sigma_b_j(βAtA, γ)
            B_mean[j, :] = cal_b_mean_j(j, A_mean, Σb_list[j], augmented_pairs, augmented_beta, augmented_Z)
        end

        # Gamma 更新
        for k in 1:length(γ)
            aTa = cal_aTa_i(k, A_mean, Σa_list)
            bTb = cal_bTb_j(k, B_mean, Σb_list)
            γ[k] = clamp((2*c_param + size(A_mean,1) + size(B_mean,1)) / (aTa + bTb + 2*d_param), 1e-6, 1e6)
        end

        X_new = Array{Float64}(A_mean * B_mean')

        # ================= 物理约束 + 智能融合 =================
        try
            P_est = X_new[:, idx_P]./(baseMVA*1000.0)
            Q_est = X_new[:, idx_Q]./(baseMVA*1000.0)
            
            if length(P_est) * 2 == size(K_mat, 2)
                PQ_vec = vcat(P_est, Q_est)
                v_complex_phys = w + M * PQ_vec       
                v_mag_phys = abs_w + K_mat * PQ_vec 
                
                # 🔧 关键：裁剪物理电压到合理范围，防止异常值
                # 🔧 2026-01-31 优化：更紧的电压范围 [0.96, 1.04]
                v_mag_phys = clamp.(v_mag_phys, 0.96, 1.04)
                
                # 强制覆盖未观测位置
                for i in 1:size(X_new, 1)
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
            end
        catch e
            @warn "Physical constraint error: $e"
        end
        
        # 🔧 全局电压裁剪：确保所有电压在物理合理范围
        # 🔧 2026-01-31 优化：更紧的电压范围 [0.96, 1.04]
        X_new[:, idx_V] .= clamp.(X_new[:, idx_V], 0.96, 1.04)

        # 收敛检查
        rel = norm(X_new - X_old) / max(norm(X_old), 1e-12)
        if it % 100 == 0 
            println("Iter $it: rel=$rel") 
        end
        push!(history[:rel_change], rel)
        X_old = copy(X_new)
        
        if rel < tolerance
            println("✅ Converged at iter=$it")
            break
        end
    end
    
    println("📈 Final V range: $(round.(extrema(X_old[:, idx_V]), digits=4))")

    var_X, std_X = calculate_matrix_uncertainty(A_mean, B_mean, Σa_list, Σb_list)
    
    return (
        X_mean = X_old,
        X_std  = std_X,
        history = history,
        flows = (P = X_old[:, 1], Q = X_old[:, 2], V = X_old[:, 5]),
        uncertainty = (P_std = std_X[:, 1], Q_std = std_X[:, 2], V_std = std_X[:, 5])
    ), phase_info
end





