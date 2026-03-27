include("../src/build_ground_truth_vector.jl")
function generate_linear_daily_predictions(result::LinearInterpolationResult, day::Int, time_step::Int)
    """
    生成所有传感器的预测（包括零负荷节点）
    """
    sensors = Dict{String, Any}()
    
    # ✅ 构建全局索引 → 活跃索引的映射
    global_to_active = Dict{Int, Int}()
    for (active_idx, global_idx) in enumerate(result.active_indices)
        global_to_active[global_idx] = active_idx
    end
    
    # ✅ 遍历原始数据中的所有传感器
    for s in 1:result.original_data.S
        sensor_name = result.original_data.sensor_names[s]
        is_zero = result.original_data.is_zero_load[s]
        
        if is_zero
            # ✅ 零负荷节点：返回零值
            sensors[sensor_name] = Dict{String, Any}(
                "prediction_mean" => [0.0],
                "prediction_std" => [0.0],
                "sensor_type" => get_sensor_type(sensor_name),
                "measurement_type" => get_measurement_type(sensor_name),
                "is_zero_load" => true
            )
        else
            # ✅ 活跃传感器：使用插值预测
            if !haskey(global_to_active, s)
                @warn "传感器 $sensor_name 标记为活跃但未找到映射！"
                sensors[sensor_name] = Dict{String, Any}(
                    "prediction_mean" => [0.0],
                    "prediction_std" => [0.0],
                    "sensor_type" => get_sensor_type(sensor_name),
                    "measurement_type" => get_measurement_type(sensor_name),
                    "is_zero_load" => true
                )
                continue
            end
            
            active_idx = global_to_active[s]
            x_test = Float32[time_step - 1]
            
            try
                y_pred, σ_pred = linear_predict(result, active_idx, x_test)
                
                sensors[sensor_name] = Dict{String, Any}(
                    "prediction_mean" => [Float64(y_pred[1])],
                    "prediction_std" => [Float64(σ_pred[1])],
                    "sensor_type" => get_sensor_type(sensor_name),
                    "measurement_type" => get_measurement_type(sensor_name),
                    "is_zero_load" => false
                )
            catch e
                @warn "传感器 $sensor_name 预测失败: $e"
                sensors[sensor_name] = Dict{String, Any}(
                    "prediction_mean" => [0.0],
                    "prediction_std" => [0.0],
                    "sensor_type" => get_sensor_type(sensor_name),
                    "measurement_type" => get_measurement_type(sensor_name),
                    "is_zero_load" => true
                )
            end
        end
    end
    
    return Dict{String, Any}("sensors" => sensors)
end

# 辅助函数：获取传感器类型
function get_sensor_type(sensor_name::String)
    if startswith(sensor_name, "SCADA")
        return :SCADA
    elseif startswith(sensor_name, "AMI")
        return :AMI
    elseif startswith(sensor_name, "PMU")
        return :PMU
    else
        return :Unknown
    end
end

# 辅助函数：获取测量类型
function get_measurement_type(sensor_name::String)
    if contains(sensor_name, "Vmag")
        return :Vmag
    elseif contains(sensor_name, "V_real")
        return :V_real
    elseif contains(sensor_name, "V_imag")
        return :V_imag
    elseif contains(sensor_name, "-P")
        return :P
    elseif contains(sensor_name, "-Q")
        return :Q
    else
        return :Unknown
    end
end

# ������ 更新后的 calculate_linear_interpolation_beta 函数
function calculate_linear_interpolation_beta(FAD, observed_matrix_Z, observed_pairs)
    """
    根据公式计算线性插值的噪声精度β:
    ⟨β⟩ = (FAD) × m × n / ||Z - P_Ω(AB^T)||²_F
    
    未观测到的位置精度设为0
    """
    
    m, n = size(observed_matrix_Z)
    
    # 计算SVD分解
    svd_res = svd(observed_matrix_Z)
    r = min(5, min(m, n))  # 潜在维度
    
    # 构建低秩近似
    U_r = svd_res.U[:, 1:r]
    Σ_r = svd_res.S[1:r]
    Vt_r = svd_res.Vt[1:r, :]
    
    # 重构矩阵
    AB_T = U_r * Diagonal(Σ_r) * Vt_r
    
    # 计算观测位置的投影误差
    projection_error = 0.0
    observed_count = 0
    
    for (i, j) in observed_pairs
        if i <= m && j <= n
            error = observed_matrix_Z[i, j] - AB_T[i, j]
            projection_error += error^2
            observed_count += 1
        end
    end
    
    # 避免除零
    if projection_error < 1e-12
        projection_error = 1e-12
    end
    
    # 计算精度β值
    beta_value = FAD * m * n / projection_error
    
    # ������ 初始化精度矩阵为全零
    beta_matrix = zeros(Float64, m, n)
    
    # ������ 只在观测位置设置精度值
    for (i, j) in observed_pairs
        if i <= m && j <= n
            beta_matrix[i, j] = beta_value
        end
    end
    
    # println("Linear Interpolation β calculation:")
    # println("  - Matrix size: ($m, $n)")
    # println("  - Observed pairs: $observed_count")
    # println("  - Rank: $r")
    # println("  - Projection error: $projection_error")
    # println("  - FAD: $FAD")
    # println("  - β value: $beta_value")
    # println("  - Non-zero β positions: $observed_count")
    # println("  - Zero β positions: $(m*n - observed_count)")
    
    return beta_matrix
end


# ������ 完整的 linear_sbmc 函数
function linear_sbmc(linear_result, FAD, noise_level,lack_a, lack_b, lack_c)
    bmc_results, phase_info= run_stage2_test_corrected_linear(linear_result, FAD, noise_level,lack_a, lack_b, lack_c)
    X_mean = bmc_results.X_mean
    Pij_sol = X_mean[:, 1]
    Qij_sol = X_mean[:, 2]
    Vr_sol = X_mean[:, 3]
    Vi_sol = X_mean[:, 4]
    V_angle = atan.(Vi_sol, Vr_sol)
    V_mag   = X_mean[:, 5]

    # 读取真实值进行比较
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


    miae_power = sum(abs.(origin_value_power - imputed_value_power)) / length(imputed_value_power) * 100
    miae_theta = sum(abs.(origin_value_theta - imputed_value_theta) ) / length(imputed_value_theta)  * 100
    mape_voltage = sum(abs.(origin_value_magnitude - imputed_value_magnitude) ./ origin_value_magnitude) / length(imputed_value_magnitude) * 100
    # println("MIAE Power = $(miae_power) %")
    # println("Linear-SBMC Results:")
    # println("MIAE Power = $(miae_power) %")
    # println("MIAE Theta = $(miae_theta) %")
    # println("MAPE Voltage = $(mape_voltage) %")
    unc_P_mean = mean(bmc_results.uncertainty.P_std)
    unc_Q_mean = mean(bmc_results.uncertainty.Q_std)
    unc_V_mean = mean(bmc_results.uncertainty.V_std)
    
    # 6. 提取计算时间
    # bmc_time = bmc_results.time
    
    return miae_power, miae_theta, mape_voltage
end
#correct version
# function run_stage2_test_corrected_linear(linear_result, FAD, noise_level,lack_a, lack_b, lack_c)
#     # ================= 1. 数据与模型初始化 =================
#     # branch = read_topology_mat("/home/user/Downloads/Distribution_System_State_Estimation-main/pf_out/topology.mat")
    
#     baseMVA = 10.0
#     # 生成数据
#     daily_predictions = generate_linear_daily_predictions(linear_result, 432000, 1)
#     observed_matrix_Z, observed_pairs, monitored_obs, phase_info = build_observed_matrix_Z(daily_predictions, lack_a, lack_b, lack_c)

#     observed_matrix_Z = Float64.(observed_matrix_Z)
    
#     # 使用公式计算线性插值的噪声精度β
#     noise_precision_β = calculate_linear_interpolation_beta(FAD, observed_matrix_Z, observed_pairs)
    
#     noise_precision_β = Float64.(noise_precision_β)

#     w, M, K_mat = get_aligned_physics_matrices(phase_info, baseMVA)
#     abs_w = abs.(w)

#     # 参数设置
#     tolerance = 1e-5
#     c_param = 1e-7
#     d_param = 1e-7
#     max_iter = 3000
    
    
#     # 定义列索引
#     idx_P, idx_Q = 1, 2
#     idx_Vr, idx_Vi, idx_V = 3, 4, 5

#     # ================= 2. SVD 初始化 =================
#     svd_res = svd(observed_matrix_Z)
#     r = adaptive_rank_selection_ln(observed_matrix_Z, noise_level)
#     U_r = svd_res.U[:, 1:r]
#     Σ_r = svd_res.S[1:r]
#     Vt_r = svd_res.Vt[1:r, :]

#     sqrtD = Diagonal(sqrt.(Σ_r))
#     A_mean = Array{Float64}(U_r * sqrtD)
#     B_mean = Array{Float64}(Vt_r' * sqrtD)

#     # 初始化协方差与超参数
#     α = 1e-3
#     Σa0 = α .* Matrix{Float64}(I, r, r)
#     Σb0 = α .* Matrix{Float64}(I, r, r)
#     Σa_list = [copy(Σa0) for _ in 1:size(A_mean, 1)]
#     Σb_list = [copy(Σb0) for _ in 1:size(B_mean, 1)]
#     γ = fill(1.0, r)

#     # X_old 初始值 (SVD 结果)
#     X_old = Array{Float64}(A_mean * B_mean')
#     latent_dim = size(A_mean, 2)

#     history = Dict{Symbol, Vector{Float64}}(:rel_change => Float64[])

#     # ================= 3. 构建反馈闭环结构 (关键修复) =================
#     # 3.1 识别真实观测位置
#     observed_rows_Vr = Set{Int}()
#     observed_rows_Vi = Set{Int}()
#     observed_rows_V  = Set{Int}()

#     for (r, col) in observed_pairs
#         if col == idx_Vr push!(observed_rows_Vr, r)
#         elseif col == idx_Vi push!(observed_rows_Vi, r)
#         elseif col == idx_V  push!(observed_rows_V, r)
#         end
#     end

#     println("Real Observations - Real: $(length(observed_rows_Vr)), Imag: $(length(observed_rows_Vi)), Mag: $(length(observed_rows_V))")

#     # 3.2 动态计算物理约束权重
#     n_total = size(observed_matrix_Z, 1) * size(observed_matrix_Z, 2)
#     n_obs = length(observed_pairs)
#     sparsity_ratio = n_obs / n_total

#     # ✅ 修复：更合理的权重设置
#     if sparsity_ratio < 0.15
#         base_beta = 3000.0  # 降低基础权重
#     elseif sparsity_ratio < 0.25
#         base_beta = 1000.0
#     else
#         base_beta = 300.0
#     end

#     # ✅ 修复：更温和的噪声调整
#     noise_factor = 1.0 / (1.0 + log(1.0 + noise_level * 100))
#     beta_phys = base_beta * noise_factor

#     println("������ Physics constraint settings:")
#     println("  Sparsity: $(round(sparsity_ratio*100, digits=2))%")
#     println("  Base beta: $base_beta")
#     println("  Noise factor: $(round(noise_factor, digits=4))")
#     println("  Final beta_phys: $(round(beta_phys, digits=2))")

#     # 3.3 构建增强观测集
#     augmented_pairs = copy(observed_pairs)
#     augmented_Z = copy(observed_matrix_Z)
#     augmented_beta = copy(noise_precision_β)
#     missing_indices = Vector{Tuple{Int, Int}}()

#     for i in 1:size(observed_matrix_Z, 1)
#         if !(i in observed_rows_Vr)
#             push!(augmented_pairs, (i, idx_Vr))
#             push!(missing_indices, (i, idx_Vr))
#             augmented_beta[i, idx_Vr] = beta_phys
#             augmented_Z[i, idx_Vr] = X_old[i, idx_Vr]
#         end
#         if !(i in observed_rows_Vi)
#             push!(augmented_pairs, (i, idx_Vi))
#             push!(missing_indices, (i, idx_Vi))
#             augmented_beta[i, idx_Vi] = beta_phys
#             augmented_Z[i, idx_Vi] = X_old[i, idx_Vi]
#         end
#         if !(i in observed_rows_V)
#             push!(augmented_pairs, (i, idx_V))
#             push!(missing_indices, (i, idx_V))
#             augmented_beta[i, idx_V] = beta_phys
#             augmented_Z[i, idx_V] = X_old[i, idx_V]
#         end
#     end

#     println("Augmented pairs: $(length(augmented_pairs)) (Original: $(length(observed_pairs)))")
#     # ================= 4. 主迭代循环 =================
#     for it in 1:max_iter
        
#         # --- [Step A] 反馈步骤 (Feedback Step) ---
#         # 将上一轮经过物理约束修正后的 X_old 中的电压值，填入 augmented_Z
#         # 这样 A 和 B 在更新时就能“看到”物理约束的结果
#         if it > 1
#             for (r, c) in missing_indices
#                 augmented_Z[r, c] = X_old[r, c]
#             end
#         end

#         # --- [Step B] 贝叶斯更新 A (使用 augmented 数据) ---
#         for i in 1:size(A_mean, 1)
#             # 注意：传入 augmented_pairs, augmented_beta, augmented_Z
#             βBtB = cal_beta_BTB_i(i, B_mean, Σb_list, augmented_pairs, augmented_beta, latent_dim)
#             Σa_list[i] = cal_sigma_a_i(βBtB, γ)
#             A_mean[i, :] = cal_a_mean_i(i, B_mean, Σa_list[i], augmented_pairs, augmented_beta, augmented_Z)
#         end

#         # --- [Step C] 贝叶斯更新 B (使用 augmented 数据) ---
#         for j in 1:size(B_mean, 1)
#             βAtA = cal_beta_ATA_j(j, A_mean, Σa_list, augmented_pairs, augmented_beta, latent_dim)
#             Σb_list[j] = cal_sigma_b_j(βAtA, γ)
#             B_mean[j, :] = cal_b_mean_j(j, A_mean, Σb_list[j], augmented_pairs, augmented_beta, augmented_Z)
#         end

#         # --- [Step D] 更新超参数 Gamma ---
#         for k in 1:length(γ)
#             aTa = cal_aTa_i(k, A_mean, Σa_list)
#             bTb = cal_bTb_j(k, B_mean, Σb_list)
#             γ[k] = clamp((2 * c_param + size(A_mean,1) + size(B_mean,1)) / (aTa + bTb + 2 * d_param), 1e-6, 1e6)
#         end

#         # --- [Step E] 重构矩阵 ---
#         X_new = Array{Float64}(A_mean * B_mean')

#         # --- [Step F] 应用物理约束 (Physical Projection) ---
#         try
#             P_est = X_new[:, idx_P] ./ (baseMVA*1000)
#             Q_est = X_new[:, idx_Q] ./ (baseMVA*1000)
            
#             if length(P_est) * 2 == size(K_mat, 2)
#                  PQ_vec = vcat(P_est, Q_est)
                 
#                  # 1. 计算物理一致的电压
#                  v_complex_phys = w + M * PQ_vec       # Eq 14
#                  v_mag_phys     = abs_w + K_mat * PQ_vec # Eq 15
                 
#                  # 2. 强制覆盖未观测位置 (Projection)
#                  for i in 1:size(X_new, 1)
#                      # 仅更新那些在 missing_indices 里的位置
#                      # (或者用之前的 Set 判断，效果一样)
                     
#                      if !(i in observed_rows_Vr)
#                          X_new[i, idx_Vr] = real(v_complex_phys[i])
#                      end
                     
#                      if !(i in observed_rows_Vi)
#                          X_new[i, idx_Vi] = imag(v_complex_phys[i])
#                      end
                     
#                      if !(i in observed_rows_V)
#                          X_new[i, idx_V] = v_mag_phys[i]
#                      end
#                  end
#             else
#                 @warn "Dimension mismatch in physical constraint."
#             end
#         catch e
#             println("Physical constraint error: $e")
#         end

#         # --- [Step G] 收敛性检查 ---
#         numerator = norm(X_new - X_old)
#         denominator = max(norm(X_old), 1e-12)
#         rel = numerator / denominator

#         if it % 10 == 0 || it == 1
#             println("Iter $it: rel_change = $rel")
#         end
#         push!(history[:rel_change], rel)
        
#         X_old = copy(X_new) # 更新 X_old 供下一轮反馈使用

#         if rel < tolerance
#             println("Converged at iter=$it, rel=$(rel)")
#             break
#         end
#     end

#     if isempty(history[:rel_change]) || history[:rel_change][end] ≥ tolerance
#         @warn "Not below tolerance yet."
#     end

#     # ================= [新增] 计算概率信息 =================
#     println("Calculating uncertainty quantification...")
#     var_X, std_X = calculate_matrix_uncertainty(A_mean, B_mean, Σa_list, Σb_list)
#     # 提取最终结果
#     Pij_sol = X_old[:, idx_P]
#     Qij_sol = X_old[:, idx_Q]
#     V_sol   = X_old[:, idx_V]

#     # 提取最终结果 (标准差/不确定性)
#     # 这告诉我们估计结果的“可信度”
#     Pij_std = std_X[:, idx_P]
#     Qij_std = std_X[:, idx_Q]
#     V_std   = std_X[:, idx_V]

#     return (
#         X_mean = X_old,          # 估计值的均值矩阵
#         X_std  = std_X,          # 估计值的标准差矩阵 (概率信息)
#         history = history,
#         flows = (P = Pij_sol, Q = Qij_sol, V = V_sol),
#         uncertainty = (P_std = Pij_std, Q_std = Qij_std, V_std = V_std)
#     )
# end

# like mtgp version
# function run_stage2_test_corrected_linear(linear_result, FAD, noise_level,lack_a, lack_b, lack_c)
#     # ================= 1. 数据与模型初始化 =================
#     # branch = read_topology_mat("/home/user/Downloads/Distribution_System_State_Estimation-main/pf_out/topology.mat")
    
#     baseMVA = 10.0
#     # 生成数据
#     daily_predictions = generate_linear_daily_predictions(linear_result, 432000, 1)
#     observed_matrix_Z, observed_pairs, monitored_obs, phase_info = build_observed_matrix_Z(daily_predictions, lack_a, lack_b, lack_c)

#     observed_matrix_Z = Float64.(observed_matrix_Z)
    
#     # 使用公式计算线性插值的噪声精度β
#     noise_precision_β = calculate_linear_interpolation_beta(FAD, observed_matrix_Z, observed_pairs)
    
#     noise_precision_β = Float64.(noise_precision_β)

#     w, M, K_mat = get_aligned_physics_matrices(phase_info, baseMVA)
#     abs_w = abs.(w)

#     # 参数设置
#     tolerance = 1e-5
#     c_param = 1e-7
#     d_param = 1e-7
#     max_iter = 3000
    
    
#     # 定义列索引
#     idx_P, idx_Q = 1, 2
#     idx_Vr, idx_Vi, idx_V = 3, 4, 5

#     # ================= 2. SVD 初始化 =================
#     svd_res = svd(observed_matrix_Z)
#     r = adaptive_rank_selection_ln(observed_matrix_Z, noise_level)
#     U_r = svd_res.U[:, 1:r]
#     Σ_r = svd_res.S[1:r]
#     Vt_r = svd_res.Vt[1:r, :]

#     sqrtD = Diagonal(sqrt.(Σ_r))
#     A_mean = Array{Float64}(U_r * sqrtD)
#     B_mean = Array{Float64}(Vt_r' * sqrtD)

#     # 初始化协方差与超参数
#     α = 1e-3
#     Σa0 = α .* Matrix{Float64}(I, r, r)
#     Σb0 = α .* Matrix{Float64}(I, r, r)
#     Σa_list = [copy(Σa0) for _ in 1:size(A_mean, 1)]
#     Σb_list = [copy(Σb0) for _ in 1:size(B_mean, 1)]
#     γ = fill(1.0, r)

#     # X_old 初始值 (SVD 结果)
#     X_old = Array{Float64}(A_mean * B_mean')
#     latent_dim = size(A_mean, 2)

#     history = Dict{Symbol, Vector{Float64}}(:rel_change => Float64[])

#     # ================= 3. 构建反馈闭环结构 (关键修复) =================
#     # 3.1 识别真实观测位置
#     observed_rows_Vr = Set{Int}()
#     observed_rows_Vi = Set{Int}()
#     observed_rows_V  = Set{Int}()

#     for (r, col) in observed_pairs
#         if col == idx_Vr push!(observed_rows_Vr, r)
#         elseif col == idx_Vi push!(observed_rows_Vi, r)
#         elseif col == idx_V  push!(observed_rows_V, r)
#         end
#     end

#     println("Real Observations - Real: $(length(observed_rows_Vr)), Imag: $(length(observed_rows_Vi)), Mag: $(length(observed_rows_V))")

#     # 3.2 动态计算物理约束权重
#     n_total = size(observed_matrix_Z, 1) * size(observed_matrix_Z, 2)
#     n_obs = length(observed_pairs)
#     sparsity_ratio = n_obs / n_total

#     # ✅ 修复：更合理的权重设置
#     if sparsity_ratio < 0.25
#     # FAD10: 稀疏，必须强依赖物理
#     base_beta = 10000.0  
#     else
#         # FAD30: 密集
#         base_beta = 2500.0    
#     end

#     # ✅ 修复：更温和的噪声调整
#     noise_factor = 1.0 / (sqrt(noise_level) + 0.01)
#     beta_phys = base_beta * noise_factor

#     println("������ Physics constraint settings:")
#     println("  Sparsity: $(round(sparsity_ratio*100, digits=2))%")
#     println("  Base beta: $base_beta")
#     println("  Noise factor: $(round(noise_factor, digits=4))")
#     println("  Final beta_phys: $(round(beta_phys, digits=2))")

#     # 3.3 构建增强观测集
#     augmented_pairs = copy(observed_pairs)
#     augmented_Z = copy(observed_matrix_Z)
#     augmented_beta = copy(noise_precision_β)
#     missing_indices = Vector{Tuple{Int, Int}}()

#     for i in 1:size(observed_matrix_Z, 1)
#         if !(i in observed_rows_Vr)
#             push!(augmented_pairs, (i, idx_Vr))
#             push!(missing_indices, (i, idx_Vr))
#             augmented_beta[i, idx_Vr] = beta_phys
#             augmented_Z[i, idx_Vr] = X_old[i, idx_Vr]
#         end
#         if !(i in observed_rows_Vi)
#             push!(augmented_pairs, (i, idx_Vi))
#             push!(missing_indices, (i, idx_Vi))
#             augmented_beta[i, idx_Vi] = beta_phys
#             augmented_Z[i, idx_Vi] = X_old[i, idx_Vi]
#         end
#         if !(i in observed_rows_V)
#             push!(augmented_pairs, (i, idx_V))
#             push!(missing_indices, (i, idx_V))
#             augmented_beta[i, idx_V] = beta_phys
#             augmented_Z[i, idx_V] = X_old[i, idx_V]
#         end
#     end

#     println("Augmented pairs: $(length(augmented_pairs)) (Original: $(length(observed_pairs)))")
#     # ================= 4. 主迭代循环 =================
#     for it in 1:max_iter
        
#         # --- [Step A] 反馈步骤 (Feedback Step) ---
#         # 将上一轮经过物理约束修正后的 X_old 中的电压值，填入 augmented_Z
#         # 这样 A 和 B 在更新时就能“看到”物理约束的结果
#         if it > 1
#             for (r, c) in missing_indices
#                 augmented_Z[r, c] = X_old[r, c]
#             end
#         end

#         # --- [Step B] 贝叶斯更新 A (使用 augmented 数据) ---
#         for i in 1:size(A_mean, 1)
#             # 注意：传入 augmented_pairs, augmented_beta, augmented_Z
#             βBtB = cal_beta_BTB_i(i, B_mean, Σb_list, augmented_pairs, augmented_beta, latent_dim)
#             Σa_list[i] = cal_sigma_a_i(βBtB, γ)
#             A_mean[i, :] = cal_a_mean_i(i, B_mean, Σa_list[i], augmented_pairs, augmented_beta, augmented_Z)
#         end

#         # --- [Step C] 贝叶斯更新 B (使用 augmented 数据) ---
#         for j in 1:size(B_mean, 1)
#             βAtA = cal_beta_ATA_j(j, A_mean, Σa_list, augmented_pairs, augmented_beta, latent_dim)
#             Σb_list[j] = cal_sigma_b_j(βAtA, γ)
#             B_mean[j, :] = cal_b_mean_j(j, A_mean, Σb_list[j], augmented_pairs, augmented_beta, augmented_Z)
#         end

#         # --- [Step D] 更新超参数 Gamma ---
#         for k in 1:length(γ)
#             aTa = cal_aTa_i(k, A_mean, Σa_list)
#             bTb = cal_bTb_j(k, B_mean, Σb_list)
#             γ[k] = clamp((2 * c_param + size(A_mean,1) + size(B_mean,1)) / (aTa + bTb + 2 * d_param), 1e-6, 1e6)
#         end

#         # --- [Step E] 重构矩阵 ---
#         X_new = Array{Float64}(A_mean * B_mean')

#         # --- [Step F] 应用物理约束 (Physical Projection) ---
#         try
#             P_est = X_new[:, idx_P] ./ (baseMVA*1000)
#             Q_est = X_new[:, idx_Q] ./ (baseMVA*1000)
            
#             if length(P_est) * 2 == size(K_mat, 2)
#                  PQ_vec = vcat(P_est, Q_est)
                 
#                  # 1. 计算物理一致的电压
#                  v_complex_phys = w + M * PQ_vec       # Eq 14
#                  v_mag_phys     = abs_w + K_mat * PQ_vec # Eq 15
                 
#                  # 2. 强制覆盖未观测位置 (Projection)
#                  for i in 1:size(X_new, 1)
#                      # 仅更新那些在 missing_indices 里的位置
#                      # (或者用之前的 Set 判断，效果一样)
                     
#                      if !(i in observed_rows_Vr)
#                          X_new[i, idx_Vr] = real(v_complex_phys[i])
#                      end
                     
#                      if !(i in observed_rows_Vi)
#                          X_new[i, idx_Vi] = imag(v_complex_phys[i])
#                      end
                     
#                      if !(i in observed_rows_V)
#                          X_new[i, idx_V] = v_mag_phys[i]
#                      end
#                  end
#             else
#                 @warn "Dimension mismatch in physical constraint."
#             end
#         catch e
#             println("Physical constraint error: $e")
#         end

#         # --- [Step G] 收敛性检查 ---
#         numerator = norm(X_new - X_old)
#         denominator = max(norm(X_old), 1e-12)
#         rel = numerator / denominator

#         if it % 10 == 0 || it == 1
#             println("Iter $it: rel_change = $rel")
#         end
#         push!(history[:rel_change], rel)
        
#         X_old = copy(X_new) # 更新 X_old 供下一轮反馈使用

#         if rel < tolerance
#             println("Converged at iter=$it, rel=$(rel)")
#             break
#         end
#     end

#     if isempty(history[:rel_change]) || history[:rel_change][end] ≥ tolerance
#         @warn "Not below tolerance yet."
#     end

#     # ================= [新增] 计算概率信息 =================
#     println("Calculating uncertainty quantification...")
#     var_X, std_X = calculate_matrix_uncertainty(A_mean, B_mean, Σa_list, Σb_list)
#     # 提取最终结果
#     Pij_sol = X_old[:, idx_P]
#     Qij_sol = X_old[:, idx_Q]
#     V_sol   = X_old[:, idx_V]

#     # 提取最终结果 (标准差/不确定性)
#     # 这告诉我们估计结果的“可信度”
#     Pij_std = std_X[:, idx_P]
#     Qij_std = std_X[:, idx_Q]
#     V_std   = std_X[:, idx_V]

#     return (
#         X_mean = X_old,          # 估计值的均值矩阵
#         X_std  = std_X,          # 估计值的标准差矩阵 (概率信息)
#         history = history,
#         flows = (P = Pij_sol, Q = Qij_sol, V = V_sol),
#         uncertainty = (P_std = Pij_std, Q_std = Qij_std, V_std = V_std)
#     )
# end

# test version
# function run_stage2_test_corrected_linear(linear_result, FAD, noise_level, lack_a, lack_b, lack_c)
#     # ================= 1. 数据与模型初始化 =================
#     baseMVA = 10.0
    
#     # 1.1 生成 Linear 预测数据
#     daily_predictions = generate_linear_daily_predictions(linear_result, 432000, 1)
#     observed_matrix_Z, observed_pairs, monitored_obs, phase_info = build_observed_matrix_Z(daily_predictions, lack_a, lack_b, lack_c)
    
#     # 1.2 计算 Linear Beta
#     noise_precision_β = calculate_linear_interpolation_beta(FAD, Float64.(observed_matrix_Z), observed_pairs)
#     noise_precision_β = Float64.(noise_precision_β)
#     observed_matrix_Z = Float64.(observed_matrix_Z)

#     # ⚠️ [关键策略 1] 数据权重封顶
#     # 限制 Linear 数据的置信度，给物理修正留出空间
#     replace!(noise_precision_β, Inf => 1000.0, NaN => 1.0)
#     noise_precision_β .= clamp.(noise_precision_β, 0.1, 1000.0)

#     # 1.3 获取物理矩阵
#     w, M, K_mat = get_aligned_physics_matrices(phase_info, baseMVA)
#     abs_w = abs.(w)

#     # =================================================================================
#     # ⚖️ [关键策略 2] 强物理权重 (Strong Physics)
#     # =================================================================================
#     n_total = size(observed_matrix_Z, 1) * size(observed_matrix_Z, 2)
#     n_obs = length(observed_pairs)
#     sparsity_ratio = n_obs / n_total

#     # 无论 FAD 多大，始终保持强物理约束以修正几何错误
#     base_beta = 10000.0  
    
#     if sparsity_ratio > 0.3
#         base_beta = 15000.0
#     end

#     noise_factor = 1.0 / (sqrt(noise_level) + 0.01)
#     PHYSICS_BETA = base_beta * noise_factor
#     PHYSICS_BETA = clamp(PHYSICS_BETA, 5000.0, 90000.0)

#     println("������ [Linear Strategy] Sparsity: $(round(sparsity_ratio*100, digits=1))%")
#     println("   - Physics Beta: $(round(PHYSICS_BETA, digits=1)) (No Damping, High Rank)")

#     # ================= 2. SVD 与 秩选择 =================
#     tolerance = 1e-5
#     c_param = 1e-7
#     d_param = 1e-7
#     max_iter = 5000 
    
#     idx_P, idx_Q = 1, 2
#     idx_Vr, idx_Vi, idx_V = 3, 4, 5

#     svd_res = svd(observed_matrix_Z)
    
#     # ⚠️ [关键策略 3] 提升秩的下限
#     # 秩 4-5 提供足够的自由度来解决 Linear 误差与物理约束的冲突
#     r_rank = adaptive_rank_selection_ln(observed_matrix_Z, noise_level)
#     max_possible_rank = size(observed_matrix_Z, 2)
    
#     r_rank = clamp(r_rank, 4, 5) 
#     r_rank = min(r_rank, max_possible_rank)
    
#     println("������ [Rank] Selected Rank: $r_rank (Fixed 4-5)")

#     U_r = svd_res.U[:, 1:r_rank]
#     Σ_r = svd_res.S[1:r_rank]
#     Vt_r = svd_res.Vt[1:r_rank, :]

#     sqrtD = Diagonal(sqrt.(Σ_r))
#     A_mean = Array{Float64}(U_r * sqrtD)
#     B_mean = Array{Float64}(Vt_r' * sqrtD)

#     α = 1e-3
#     Σa0 = α .* Matrix{Float64}(I, r_rank, r_rank)
#     Σb0 = α .* Matrix{Float64}(I, r_rank, r_rank)
#     Σa_list = [copy(Σa0) for _ in 1:size(A_mean, 1)]
#     Σb_list = [copy(Σb0) for _ in 1:size(B_mean, 1)]
#     γ = fill(1.0, r_rank)

#     X_old = Array{Float64}(A_mean * B_mean')
#     latent_dim = size(A_mean, 2)
#     history = Dict{Symbol, Vector{Float64}}(:rel_change => Float64[])

#     # ================= 3. 构建增强观测集 =================
#     observed_rows_Vr = Set{Int}()
#     observed_rows_Vi = Set{Int}()
#     observed_rows_V  = Set{Int}()
    
#     for (r, col) in observed_pairs
#         if col == idx_Vr push!(observed_rows_Vr, r)
#         elseif col == idx_Vi push!(observed_rows_Vi, r)
#         elseif col == idx_V  push!(observed_rows_V, r)
#         end
#     end

#     augmented_pairs = copy(observed_pairs)
#     augmented_Z = copy(observed_matrix_Z) 
#     augmented_beta = copy(noise_precision_β)
#     missing_indices = Vector{Tuple{Int, Int}}()
    
#     for i in 1:size(observed_matrix_Z, 1)
#         if !(i in observed_rows_Vr)
#             push!(augmented_pairs, (i, idx_Vr))
#             push!(missing_indices, (i, idx_Vr))
#             augmented_beta[i, idx_Vr] = PHYSICS_BETA
#             augmented_Z[i, idx_Vr] = X_old[i, idx_Vr]
#         end
#         if !(i in observed_rows_Vi)
#             push!(augmented_pairs, (i, idx_Vi))
#             push!(missing_indices, (i, idx_Vi))
#             augmented_beta[i, idx_Vi] = PHYSICS_BETA
#             augmented_Z[i, idx_Vi] = X_old[i, idx_Vi]
#         end
#         if !(i in observed_rows_V)
#             push!(augmented_pairs, (i, idx_V))
#             push!(missing_indices, (i, idx_V))
#             augmented_beta[i, idx_V] = PHYSICS_BETA
#             augmented_Z[i, idx_V] = X_old[i, idx_V]
#         end
#     end

#     # ================= 4. 主迭代循环 =================
#     for it in 1:max_iter
        
#         # --- [Step A] 反馈 ---
#         if it > 1
#             for (r, c) in missing_indices
#                 augmented_Z[r, c] = X_old[r, c]
#             end
#         end

#         # --- [Step B & C] 贝叶斯更新 ---
#         for i in 1:size(A_mean, 1)
#             βBtB = cal_beta_BTB_i(i, B_mean, Σb_list, augmented_pairs, augmented_beta, latent_dim)
#             Σa_list[i] = cal_sigma_a_i(βBtB, γ)
#             A_mean[i, :] = cal_a_mean_i(i, B_mean, Σa_list[i], augmented_pairs, augmented_beta, augmented_Z)
#         end

#         for j in 1:size(B_mean, 1)
#             βAtA = cal_beta_ATA_j(j, A_mean, Σa_list, augmented_pairs, augmented_beta, latent_dim)
#             Σb_list[j] = cal_sigma_b_j(βAtA, γ)
#             B_mean[j, :] = cal_b_mean_j(j, A_mean, Σb_list[j], augmented_pairs, augmented_beta, augmented_Z)
#         end

#         # --- [Step D] Gamma 更新 ---
#         for k in 1:length(γ)
#             aTa = cal_aTa_i(k, A_mean, Σa_list)
#             bTb = cal_bTb_j(k, B_mean, Σb_list)
#             γ[k] = clamp((2 * c_param + size(A_mean,1) + size(B_mean,1)) / (aTa + bTb + 2 * d_param), 1e-6, 1e6)
#         end

#         X_temp = Array{Float64}(A_mean * B_mean')

#         # 数值稳定性检查
#         if any(isnan.(X_temp)) || any(isinf.(X_temp))
#             @warn "Numerical instability detected at iter $it. Rolling back."
#             X_temp = copy(X_old)
#         end

#         # --- [Step F] 物理约束投影 (Physics Projection) ---
#         try
#             P_input = X_temp[:, idx_P] ./ (baseMVA * 1000.0)
#             Q_input = X_temp[:, idx_Q] ./ (baseMVA * 1000.0)
            
#             # 简单的数值截断
#             P_input .= clamp.(P_input, -10.0, 10.0)
#             Q_input .= clamp.(Q_input, -10.0, 10.0)
            
#             if length(P_input) * 2 == size(K_mat, 2)
#                  PQ_vec = vcat(P_input, Q_input)
#                  v_complex_phys = w + M * PQ_vec       
#                  v_mag_phys     = abs_w + K_mat * PQ_vec 
                 
#                  if !any(isnan.(v_complex_phys)) && !any(isnan.(v_mag_phys))
#                      for i in 1:size(X_temp, 1)
#                          # 仅覆盖未观测位置
#                          if !(i in observed_rows_Vr)
#                              X_temp[i, idx_Vr] = real(v_complex_phys[i])
#                          end
#                          if !(i in observed_rows_Vi)
#                              X_temp[i, idx_Vi] = imag(v_complex_phys[i])
#                          end
#                          if !(i in observed_rows_V)
#                              X_temp[i, idx_V] = v_mag_phys[i]
#                          end
#                      end
#                  end
#             end
#         catch e
#             # ignore physics error
#         end

#         # --- [Step G] 移除阻尼，直接更新 ---
#         # 既然阻尼影响效率且无助收敛，直接采用激进更新
#         X_new = X_temp

#         # --- [Step H] 收敛检查 ---
#         rel = norm(X_new - X_old) / max(norm(X_old), 1e-12)
#         if it % 50 == 0 
#             println("Iter $it: rel=$rel") 
#         end
#         push!(history[:rel_change], rel)
#         X_old = copy(X_new)
        
#         if rel < tolerance
#             println("✅ Converged at iter=$it, rel=$(rel)")
#             break
#         end
#     end

#     # ================= 5. 不确定性量化 =================
#     println("Calculating uncertainty quantification...")
#     var_X, std_X = calculate_matrix_uncertainty(A_mean, B_mean, Σa_list, Σb_list)
    
#     return (
#         X_mean = X_old,
#         X_std  = std_X,
#         history = history,
#         flows = (P = X_old[:, 1], Q = X_old[:, 2], V = X_old[:, 5]),
#         uncertainty = (P_std = std_X[:, 1], Q_std = std_X[:, 2], V_std = std_X[:, 5])
#     )
# end

function run_stage2_test_corrected_linear(linear_result, FAD, noise_level, lack_a, lack_b, lack_c)
    # ================= 1. 数据准备 =================
    baseMVA = 10.0
    
    daily_predictions = generate_linear_daily_predictions(linear_result, 432000, 1)
    observed_matrix_Z, observed_pairs, monitored_obs, phase_info = build_observed_matrix_Z(daily_predictions, lack_a, lack_b, lack_c, baseMVA)
    noise_precision_β = calculate_linear_interpolation_beta(FAD, Float64.(observed_matrix_Z), observed_pairs)

    observed_matrix_Z = Float64.(observed_matrix_Z)
    noise_precision_β = Float64.(noise_precision_β)

    # ������ 数据权重封顶 - 关键参数
    # data_beta_cap = 300.0
    # replace!(noise_precision_β, Inf => data_beta_cap, NaN => 1.0)
    # noise_precision_β .= clamp.(noise_precision_β, 0.1, data_beta_cap)

    w, M, K_mat = get_aligned_physics_matrices(phase_info, baseMVA)
    abs_w = abs.(w)

    tolerance = 1e-5
    c_param = 1e-7
    d_param = 1e-7
    max_iter = 20000
    
    idx_P, idx_Q = 1, 2
    idx_Vr, idx_Vi, idx_V = 3, 4, 5

    # ================= 2. 初始化 =================
    # 从 MTGP 预测构建初始矩阵
    init_matrix = build_full_matrix_from_predictions(daily_predictions, phase_info)
    
    svd_res = svd(init_matrix)
    r = 4  # Rank 4 平衡去噪和灵活性
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
    println("������ Voltage observations: $n_voltage_obs")

    # ================= 4. 构建增强观测集 =================
    # 🔧 只使用P,Q观测进行Bayesian更新，电压完全由物理模型决定
    augmented_pairs = [(r, c) for (r, c) in observed_pairs if c <= idx_Q]  # 只保留P,Q
    augmented_Z = copy(observed_matrix_Z) 
    augmented_beta = copy(noise_precision_β)
    
    # 物理权重 - 根据噪声动态调整
    beta_phys = clamp(8000.0 / (sqrt(noise_level) + 0.01), 3000.0, 90000.0)
    println("⚖️  Physics Beta: $(round(beta_phys, digits=1)), Mode: Physics-only for voltages")
    println("📌 P,Q observations: $(length(augmented_pairs))")
    
    n_rows = size(observed_matrix_Z, 1)
    # 🔧 如果P,Q观测太少，添加伪观测
    if length(augmented_pairs) < n_rows
        println("⚠️  Adding pseudo-observations for P,Q...")
        existing_pairs = Set(augmented_pairs)
        for i in 1:n_rows
            if !((i, idx_P) in existing_pairs)
                push!(augmented_pairs, (i, idx_P))
                augmented_beta[i, idx_P] = 1.0
            end
            if !((i, idx_Q) in existing_pairs)
                push!(augmented_pairs, (i, idx_Q))
                augmented_beta[i, idx_Q] = 1.0
            end
        end
    end

    # ================= 6. 主迭代循环 =================
    for it in 1:max_iter

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

        # ================= 物理约束：所有电压100%物理 =================
        try
            P_est = X_new[:, idx_P]./(baseMVA*1000.0)
            Q_est = X_new[:, idx_Q]./(baseMVA*1000.0)
            
            if length(P_est) * 2 == size(K_mat, 2)
                PQ_vec = vcat(P_est, Q_est)
                v_complex_phys = w + M * PQ_vec       
                v_mag_phys = abs_w + K_mat * PQ_vec 
                
                # 🔧 所有电压都使用100%物理
                for i in 1:size(X_new, 1)
                    X_new[i, idx_Vr] = real(v_complex_phys[i])
                    X_new[i, idx_Vi] = imag(v_complex_phys[i])
                    X_new[i, idx_V] = v_mag_phys[i]
                end
            end
        catch e
            @warn "Physical constraint error: $e"
        end

        # 收敛检查
        rel = norm(X_new - X_old) / max(norm(X_old), 1e-12)
        if it % 100 == 0 
            V_range = extrema(X_new[:, idx_V])
            println("Iter $it: rel=$(round(rel, sigdigits=4)), V=$(round.(V_range, digits=4))") 
        end
        push!(history[:rel_change], rel)
        X_old = copy(X_new)
        
        if rel < tolerance
            println("✅ Converged at iter=$it")
            break
        end
    end

    var_X, std_X = calculate_matrix_uncertainty(A_mean, B_mean, Σa_list, Σb_list)
    
    return (
        X_mean = X_old,
        X_std  = std_X,
        history = history,
        flows = (P = X_old[:, 1], Q = X_old[:, 2], V = X_old[:, 5]),
        uncertainty = (P_std = std_X[:, 1], Q_std = std_X[:, 2], V_std = std_X[:, 5])
    ), phase_info
end





# 辅助函数
function build_full_matrix_from_predictions(daily_predictions::Dict{String, Any}, phase_info::Dict{String, Any})
    n_rows = phase_info["n_valid_phases"]
    matrix = zeros(Float64, n_rows, 5)
    
    sensors = get(daily_predictions, "sensors", Dict{String, Any}())
    bus_phase_to_row = phase_info["bus_phase_to_row"]
    
    column_map = Dict("p" => 1, "q" => 2, "vreal" => 3, "v_real" => 3,
                      "vimag" => 4, "v_imag" => 4, "vmag" => 5, "v_mag" => 5)
    phase_map = Dict("a" => 1, "b" => 2, "c" => 3)
    
    for (name, data) in sensors
        parts = split(name, '-')
        length(parts) < 4 && continue
        
        node_idx = try parse(Int, parts[2]) catch; continue end
        node_idx == 1 && continue
        
        phase_str = lowercase(parts[3])
        haskey(phase_map, phase_str) || continue
        phase_offset = phase_map[phase_str]
        
        haskey(bus_phase_to_row, (node_idx, phase_offset)) || continue
        row_idx = bus_phase_to_row[(node_idx, phase_offset)]
        
        token = lowercase(replace(join(parts[4:end], ""), r"[\s_()]" => ""))
        haskey(column_map, token) || continue
        col = column_map[token]
        
        means = get(data, "prediction_mean", Float64[])
        !isempty(means) && (matrix[row_idx, col] = means[1])
    end
    
    return matrix
end





function adaptive_rank_selection_ln(observed_matrix_Z::Matrix{Float64}, noise_level::Float64)
    """
    改进的秩选择策略：
    1. 使用能量阈值法（保留 95% 能量）
    2. 使用 Gavish-Donohe 阈值
    3. 考虑噪声水平
    4. 强制最小秩为 3（避免秩过低）
    """
    svd_res = svd(observed_matrix_Z)
    singular_values = svd_res.S
    
    m, n = size(observed_matrix_Z)
    max_rank = min(m, n, 10)
    
    # 方法 1: 能量阈值法（保留 95% 能量）
    total_energy = sum(singular_values.^2)
    cumsum_energy = cumsum(singular_values.^2)
    energy_ratio = cumsum_energy ./ total_energy
    r_energy = findfirst(energy_ratio .>= 0.95)
    r_energy = isnothing(r_energy) ? length(singular_values) : r_energy
    
    # 方法 2: Gavish-Donoho 最优阈值
    β = min(m, n) / max(m, n)
    ω_β = 0.56 * β^3 - 0.95 * β^2 + 1.82 * β + 1.43
    median_sv = median(singular_values)
    τ = ω_β * median_sv
    r_noise = sum(singular_values .> τ)
    
    # 方法 3: 基于噪声的阈值
    if noise_level > 0
        # 噪声阈值：最大奇异值的 (噪声水平 * 5) 倍
        noise_threshold = noise_level * maximum(singular_values) * 5
        r_noise_based = sum(singular_values .> noise_threshold)
    else
        r_noise_based = r_energy
    end
    
    # ✅ 综合选择：取三者的最大值，确保足够的表达能力
    r_final = max(r_energy, r_noise, r_noise_based)
    
    # ✅ 强制最小秩为 3（避免欠拟合）
    r_final = max(r_final, 3)
    
    # 限制最大秩
    r_final = min(r_final, max_rank)
    
    # println("������ Adaptive rank selection (Enhanced):")
    # println("  Energy-based (95%): $r_energy")
    # println("  Gavish-Donoho (τ=$(round(τ, digits=2))): $r_noise")
    # println("  Noise-based: $r_noise_based")
    println("  Final linear rank: $r_final")
    # println("  Top 8 singular values: $(round.(singular_values[1:min(8, length(singular_values))], digits=4))")
    
    return r_final
end
# function adaptive_rank_selection(observed_matrix_Z::Matrix{Float64}, noise_level::Float64)
#     """根据噪声水平自适应选择矩阵的秩"""
    
#     svd_res = svd(observed_matrix_Z)
#     singular_values = svd_res.S
    
#     # 根据噪声水平确定阈值
#     if noise_level == 0.0
#         # 无噪声时，使用更严格的阈值
#         threshold = 1e-10
#     else
#         # 有噪声时，阈值与噪声水平相关
#         threshold = noise_level * maximum(singular_values) * 0.1
#     end
    
#     # 选择大于阈值的奇异值
#     r_adaptive = sum(singular_values .> threshold)
#     r_adaptive = max(1, min(r_adaptive, 10))  # 限制在合理范围内
    
#     println("������ Adaptive rank selection:")
#     println("  Noise level: $(noise_level)")
#     println("  Threshold: $(threshold)")
#     println("  Selected rank: $r_adaptive")
#     println("  Singular values: $(round.(singular_values[1:min(10, length(singular_values))], digits=6))")
    
#     return r_adaptive
# end