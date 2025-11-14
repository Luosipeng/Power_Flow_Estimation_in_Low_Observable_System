function generate_linear_daily_predictions(result::LinearInterpolationResult, day::Int, time_step::Int)
    """
    从线性插值结果中生成指定时刻的预测数据，格式兼容 build_observed_matrix_Z 函数
    
    Args:
        result: LinearInterpolationResult 结构
        day: 天数 (目前未使用，保持接口一致性)
        time_step: 时间步长 (1表示第一时刻)
    
    Returns:
        daily_predictions: Dict，包含 "sensors" 键，格式兼容 build_observed_matrix_Z
    """
    
    # 创建传感器预测结果字典
    sensors = Dict{String, Any}()
    
    # 为每个传感器生成预测
    for (sensor_idx, sensor_name) in enumerate(result.sensor_names)
        # 创建测试时间点 (使用time_step作为输入)
        x_test = Float32[time_step - 1]  # 转换为0-based索引
        
        # 使用线性插值进行预测
        y_pred, σ_pred = linear_predict(result, sensor_idx, x_test)
        
        # 存储预测结果，格式兼容 build_observed_matrix_Z
        sensors[sensor_name] = Dict{String, Any}(
            "prediction_mean" => [y_pred[1]],     # 必须是数组格式
            "prediction_std" => [σ_pred[1]],      # 预测标准差
            "sensor_type" => get_sensor_type(sensor_name),      # 传感器类型
            "measurement_type" => get_measurement_type(sensor_name)  # 测量类型
        )
    end
    
    # 返回兼容格式的字典
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

# 🔥 更新后的 calculate_linear_interpolation_beta 函数
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
    
    # 🔥 初始化精度矩阵为全零
    beta_matrix = zeros(Float64, m, n)
    
    # 🔥 只在观测位置设置精度值
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


# 🔥 完整的 linear_sbmc 函数
function linear_sbmc(linear_result; max_iter, tolerance, c, d, nac, ndc, root_bus, inv_bus, rec_bus, eta, Vref, FAD, noise_level)
    branchAC, branchDC = read_topology_mat("C:/Users/PC/Desktop/paper_case/topology_results.mat")
    
    # 🔥 使用新的线性插值预测函数，格式兼容 build_observed_matrix_Z
    daily_predictions = generate_linear_daily_predictions(linear_result, 180000, 1)
    observed_matrix_Z, observed_pairs, monitored_obs = build_observed_matrix_Z(daily_predictions)
    
    # 使用公式计算线性插值的噪声精度β
    noise_precision_β = calculate_linear_interpolation_beta(FAD, observed_matrix_Z, observed_pairs)
    
    observed_matrix_Z = Array{Float64}(observed_matrix_Z)
    noise_precision_β = Array{Float64}(noise_precision_β)

    # SVD分解
    svd_res = svd(observed_matrix_Z)
    r = adaptive_rank_selection(observed_matrix_Z, noise_level)
    U_r = svd_res.U[:, 1:r]
    Σ_r = svd_res.S[1:r]
    Vt_r = svd_res.Vt[1:r, :]

    sqrtD = Diagonal(sqrt.(Σ_r))
    A_mean = Array{Float64}(U_r * sqrtD)
    B_mean = Array{Float64}(Vt_r' * sqrtD)

    # 初始化参数
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

    # SBMC迭代过程
    for it in 1:max_iter
        # 更新A
        for i in 1:size(A_mean, 1)
            βBtB = cal_beta_BTB_i(i, B_mean, Σb_list, observed_pairs, noise_precision_β, latent_dim)
            Σa_list[i] = cal_sigma_a_i(βBtB, γ)
            A_mean[i, :] = cal_a_mean_i(i, B_mean, Σa_list[i], observed_pairs, noise_precision_β, observed_matrix_Z)
        end
        
        # 更新B
        for j in 1:size(B_mean, 1)
            βAtA = cal_beta_ATA_j(j, A_mean, Σa_list, observed_pairs, noise_precision_β, latent_dim)
            Σb_list[j] = cal_sigma_b_j(βAtA, γ)
            B_mean[j, :] = cal_b_mean_j(j, A_mean, Σb_list[j], observed_pairs, noise_precision_β, observed_matrix_Z)
        end
        
        # 更新γ
        for k in 1:length(γ)
            aTa = cal_aTa_i(k, A_mean, Σa_list)
            bTb = cal_bTb_j(k, B_mean, Σb_list)
            γ[k] = clamp((2c + size(A_mean,1) + size(B_mean,1)) / (aTa + bTb + 2d), 1e-6, 1e6)
        end

        # 物理约束更新
        X_new = Array{Float64}(A_mean * B_mean')
        P_inj = X_new[:, 1]./100
        Q_inj = X_new[:, 2]./100
        Vb = X_new[:, 5]
        Vbr = X_new[:, 3]
        Vbi = X_new[:, 4]

        # 潮流计算
        Vr_ac_sol, Vi_ac_sol, V_ac_sol, Pinj_ac_sol, Qinj_ac_sol, V_dc_sol, Pinj_dc_sol = 
            ac_dc_power_flow(branchAC, branchDC, nac, ndc, P_inj, Q_inj, Vb, Vbr, Vbi, 
                           root_bus, 1, 1.002331602909188, inv_bus, rec_bus, eta, Vref, observed_pairs, true)

        # 更新状态变量
        X_new[:, 5] .= vcat(V_ac_sol, V_dc_sol)
        X_new[:, 1] .= vcat(Pinj_ac_sol.*100, Pinj_dc_sol.*100)
        X_new[:, 2] .= vcat(Qinj_ac_sol.*100, zeros(length(Pinj_dc_sol)).*100)
        X_new[:, 3] .= vcat(Vr_ac_sol, V_dc_sol)
        X_new[:, 4] .= vcat(Vi_ac_sol, zeros(length(V_dc_sol)))

        # 收敛性检查
        numerator = norm(X_new - X_old)
        denominator = max(norm(X_old), 1e-12)
        rel = numerator / denominator

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

    # 计算最终结果
    Vb_mag = X_old[:, 5]
    Vb_angle = atan.(X_old[:, 4], X_old[:, 3]) .* (180 / π)

    # 读取真实值进行比较
    batch_path_1 = "C:/Users/PC/Desktop/paper_case/results_thread_4.mat"
    batch_data_1 = read_batch_mat(batch_path_1)
    origin_value_power_ac = batch_data_1.Pd_out_ac[:,30000]
    origin_value_power_dc = batch_data_1.Pd_out_dc[:,30000]
    origin_value_theta = batch_data_1.Vang_out_ac[:,30000]
    origin_value_magnitude_ac = batch_data_1.Vmag_out_ac[:,30000]
    origin_value_magnitude_dc = batch_data_1.vmag_out_dc[:,30000]

    # 提取插值结果
    imputed_value_power_ac = X_old[1:nac, 1]
    imputed_value_power_dc = X_old[nac+1:end, 1]
    imputed_value_theta = Vb_angle[1:nac]
    imputed_value_magnitude_ac = Vb_mag[1:nac]
    imputed_value_magnitude_dc = Vb_mag[nac+1:end]

    # 计算误差指标
    miae_power = sum(abs.(origin_value_power_ac[2:end] - imputed_value_power_ac[2:end]) ) / (nac -1) * 100
    miae_theta = sum(abs.(origin_value_theta[2:end] - imputed_value_theta[2:end])) / (nac - 1) * 100
    mape_voltage = sum(abs.(origin_value_magnitude_ac[2:end] - imputed_value_magnitude_ac[2:end]) / origin_value_magnitude_ac[2:end]) / (nac - 1) * 100

    # println("Linear-SBMC Results:")
    # println("MIAE Power = $(miae_power) %")
    # println("MIAE Theta = $(miae_theta) %")
    # println("MAPE Voltage = $(mape_voltage) %")
    
    return miae_power, miae_theta, mape_voltage
end
