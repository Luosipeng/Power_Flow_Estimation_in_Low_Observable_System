"""
自动调参模块 - 用于寻找最优的 BMC 参数组合
"""

using Printf
using Dates

"""
预计算数据结构 - 存储所有不依赖调参的预处理结果
"""
struct PrecomputedData
    daily_predictions::Any
    observed_matrix_Z::Matrix{Float64}
    observed_pairs::Vector{Tuple{Int,Int}}
    monitored_obs::Any
    phase_info::Any
    noise_precision_β_raw::Matrix{Float64}  # 未裁剪的原始 β
    init_matrix::Matrix{Float64}
    w::Vector{ComplexF64}
    M::Matrix{ComplexF64}
    K_mat::Matrix{Float64}
    abs_w::Vector{Float64}
    origin_voltage::Vector{Float64}  # 真实电压值（用于计算MAPE）
    baseMVA::Float64
end


"""
预计算所有不依赖调参的数据（只执行一次）
"""
function precompute_data(mtgp_result, lack_a, lack_b, lack_c)
    baseMVA = 10.0
    
    # 这些只需要计算一次
    daily_predictions = generate_daily_predictions(mtgp_result, 432000, 1)
    observed_matrix_Z, observed_pairs, monitored_obs, phase_info = build_observed_matrix_Z(
        daily_predictions, lack_a, lack_b, lack_c, baseMVA
    )
    noise_precision_β_raw = build_noise_precision_beta(daily_predictions, phase_info)
    
    observed_matrix_Z = Float64.(observed_matrix_Z)
    noise_precision_β_raw = Float64.(noise_precision_β_raw)
    
    # 物理矩阵
    w, M, K_mat = get_aligned_physics_matrices(phase_info, baseMVA)
    abs_w = abs.(w)
    
    # 初始化矩阵
    init_matrix = build_full_matrix_from_predictions(daily_predictions, phase_info)
    
    # 真实电压值（用于计算MAPE）
    (voltage_mag_a, voltage_mag_b, voltage_mag_c,
     voltage_ang_a, voltage_ang_b, voltage_ang_c,
     power_p_a, power_p_b, power_p_c,
     power_q_a, power_q_b, power_q_c) = read_all_opendss_data()
    
    origin_voltage = build_ground_truth_vector(
        voltage_mag_a, voltage_mag_b, voltage_mag_c,
        lack_a, lack_b, lack_c
    )
    
    return PrecomputedData(
        daily_predictions,
        observed_matrix_Z,
        observed_pairs,
        monitored_obs,
        phase_info,
        noise_precision_β_raw,
        init_matrix,
        w, M, K_mat, abs_w,
        origin_voltage,
        baseMVA
    )
end

"""
    auto_tune_bmc(mtgp_result, noise_level, lack_a, lack_b, lack_c; kwargs...)

自动调参函数，通过网格搜索寻找最优参数组合。

# 参数
- `mtgp_result`: GP预测结果
- `noise_level`: 噪声水平
- `lack_a, lack_b, lack_c`: 缺失相位信息

# 可选参数
- `data_beta_caps`: data_beta_cap 的候选值列表，默认 [25, 50, 100, 200]
- `beta_phys_multipliers`: beta_phys 乘数候选值，默认 [0.5, 1.0, 2.0, 3.0]
- `voltage_clamp_ranges`: 电压裁剪范围候选值，默认 [(0.90, 1.10), (0.92, 1.08), (0.94, 1.06)]
- `max_iter`: 每次调参的最大迭代次数，默认 5000（加速搜索）
- `verbose`: 是否打印详细信息，默认 true

# 返回
- 最优参数字典和对应的 MAPE
"""
function auto_tune_bmc(mtgp_result, noise_level, lack_a, lack_b, lack_c;
                       data_beta_caps = [25.0, 50.0, 100.0, 200.0],
                       beta_phys_values = [500.0, 1000.0, 2000.0, 5000.0, 10000.0],
                       voltage_clamp_ranges = [(0.90, 1.10), (0.92, 1.08), (0.94, 1.06)],
                       max_iter = 5000,
                       verbose = true)
    
    println("\n" * "="^70)
    println("🔍 开始自动调参...")
    println("="^70)
    
    # ================= 预处理：只执行一次 =================
    println("📦 预处理数据（仅执行一次）...")
    preprocess_time = @elapsed begin
        precomputed = precompute_data(mtgp_result, lack_a, lack_b, lack_c)
    end
    println("   ✓ 预处理完成，耗时: $(round(preprocess_time, digits=2)) 秒")
    
    # 计算总搜索空间
    total_combinations = length(data_beta_caps) * length(beta_phys_values) * length(voltage_clamp_ranges)
    println("\n📊 参数搜索空间:")
    println("   data_beta_cap: $data_beta_caps")
    println("   beta_phys: $beta_phys_values")
    println("   voltage_clamp: $voltage_clamp_ranges")
    println("   总组合数: $total_combinations")
    println("-"^70)
    
    # 记录结果
    results = Vector{NamedTuple}()
    best_mape = Inf
    best_params = nothing
    best_result = nothing
    
    start_time = time()
    count = 0
    
    for data_beta_cap in data_beta_caps
        for beta_phys in beta_phys_values
            for (v_low, v_high) in voltage_clamp_ranges
                count += 1
                
                if verbose
                    print("\r[$(count)/$(total_combinations)] Testing: β_cap=$data_beta_cap, β_phys=$beta_phys, V∈[$v_low,$v_high]")
                end
                
                try
                    # 运行带参数的 BMC（使用预计算数据）
                    bmc_result, phase_info = run_stage2_test_with_params_fast(
                        precomputed;
                        data_beta_cap = data_beta_cap,
                        beta_phys = beta_phys,
                        v_clamp_low = v_low,
                        v_clamp_high = v_high,
                        max_iter = max_iter,
                        verbose = false
                    )
                    
                    # 计算 MAPE
                    mape = calculate_voltage_mape_fast(bmc_result.X_mean, precomputed.origin_voltage)
                    
                    # 检查收敛状态（放宽条件）
                    n_iter = length(bmc_result.history[:rel_change])
                    final_rel = n_iter > 0 ? bmc_result.history[:rel_change][end] : Inf
                    # 收敛条件：迭代未达上限，或者最终相对变化足够小
                    converged = (n_iter < max_iter) || (final_rel < 1e-3)
                    
                    # 记录结果
                    result = (
                        data_beta_cap = data_beta_cap,
                        beta_phys = beta_phys,
                        v_clamp = (v_low, v_high),
                        mape = mape,
                        converged = converged,
                        iterations = length(bmc_result.history[:rel_change])
                    )
                    push!(results, result)
                    
                    # 更新最优（即使未完全收敛也考虑，只要MAPE更好）
                    if mape < best_mape
                        best_mape = mape
                        best_params = result
                        best_result = bmc_result
                        if verbose
                            status = converged ? "✓" : "~"
                            println(" $status New best: MAPE=$(round(mape, digits=4))%")
                        end
                    elseif verbose
                        status = converged ? "✓" : "~"
                        println(" $status MAPE=$(round(mape, digits=4))%")
                    end
                    
                catch e
                    if verbose
                        println(" ✗ Error: $e")
                    end
                end
            end
        end
    end
    
    elapsed = time() - start_time
    
    # 打印结果汇总
    println("\n" * "="^70)
    println("📈 调参完成！耗时: $(round(elapsed/60, digits=2)) 分钟")
    println("="^70)
    
    if best_params !== nothing
        status = best_params.converged ? "已收敛" : "未完全收敛"
        println("\n🏆 最优参数组合:")
        println("   data_beta_cap: $(best_params.data_beta_cap)")
        println("   beta_phys: $(best_params.beta_phys)")
        println("   voltage_clamp: $(best_params.v_clamp)")
        println("   MAPE: $(round(best_params.mape, digits=4))%")
        println("   迭代数: $(best_params.iterations) ($status)")
    else
        println("\n⚠️  未找到有效参数组合")
    end
    
    # 打印前5名结果
    if length(results) > 0
        sorted_results = sort(results, by = x -> x.mape)
        println("\n📊 Top 5 参数组合:")
        for (i, r) in enumerate(sorted_results[1:min(5, length(sorted_results))])
            status = r.converged ? "✓" : "✗"
            println("   $i. [$status] β_cap=$(r.data_beta_cap), β_phys=$(r.beta_phys), " *
                    "V∈$(r.v_clamp) → MAPE=$(round(r.mape, digits=4))%")
        end
    end
    
    println("="^70)
    
    return best_params, best_result, results
end


"""
带参数的 BMC 运行函数（用于调参）
"""
function run_stage2_test_with_params(mtgp_result, noise_level, lack_a, lack_b, lack_c;
                                     data_beta_cap = 50.0,
                                     beta_phys = 5000.0,
                                     v_clamp_low = 0.92,
                                     v_clamp_high = 1.08,
                                     max_iter = 20000,
                                     tolerance = 1e-5,
                                     verbose = true)
    # 先预计算
    precomputed = precompute_data(mtgp_result, lack_a, lack_b, lack_c)
    
    # 使用快速版本
    return run_stage2_test_with_params_fast(
        precomputed;
        data_beta_cap = data_beta_cap,
        beta_phys = beta_phys,
        v_clamp_low = v_clamp_low,
        v_clamp_high = v_clamp_high,
        max_iter = max_iter,
        tolerance = tolerance,
        verbose = verbose
    )
end


"""
快速版 BMC 运行函数（使用预计算数据，用于调参循环）
"""
function run_stage2_test_with_params_fast(precomputed::PrecomputedData;
                                          data_beta_cap = 50.0,
                                          beta_phys = 5000.0,
                                          v_clamp_low = 0.92,
                                          v_clamp_high = 1.08,
                                          max_iter = 20000,
                                          tolerance = 1e-5,
                                          verbose = true)
    # ================= 1. 从预计算数据中提取 =================
    baseMVA = precomputed.baseMVA
    observed_matrix_Z = copy(precomputed.observed_matrix_Z)
    observed_pairs = precomputed.observed_pairs
    phase_info = precomputed.phase_info
    init_matrix = copy(precomputed.init_matrix)
    w = precomputed.w
    M = precomputed.M
    K_mat = precomputed.K_mat
    abs_w = precomputed.abs_w
    
    # 复制并裁剪 noise_precision_β（这是依赖 data_beta_cap 的）
    noise_precision_β = copy(precomputed.noise_precision_β_raw)
    replace!(noise_precision_β, Inf => data_beta_cap, NaN => 1.0)
    noise_precision_β .= clamp.(noise_precision_β, 0.1, data_beta_cap)

    c_param = 1e-7
    d_param = 1e-7
    
    idx_P, idx_Q = 1, 2
    idx_Vr, idx_Vi, idx_V = 3, 4, 5

    # ================= 2. 初始化 =================
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
    
    for (row, col) in observed_pairs
        if col == idx_Vr push!(observed_rows_Vr, row)
        elseif col == idx_Vi push!(observed_rows_Vi, row)
        elseif col == idx_V  push!(observed_rows_V, row)
        end
    end

    # ================= 4. 构建增强观测集 =================
    augmented_pairs = copy(observed_pairs)
    augmented_Z = copy(observed_matrix_Z) 
    augmented_beta = copy(noise_precision_β)
    missing_indices = Vector{Tuple{Int, Int}}()

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

    # ================= 5. 主迭代循环 =================
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

        # ================= 物理约束 =================
        try
            P_est = X_new[:, idx_P]./(baseMVA*1000.0)
            Q_est = X_new[:, idx_Q]./(baseMVA*1000.0)
            
            if length(P_est) * 2 == size(K_mat, 2)
                PQ_vec = vcat(P_est, Q_est)
                v_complex_phys = w + M * PQ_vec       
                v_mag_phys = abs_w + K_mat * PQ_vec 
                
                v_mag_phys = clamp.(v_mag_phys, v_clamp_low, v_clamp_high)
                
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
        end
        
        X_new[:, idx_V] .= clamp.(X_new[:, idx_V], v_clamp_low, v_clamp_high)

        rel = norm(X_new - X_old) / max(norm(X_old), 1e-12)
        push!(history[:rel_change], rel)
        X_old = copy(X_new)
        
        if rel < tolerance
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


"""
计算电压 MAPE（用于调参评估）
"""
function calculate_voltage_mape(X_mean, lack_a, lack_b, lack_c)
    # 读取真实值
    (voltage_mag_a, voltage_mag_b, voltage_mag_c,
     voltage_ang_a, voltage_ang_b, voltage_ang_c,
     power_p_a, power_p_b, power_p_c,
     power_q_a, power_q_b, power_q_c) = read_all_opendss_data()
    
    origin_value_magnitude = build_ground_truth_vector(
        voltage_mag_a, voltage_mag_b, voltage_mag_c,
        lack_a, lack_b, lack_c
    )
    
    V_mag = X_mean[:, 5]
    mape_voltage = sum(abs.(origin_value_magnitude - V_mag) ./ origin_value_magnitude) / length(V_mag) * 100
    
    return mape_voltage
end


"""
快速计算电压 MAPE（使用预计算的真实电压值）
"""
function calculate_voltage_mape_fast(X_mean, origin_voltage::Vector{Float64})
    V_mag = X_mean[:, 5]
    mape_voltage = sum(abs.(origin_voltage - V_mag) ./ origin_voltage) / length(V_mag) * 100
    return mape_voltage
end


"""
    quick_tune(mtgp_result, noise_level, lack_a, lack_b, lack_c)

快速调参 - 使用较少的参数组合快速找到合理的参数
"""
function quick_tune(mtgp_result, noise_level, lack_a, lack_b, lack_c)
    return auto_tune_bmc(
        mtgp_result, noise_level, lack_a, lack_b, lack_c;
        data_beta_caps = [150.0, 200.0, 250.0],
        beta_phys_values = [25000.0, 30000.0, 40000.0, 50000.0],
        voltage_clamp_ranges = [(0.95, 1.05), (0.96, 1.04)],
        max_iter = 10000,
        verbose = true
    )
end


"""
    fine_tune(mtgp_result, noise_level, lack_a, lack_b, lack_c, base_params)

精细调参 - 在已知较好参数附近进行精细搜索
"""
function fine_tune(mtgp_result, noise_level, lack_a, lack_b, lack_c, base_params)
    β_cap = base_params.data_beta_cap
    β_phys = base_params.beta_phys
    v_low, v_high = base_params.v_clamp
    
    # 在基础参数附近生成精细搜索空间
    data_beta_caps = [β_cap * 0.7, β_cap, β_cap * 1.3, β_cap * 1.5]
    beta_phys_values = [β_phys * 0.5, β_phys * 0.75, β_phys, β_phys * 1.25, β_phys * 1.5]
    
    # 电压范围微调
    margin = (v_high - v_low) / 2
    voltage_clamp_ranges = [
        (v_low - 0.01, v_high + 0.01),
        (v_low, v_high),
        (v_low + 0.01, v_high - 0.01)
    ]
    
    return auto_tune_bmc(
        mtgp_result, noise_level, lack_a, lack_b, lack_c;
        data_beta_caps = data_beta_caps,
        beta_phys_values = beta_phys_values,
        voltage_clamp_ranges = voltage_clamp_ranges,
        max_iter = 10000,
        verbose = true
    )
end


"""
打印调参指南
"""
function print_tuning_guide()
    println("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                        BMC 自动调参使用指南                           ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║                                                                      ║
    ║  1. 快速调参（推荐首次使用）:                                          ║
    ║     best_params, best_result, all_results = quick_tune(              ║
    ║         mtgp_result, noise_level, lack_a, lack_b, lack_c             ║
    ║     )                                                                ║
    ║                                                                      ║
    ║  2. 完整网格搜索:                                                     ║
    ║     best_params, best_result, all_results = auto_tune_bmc(           ║
    ║         mtgp_result, noise_level, lack_a, lack_b, lack_c;            ║
    ║         data_beta_caps = [25, 50, 100, 200],                         ║
    ║         beta_phys_values = [500, 1000, 2000, 5000],                  ║
    ║         voltage_clamp_ranges = [(0.92, 1.08), (0.94, 1.06)],         ║
    ║         max_iter = 5000                                              ║
    ║     )                                                                ║
    ║                                                                      ║
    ║  3. 精细调参（在已知较好参数附近搜索）:                                  ║
    ║     final_params, final_result, _ = fine_tune(                       ║
    ║         mtgp_result, noise_level, lack_a, lack_b, lack_c,            ║
    ║         best_params                                                  ║
    ║     )                                                                ║
    ║                                                                      ║
    ║  4. 使用最优参数运行完整 BMC:                                          ║
    ║     result, phase_info = run_stage2_test_with_params(                ║
    ║         mtgp_result, noise_level, lack_a, lack_b, lack_c;            ║
    ║         data_beta_cap = best_params.data_beta_cap,                   ║
    ║         beta_phys = best_params.beta_phys,                           ║
    ║         v_clamp_low = best_params.v_clamp[1],                        ║
    ║         v_clamp_high = best_params.v_clamp[2],                       ║
    ║         max_iter = 20000,                                            ║
    ║         verbose = true                                               ║
    ║     )                                                                ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
end
