"""
多目标自动调参模块 - 同时优化电压幅值和电压相角
"""

using Printf
using Dates
using LinearAlgebra

"""
预计算数据结构 - 存储所有不依赖调参的预处理结果（扩展版，包含相角）
"""
struct PrecomputedDataMulti
    daily_predictions::Any
    observed_matrix_Z::Matrix{Float64}
    observed_pairs::Vector{Tuple{Int,Int}}
    monitored_obs::Any
    phase_info::Any
    noise_precision_β_raw::Matrix{Float64}
    init_matrix::Matrix{Float64}
    w::Vector{ComplexF64}
    M::Matrix{ComplexF64}
    K_mat::Matrix{Float64}
    abs_w::Vector{Float64}
    origin_voltage::Vector{Float64}      # 真实电压幅值
    origin_theta::Vector{Float64}        # 真实电压相角（弧度）
    baseMVA::Float64
end


"""
预计算所有数据（包含相角真实值）
"""
function precompute_data_multi(mtgp_result, lack_a, lack_b, lack_c)
    baseMVA = 10.0
    
    daily_predictions = generate_daily_predictions(mtgp_result, 432000, 1)
    observed_matrix_Z, observed_pairs, monitored_obs, phase_info = build_observed_matrix_Z(
        daily_predictions, lack_a, lack_b, lack_c, baseMVA
    )
    noise_precision_β_raw = build_noise_precision_beta(daily_predictions, phase_info)
    
    observed_matrix_Z = Float64.(observed_matrix_Z)
    noise_precision_β_raw = Float64.(noise_precision_β_raw)
    
    w, M, K_mat = get_aligned_physics_matrices(phase_info, baseMVA)
    abs_w = abs.(w)
    
    init_matrix = build_full_matrix_from_predictions(daily_predictions, phase_info)
    
    # 读取真实值
    (voltage_mag_a, voltage_mag_b, voltage_mag_c,
     voltage_ang_a, voltage_ang_b, voltage_ang_c,
     power_p_a, power_p_b, power_p_c,
     power_q_a, power_q_b, power_q_c) = read_all_opendss_data()
    
    # 电压幅值真实值
    origin_voltage = build_ground_truth_vector(
        voltage_mag_a, voltage_mag_b, voltage_mag_c,
        lack_a, lack_b, lack_c
    )
    
    # 电压相角真实值（转换为弧度）
    origin_theta = build_ground_truth_vector(
        voltage_ang_a .* (π/180), 
        voltage_ang_b .* (π/180), 
        voltage_ang_c .* (π/180),
        lack_a, lack_b, lack_c
    )
    
    return PrecomputedDataMulti(
        daily_predictions,
        observed_matrix_Z,
        observed_pairs,
        monitored_obs,
        phase_info,
        noise_precision_β_raw,
        init_matrix,
        w, M, K_mat, abs_w,
        origin_voltage,
        origin_theta,
        baseMVA
    )
end


"""
计算电压幅值MAPE和相角MIAE
"""
function calculate_metrics_fast(X_mean, origin_voltage::Vector{Float64}, origin_theta::Vector{Float64})
    Vr = X_mean[:, 3]
    Vi = X_mean[:, 4]
    V_mag = X_mean[:, 5]
    
    # 计算估计的相角
    estimated_theta = atan.(Vi, Vr)
    
    # MAPE 电压幅值
    mape_voltage = sum(abs.(origin_voltage - V_mag) ./ origin_voltage) / length(V_mag) * 100
    
    # MIAE 相角（百分比形式）
    miae_theta = sum(abs.(origin_theta - estimated_theta)) / length(estimated_theta) * 100
    
    return mape_voltage, miae_theta
end


"""
快速版 BMC（用于多目标调参）
"""
function run_bmc_multi_fast(precomputed::PrecomputedDataMulti;
                            data_beta_cap = 50.0,
                            beta_phys = 5000.0,
                            v_clamp_low = 0.92,
                            v_clamp_high = 1.08,
                            max_iter = 20000,
                            tolerance = 1e-5,
                            verbose = false)
    
    baseMVA = precomputed.baseMVA
    observed_matrix_Z = copy(precomputed.observed_matrix_Z)
    observed_pairs = precomputed.observed_pairs
    phase_info = precomputed.phase_info
    init_matrix = copy(precomputed.init_matrix)
    w = precomputed.w
    M = precomputed.M
    K_mat = precomputed.K_mat
    abs_w = precomputed.abs_w
    
    noise_precision_β = copy(precomputed.noise_precision_β_raw)
    replace!(noise_precision_β, Inf => data_beta_cap, NaN => 1.0)
    noise_precision_β .= clamp.(noise_precision_β, 0.1, data_beta_cap)

    c_param = 1e-7
    d_param = 1e-7
    
    idx_P, idx_Q = 1, 2
    idx_Vr, idx_Vi, idx_V = 3, 4, 5

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

    observed_rows_Vr = Set{Int}()
    observed_rows_Vi = Set{Int}()
    observed_rows_V  = Set{Int}()
    
    for (row, col) in observed_pairs
        if col == idx_Vr push!(observed_rows_Vr, row)
        elseif col == idx_Vi push!(observed_rows_Vi, row)
        elseif col == idx_V  push!(observed_rows_V, row)
        end
    end

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

    for it in 1:max_iter
        
        if it > 1
            for (r_idx, c_idx) in missing_indices
                augmented_Z[r_idx, c_idx] = X_old[r_idx, c_idx]
            end
        end

        for i in 1:size(A_mean, 1)
            βBtB = cal_beta_BTB_i(i, B_mean, Σb_list, augmented_pairs, augmented_beta, latent_dim)
            Σa_list[i] = cal_sigma_a_i(βBtB, γ)
            A_mean[i, :] = cal_a_mean_i(i, B_mean, Σa_list[i], augmented_pairs, augmented_beta, augmented_Z)
        end

        for j in 1:size(B_mean, 1)
            βAtA = cal_beta_ATA_j(j, A_mean, Σa_list, augmented_pairs, augmented_beta, latent_dim)
            Σb_list[j] = cal_sigma_b_j(βAtA, γ)
            B_mean[j, :] = cal_b_mean_j(j, A_mean, Σb_list[j], augmented_pairs, augmented_beta, augmented_Z)
        end

        for k in 1:length(γ)
            aTa = cal_aTa_i(k, A_mean, Σa_list)
            bTb = cal_bTb_j(k, B_mean, Σb_list)
            γ[k] = clamp((2*c_param + size(A_mean,1) + size(B_mean,1)) / (aTa + bTb + 2*d_param), 1e-6, 1e6)
        end

        X_new = Array{Float64}(A_mean * B_mean')

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
        history = history
    ), phase_info
end


"""
    auto_tune_multi(mtgp_result, noise_level, lack_a, lack_b, lack_c; kwargs...)

多目标自动调参函数，同时优化电压幅值和相角。

# 参数
- `weight_voltage`: 电压幅值权重，默认 0.5
- `weight_theta`: 相角权重，默认 0.5
- `target`: 优化目标，可选 :combined（加权和）, :voltage（仅电压）, :theta（仅相角）, :pareto（帕累托前沿）

# 返回
- 最优参数和对应的误差指标
"""
function auto_tune_multi(mtgp_result, noise_level, lack_a, lack_b, lack_c;
                         data_beta_caps = [50.0, 100.0, 200.0, 300.0, 500.0],
                         beta_phys_values = [5000.0, 10000.0, 20000.0, 30000.0, 50000.0],
                         voltage_clamp_ranges = [(0.90, 1.10), (0.92, 1.08), (0.94, 1.06), (0.96, 1.04)],
                         max_iter = 10000,
                         weight_voltage = 0.5,
                         weight_theta = 0.5,
                         target = :combined,  # :combined, :voltage, :theta, :pareto
                         verbose = true)
    
    println("\n" * "="^70)
    println("🔍 开始多目标自动调参...")
    println("   优化目标: $(target)")
    if target == :combined
        println("   权重: 电压=$(weight_voltage), 相角=$(weight_theta)")
    end
    println("="^70)
    
    # 预处理
    println("📦 预处理数据...")
    preprocess_time = @elapsed begin
        precomputed = precompute_data_multi(mtgp_result, lack_a, lack_b, lack_c)
    end
    println("   ✓ 预处理完成，耗时: $(round(preprocess_time, digits=2)) 秒")
    
    total_combinations = length(data_beta_caps) * length(beta_phys_values) * length(voltage_clamp_ranges)
    println("\n📊 参数搜索空间:")
    println("   data_beta_cap: $data_beta_caps")
    println("   beta_phys: $beta_phys_values")
    println("   voltage_clamp: $voltage_clamp_ranges")
    println("   总组合数: $total_combinations")
    println("-"^70)
    
    results = Vector{NamedTuple}()
    best_score = Inf
    best_params = nothing
    best_result = nothing
    
    # 用于 Pareto 前沿
    pareto_front = Vector{NamedTuple}()
    
    start_time = time()
    count = 0
    
    for data_beta_cap in data_beta_caps
        for beta_phys in beta_phys_values
            for (v_low, v_high) in voltage_clamp_ranges
                count += 1
                
                if verbose
                    print("\r[$(count)/$(total_combinations)] β_cap=$data_beta_cap, β_phys=$beta_phys, V∈[$v_low,$v_high]")
                end
                
                try
                    bmc_result, phase_info = run_bmc_multi_fast(
                        precomputed;
                        data_beta_cap = data_beta_cap,
                        beta_phys = beta_phys,
                        v_clamp_low = v_low,
                        v_clamp_high = v_high,
                        max_iter = max_iter,
                        verbose = false
                    )
                    
                    mape_voltage, miae_theta = calculate_metrics_fast(
                        bmc_result.X_mean, 
                        precomputed.origin_voltage, 
                        precomputed.origin_theta
                    )
                    
                    n_iter = length(bmc_result.history[:rel_change])
                    final_rel = n_iter > 0 ? bmc_result.history[:rel_change][end] : Inf
                    converged = (n_iter < max_iter) || (final_rel < 1e-3)
                    
                    # 计算综合得分
                    if target == :combined
                        score = weight_voltage * mape_voltage + weight_theta * miae_theta
                    elseif target == :voltage
                        score = mape_voltage
                    elseif target == :theta
                        score = miae_theta
                    else  # :pareto
                        score = mape_voltage + miae_theta  # 简单和作为参考
                    end
                    
                    result = (
                        data_beta_cap = data_beta_cap,
                        beta_phys = beta_phys,
                        v_clamp = (v_low, v_high),
                        mape_voltage = mape_voltage,
                        miae_theta = miae_theta,
                        score = score,
                        converged = converged,
                        iterations = n_iter
                    )
                    push!(results, result)
                    
                    # 更新 Pareto 前沿
                    if target == :pareto
                        is_dominated = false
                        to_remove = Int[]
                        for (idx, p) in enumerate(pareto_front)
                            if p.mape_voltage <= mape_voltage && p.miae_theta <= miae_theta
                                if p.mape_voltage < mape_voltage || p.miae_theta < miae_theta
                                    is_dominated = true
                                    break
                                end
                            end
                            if mape_voltage <= p.mape_voltage && miae_theta <= p.miae_theta
                                if mape_voltage < p.mape_voltage || miae_theta < p.miae_theta
                                    push!(to_remove, idx)
                                end
                            end
                        end
                        if !is_dominated
                            deleteat!(pareto_front, sort(to_remove, rev=true))
                            push!(pareto_front, result)
                        end
                    end
                    
                    if score < best_score
                        best_score = score
                        best_params = result
                        best_result = bmc_result
                        if verbose
                            status = converged ? "✓" : "~"
                            println(" $status NEW BEST: V=$(round(mape_voltage, digits=4))%, θ=$(round(miae_theta, digits=4))%")
                        end
                    elseif verbose
                        status = converged ? "✓" : "~"
                        println(" $status V=$(round(mape_voltage, digits=4))%, θ=$(round(miae_theta, digits=4))%")
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
    
    # 结果汇总
    println("\n" * "="^70)
    println("📈 调参完成！耗时: $(round(elapsed/60, digits=2)) 分钟")
    println("="^70)
    
    if best_params !== nothing
        status = best_params.converged ? "已收敛" : "未完全收敛"
        println("\n🏆 最优参数组合 (目标: $target):")
        println("   data_beta_cap: $(best_params.data_beta_cap)")
        println("   beta_phys: $(best_params.beta_phys)")
        println("   voltage_clamp: $(best_params.v_clamp)")
        println("   ────────────────────────────")
        println("   📊 MAPE 电压幅值: $(round(best_params.mape_voltage, digits=4))%")
        println("   📐 MIAE 电压相角: $(round(best_params.miae_theta, digits=4))%")
        println("   📈 综合得分: $(round(best_params.score, digits=4))")
        println("   迭代数: $(best_params.iterations) ($status)")
    end
    
    # 打印 Top 5
    if length(results) > 0
        sorted_results = sort(results, by = x -> x.score)
        println("\n📊 Top 5 参数组合 (按综合得分):")
        for (i, r) in enumerate(sorted_results[1:min(5, length(sorted_results))])
            status = r.converged ? "✓" : "✗"
            println("   $i. [$status] β_cap=$(r.data_beta_cap), β_phys=$(r.beta_phys), V∈$(r.v_clamp)")
            println("      → MAPE_V=$(round(r.mape_voltage, digits=4))%, MIAE_θ=$(round(r.miae_theta, digits=4))%")
        end
    end
    
    # 打印 Pareto 前沿
    if target == :pareto && length(pareto_front) > 0
        println("\n🎯 Pareto 前沿 ($(length(pareto_front)) 个非支配解):")
        pareto_sorted = sort(pareto_front, by = x -> x.mape_voltage)
        for (i, r) in enumerate(pareto_sorted)
            println("   $i. β_cap=$(r.data_beta_cap), β_phys=$(r.beta_phys), V∈$(r.v_clamp)")
            println("      → MAPE_V=$(round(r.mape_voltage, digits=4))%, MIAE_θ=$(round(r.miae_theta, digits=4))%")
        end
    end
    
    # 分析最优电压和最优相角的参数差异
    if length(results) > 0
        best_voltage = argmin(r -> r.mape_voltage, results)
        best_theta = argmin(r -> r.miae_theta, results)
        
        println("\n📋 单目标最优对比:")
        println("   最优电压幅值: β_cap=$(best_voltage.data_beta_cap), β_phys=$(best_voltage.beta_phys), V∈$(best_voltage.v_clamp)")
        println("      → MAPE_V=$(round(best_voltage.mape_voltage, digits=4))%, MIAE_θ=$(round(best_voltage.miae_theta, digits=4))%")
        println("   最优电压相角: β_cap=$(best_theta.data_beta_cap), β_phys=$(best_theta.beta_phys), V∈=$(best_theta.v_clamp)")
        println("      → MAPE_V=$(round(best_theta.mape_voltage, digits=4))%, MIAE_θ=$(round(best_theta.miae_theta, digits=4))%")
    end
    
    println("="^70)
    
    if target == :pareto
        return best_params, best_result, results, pareto_front
    else
        return best_params, best_result, results
    end
end


"""
快速调参 - 同时优化电压幅值和相角
"""
function quick_tune_multi(mtgp_result, noise_level, lack_a, lack_b, lack_c; 
                          weight_voltage = 0.5, weight_theta = 0.5)
    return auto_tune_multi(
        mtgp_result, noise_level, lack_a, lack_b, lack_c;
        data_beta_caps = [100.0, 200.0, 300.0, 500.0],
        beta_phys_values = [10000.0, 20000.0, 30000.0, 50000.0],
        voltage_clamp_ranges = [(0.90, 1.10), (0.92, 1.08), (0.94, 1.06)],
        max_iter = 10000,
        weight_voltage = weight_voltage,
        weight_theta = weight_theta,
        target = :combined,
        verbose = true
    )
end


"""
Pareto 前沿搜索 - 找到电压幅值和相角的权衡曲线
"""
function pareto_tune(mtgp_result, noise_level, lack_a, lack_b, lack_c)
    return auto_tune_multi(
        mtgp_result, noise_level, lack_a, lack_b, lack_c;
        data_beta_caps = [50.0, 100.0, 200.0, 300.0, 500.0],
        beta_phys_values = [5000.0, 10000.0, 20000.0, 30000.0, 50000.0],
        voltage_clamp_ranges = [(0.88, 1.12), (0.90, 1.10), (0.92, 1.08), (0.94, 1.06), (0.96, 1.04)],
        max_iter = 10000,
        target = :pareto,
        verbose = true
    )
end


"""
仅优化相角
"""
function tune_theta(mtgp_result, noise_level, lack_a, lack_b, lack_c)
    return auto_tune_multi(
        mtgp_result, noise_level, lack_a, lack_b, lack_c;
        data_beta_caps = [50.0, 100.0, 200.0, 300.0, 500.0],
        beta_phys_values = [5000.0, 10000.0, 20000.0, 30000.0, 50000.0],
        voltage_clamp_ranges = [(0.88, 1.12), (0.90, 1.10), (0.92, 1.08), (0.94, 1.06)],
        max_iter = 10000,
        target = :theta,
        verbose = true
    )
end


"""
打印使用指南
"""
function print_multi_tuning_guide()
    println("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                   多目标 BMC 自动调参使用指南                         ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║                                                                      ║
    ║  1. 快速调参（同时优化电压幅值和相角）:                                 ║
    ║     best, result, all = quick_tune_multi(                            ║
    ║         mtgp_result, noise_level, lack_a, lack_b, lack_c;            ║
    ║         weight_voltage = 0.5,  # 电压权重                             ║
    ║         weight_theta = 0.5     # 相角权重                             ║
    ║     )                                                                ║
    ║                                                                      ║
    ║  2. Pareto 前沿搜索（找到权衡曲线）:                                   ║
    ║     best, result, all, pareto = pareto_tune(                         ║
    ║         mtgp_result, noise_level, lack_a, lack_b, lack_c             ║
    ║     )                                                                ║
    ║                                                                      ║
    ║  3. 仅优化相角:                                                       ║
    ║     best, result, all = tune_theta(                                  ║
    ║         mtgp_result, noise_level, lack_a, lack_b, lack_c             ║
    ║     )                                                                ║
    ║                                                                      ║
    ║  4. 完整自定义搜索:                                                   ║
    ║     best, result, all = auto_tune_multi(                             ║
    ║         mtgp_result, noise_level, lack_a, lack_b, lack_c;            ║
    ║         data_beta_caps = [50, 100, 200, 500],                        ║
    ║         beta_phys_values = [10000, 30000, 50000],                    ║
    ║         voltage_clamp_ranges = [(0.90, 1.10), (0.94, 1.06)],         ║
    ║         weight_voltage = 0.3,  # 更注重相角                           ║
    ║         weight_theta = 0.7,                                          ║
    ║         target = :combined,    # :combined/:voltage/:theta/:pareto   ║
    ║         max_iter = 10000                                             ║
    ║     )                                                                ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
end
