function mtgpbmc(mtgp_result; max_iter, tolerance, c, d, nac, ndc, root_bus, inv_bus, rec_bus, eta, Vref, noise_level)
    branchAC, branchDC = read_topology_mat("C:/Users/PC/Desktop/paper_case/topology_results.mat")
    daily_predictions = generate_daily_predictions(mtgp_result, 180000, 1)
    observed_matrix_Z, observed_pairs, monitored_obs = build_observed_matrix_Z(daily_predictions)
    noise_precision_β = build_noise_precision_beta(daily_predictions)

    observed_matrix_Z = Array{Float64}(observed_matrix_Z)
    noise_precision_β = Array{Float64}(noise_precision_β)

    svd_res = svd(observed_matrix_Z)
    r = adaptive_rank_selection(observed_matrix_Z, noise_level)
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

        X_new = Array{Float64}(A_mean * B_mean')
        P_inj = X_new[:, 1]./100; Q_inj = X_new[:, 2]./100; Vb = X_new[:, 5]; 
        Vbr = X_new[:, 3]; Vbi = X_new[:, 4];

        Vr_ac_sol, Vi_ac_sol, V_ac_sol, Pinj_ac_sol, Qinj_ac_sol, V_dc_sol, Pinj_dc_sol = ac_dc_power_flow(branchAC,branchDC, nac, ndc, P_inj, Q_inj, Vb, Vbr, Vbi, root_bus, 1, 1.002331602909188, inv_bus, rec_bus, eta, Vref, observed_pairs, true)

        X_new[:, 5] .= vcat(V_ac_sol, V_dc_sol)
        X_new[:, 1] .= vcat(Pinj_ac_sol.*100, Pinj_dc_sol.*100)
        X_new[:, 2] .= vcat(Qinj_ac_sol.*100, zeros(length(Pinj_dc_sol)).*100)
        X_new[:, 3] .= vcat(Vr_ac_sol, V_dc_sol)
        X_new[:, 4] .= vcat(Vi_ac_sol, zeros(length(V_dc_sol)))

        numerator = norm(X_new - X_old)
        denominator = max(norm(X_old), 1e-12)
        rel = numerator / denominator

        println("Iter $it: rel_change = $rel")
        push!(history[:rel_change], rel)
        X_old = X_new

        if rel < tolerance
            # println("Converged at iter=$it, rel=$(rel)")
            break
        end
    end
    if isempty(history[:rel_change]) || history[:rel_change][end] ≥ tolerance
        @warn "Not below tolerance yet. tail(rel)=$(history[:rel_change][max(end-4,1):end])"
    end
    Vb_mag = X_old[:, 5]
    Vb_angle = atan.(X_old[:, 4], X_old[:, 3]) .* (180 / π)

    batch_path_1 = "C:/Users/PC/Desktop/paper_case/results_thread_4.mat"
    batch_data_1 = read_batch_mat(batch_path_1)
    origin_value_power_ac = batch_data_1.Pd_out_ac[:,30000]
    origin_value_power_dc = batch_data_1.Pd_out_dc[:,30000]
    origin_value_theta = batch_data_1.Vang_out_ac[:,30000]
    origin_value_magnitude_ac = batch_data_1.Vmag_out_ac[:,30000]
    origin_value_magnitude_dc = batch_data_1.vmag_out_dc[:,30000]

    imputed_value_power_ac = X_old[1:nac, 1]
    imputed_value_power_dc = X_old[nac+1:end, 1]
    imputed_value_theta = Vb_angle[1:nac]
    imputed_value_magnitude_ac = Vb_mag[1:nac]
    imputed_value_magnitude_dc = Vb_mag[nac+1:end]

    miae_power = sum(abs.(origin_value_power_ac[2:end] - imputed_value_power_ac[2:end]) ) / (nac -1) * 100
    miae_theta = sum(abs.(origin_value_theta[2:end] - imputed_value_theta[2:end])) / (nac - 1) * 100
    mape_voltage = sum(abs.(origin_value_magnitude_ac[2:end] - imputed_value_magnitude_ac[2:end]) / origin_value_magnitude_ac[2:end]) / (nac - 1) * 100
    # println("MIAE Power = $(miae_power) %")
    # println("MIAE Theta = $(miae_theta) %")
    # println("MAPE Voltage = $(mape_voltage) %")
    return miae_power, miae_theta, mape_voltage
end

function adaptive_rank_selection(observed_matrix_Z::Matrix{Float64}, noise_level::Float64)
    """根据噪声水平自适应选择矩阵的秩"""
    
    svd_res = svd(observed_matrix_Z)
    singular_values = svd_res.S
    
    # 根据噪声水平确定阈值
    if noise_level == 0.0
        # 无噪声时，使用更严格的阈值
        threshold = 1e-10
    else
        # 有噪声时，阈值与噪声水平相关
        threshold = noise_level * maximum(singular_values) * 0.1
    end
    
    # 选择大于阈值的奇异值
    r_adaptive = sum(singular_values .> threshold)
    r_adaptive = max(1, min(r_adaptive, 10))  # 限制在合理范围内
    
    println("🎯 Adaptive rank selection:")
    println("  Noise level: $(noise_level)")
    println("  Threshold: $(threshold)")
    println("  Selected rank: $r_adaptive")
    println("  Singular values: $(round.(singular_values[1:min(10, length(singular_values))], digits=6))")
    
    return r_adaptive
end