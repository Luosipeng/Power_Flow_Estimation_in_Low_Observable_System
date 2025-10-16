function robust_cholesky(K::Matrix{Float32}; max_tries::Int=5, initial_jitter::Float32=1.0f-6)
    jitter = initial_jitter
    
    for i in 1:max_tries
        try
            K_noisy = K + jitter * I
            K_noisy = Hermitian(0.5f0 .* (K_noisy .+ K_noisy'))
            L = cholesky(K_noisy).L
            return L, true, jitter
        catch e
            if i < max_tries
                jitter *= 10.0f0
            else
                return nothing, false, jitter
            end
        end
    end
end

function multitask_gp_predict(result, sensor_idx::Int, x_test::Vector{Float32})
    data = result.data
    norm_params = result.norm_params
    
    x_train = (data.times[sensor_idx] .- norm_params.x_mean) ./ norm_params.x_std
    y_train = (data.values[sensor_idx] .- norm_params.y_means[sensor_idx]) ./ 
              norm_params.y_stds[sensor_idx]
    x_test_norm = (x_test .- norm_params.x_mean) ./ norm_params.x_std
    
    σ_g_norm = result.σ_g / norm_params.y_stds[sensor_idx]
    ℓ_g_norm = result.ℓ_g / norm_params.x_std
    σ_s_norm = result.σ_s ./ norm_params.y_stds
    ℓ_s_norm = result.ℓ_s ./ norm_params.x_std
    σ_noise_norm = result.σ_noise[sensor_idx] / norm_params.y_stds[sensor_idx]
    
    m_train = vec(result.mean_func(reshape(x_train, 1, :)))
    m_test = vec(result.mean_func(reshape(x_test_norm, 1, :)))
    
    K_xx = compute_multitask_kernel(x_train, x_train, sensor_idx, sensor_idx,
                                     σ_g_norm, ℓ_g_norm, σ_s_norm, ℓ_s_norm)
    
    K_xx_base = K_xx + σ_noise_norm^2 * I
    L, success, used_jitter = robust_cholesky(K_xx_base, initial_jitter=result.jitter)
    
    if !success
        @warn "Cholesky failed during prediction for sensor $(data.sensor_names[sensor_idx])"
        μ_star = m_test .* norm_params.y_stds[sensor_idx] .+ norm_params.y_means[sensor_idx]
        σ_star = ones(Float32, length(m_test)) .* norm_params.y_stds[sensor_idx]
        return μ_star, σ_star
    end
    
    K_x_star = compute_multitask_kernel(x_train, x_test_norm, sensor_idx, sensor_idx,
                                         σ_g_norm, ℓ_g_norm, σ_s_norm, ℓ_s_norm)
    K_star_star = compute_multitask_kernel(x_test_norm, x_test_norm, 
                                            sensor_idx, sensor_idx,
                                            σ_g_norm, ℓ_g_norm, σ_s_norm, ℓ_s_norm)
    
    residual = y_train .- m_train
    α = L' \ (L \ residual)
    μ_star_norm = m_test .+ K_x_star' * α
    
    v = L \ K_x_star
    cov_star = K_star_star - v' * v
    σ_star_norm = sqrt.(max.(diag(cov_star), 1.0f-8))
    
    μ_star = μ_star_norm .* norm_params.y_stds[sensor_idx] .+ norm_params.y_means[sensor_idx]
    σ_star = σ_star_norm .* norm_params.y_stds[sensor_idx]
    
    return μ_star, σ_star
end