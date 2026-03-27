# Predict jointly for a given task s
function icm_predict(result, s::Int, x_test::Vector{Float32})
    """
    参数：
    - s: 活跃传感器索引（1-125），用于访问模型参数和归一化数据
    """
    S_active = length(result.active_indices)  # ✅ 使用活跃传感器数（125）
    normp = result.norm_params

    x_test_norm = (x_test .- normp.x_mean) ./ normp.x_std

    joint_pack = result.joint_pack
    x_all = joint_pack.x_all
    y_all = joint_pack.y_all

    cache = hasproperty(result, :posterior_cache) ? result.posterior_cache : build_icm_posterior_cache(
        result.mean_func,
        result.L_params,
        result.log_σ_time,
        result.log_ℓ_time,
        result.log_σ_locals,
        result.log_ℓ_locals,
        result.log_σ_noise,
        joint_pack;
        jitter=1f-5
    )

    B = cache.B
    σ_time = cache.σ_time
    ℓ_time = cache.ℓ_time
    σ_locals = cache.σ_locals
    ℓ_locals = cache.ℓ_locals
    L = cache.L

    # K_x_star: N x n*
    n_star = length(x_test_norm)
    K_x_star = zeros(Float32, length(x_all), n_star)

    offsets = joint_pack.offsets
    n_per_task = joint_pack.n_per_task

    # ✅ 修改：遍历活跃传感器（1-125）而不是全局传感器（1-186）
    for i in 1:S_active
        xi = joint_pack.norm_data.times[i]
        K_time_is = rbf_kernel(xi, x_test_norm, σ_time, ℓ_time)
        K_local = (i == s) ? rbf_kernel(xi, x_test_norm, σ_locals[s], ℓ_locals[s]) :
                             zeros(Float32, size(K_time_is))
        block = B[i,s] .* K_time_is .+ K_local
        os = offsets[i]; ns = n_per_task[i]
        @views K_x_star[os+1:os+ns, :] .= block
    end

    α = cache.α

    m_star = vec(result.mean_func(reshape(x_test_norm, 1, :)))
    μ_star_norm = m_star .+ K_x_star' * α

    v = L \ K_x_star
    Kss = B[s,s] .* rbf_kernel(x_test_norm, x_test_norm, σ_time, ℓ_time) .+
          rbf_kernel(x_test_norm, x_test_norm, σ_locals[s], ℓ_locals[s])
    cov_star = Kss .- v' * v
    σ_star_norm = sqrt.(max.(diag(cov_star), 1f-8))

    # ✅ 使用活跃索引 s 访问归一化参数
    μ_star = μ_star_norm .* joint_pack.norm_params.y_stds[s] .+ joint_pack.norm_params.y_means[s]
    σ_star = σ_star_norm .* joint_pack.norm_params.y_stds[s]
    return μ_star, σ_star
end


function multitask_gp_predict(result, s::Int, x::AbstractVector)
    μ, σ = icm_predict(result, s, Float32.(x))
    return μ, σ
end