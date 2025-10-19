# Predict jointly for a given task s
function icm_predict(result, s::Int, x_test::Vector{Float32})
    data = result.data
    S = data.S
    normp = result.norm_params

    x_test_norm = (x_test .- normp.x_mean) ./ normp.x_std

    joint_pack = result.joint_pack
    x_all = joint_pack.x_all
    y_all = joint_pack.y_all

    B = build_task_correlation_matrix(result.L_params)

    # Predict in normalized space for stability
    σ_time = exp(result.log_σ_time[])
    ℓ_time = exp(result.log_ℓ_time[])
    σ_locals = exp.(result.log_σ_locals)
    ℓ_locals = exp.(result.log_ℓ_locals)
    σ_noise  = exp.(result.log_σ_noise)

    # K_all with noise inside construction (same as training)
    K_all = construct_joint_K_all(joint_pack.norm_data.times, B, σ_time, ℓ_time, σ_locals, ℓ_locals, σ_noise, 1f-6)
    K_sym = Hermitian(0.5f0 .* (K_all .+ K_all'))
    L = cholesky(K_sym).L

    # K_x_star: N x n*
    n_star = length(x_test_norm)
    K_x_star = zeros(Float32, length(x_all), n_star)

    offsets = joint_pack.offsets
    n_per_task = joint_pack.n_per_task

    for i in 1:S
        xi = joint_pack.norm_data.times[i]
        K_time_is = rbf_kernel(xi, x_test_norm, σ_time, ℓ_time)
        K_local = (i == s) ? rbf_kernel(xi, x_test_norm, σ_locals[s], ℓ_locals[s]) :
                             zeros(Float32, size(K_time_is))
        block = B[i,s] .* K_time_is .+ K_local
        os = offsets[i]; ns = n_per_task[i]
        @views K_x_star[os+1:os+ns, :] .= block
    end

    m_all = mean_forward(result.mean_func, x_all)
    r = y_all .- m_all
    α = L' \ (L \ r)

    m_star = vec(result.mean_func(reshape(x_test_norm, 1, :)))
    μ_star_norm = m_star .+ K_x_star' * α

    v = L \ K_x_star
    Kss = B[s,s] .* rbf_kernel(x_test_norm, x_test_norm, σ_time, ℓ_time) .+
          rbf_kernel(x_test_norm, x_test_norm, σ_locals[s], ℓ_locals[s])
    cov_star = Kss .- v' * v
    σ_star_norm = sqrt.(max.(diag(cov_star), 1f-8))

    μ_star = μ_star_norm .* joint_pack.norm_params.y_stds[s] .+ joint_pack.norm_params.y_means[s]
    σ_star = σ_star_norm .* joint_pack.norm_params.y_stds[s]
    return μ_star, σ_star
end

function multitask_gp_predict(result, s::Int, x::AbstractVector)
    μ, σ = icm_predict(result, s, Float32.(x))
    return μ, σ
end