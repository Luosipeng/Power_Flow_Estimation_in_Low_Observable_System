"""
Compute correlation matrix between sensors using learned kernel
"""
function compute_sensor_correlation_matrix(result)
    S = result.data.S
    corr_matrix = zeros(Float32, S, S)
    
    # Use a common time point (e.g., midpoint)
    t_ref = Float32[0.0]  # Normalized time
    
    for i in 1:S
        for j in 1:S
            K_ij = compute_multitask_kernel(
                t_ref, t_ref, i, j,
                result.σ_g / result.norm_params.y_stds[i],
                result.ℓ_g / result.norm_params.x_std,
                result.σ_s ./ result.norm_params.y_stds,
                result.ℓ_s ./ result.norm_params.x_std
            )
            
            K_ii = compute_multitask_kernel(
                t_ref, t_ref, i, i,
                result.σ_g / result.norm_params.y_stds[i],
                result.ℓ_g / result.norm_params.x_std,
                result.σ_s ./ result.norm_params.y_stds,
                result.ℓ_s ./ result.norm_params.x_std
            )
            
            K_jj = compute_multitask_kernel(
                t_ref, t_ref, j, j,
                result.σ_g / result.norm_params.y_stds[j],
                result.ℓ_g / result.norm_params.x_std,
                result.σ_s ./ result.norm_params.y_stds,
                result.ℓ_s ./ result.norm_params.x_std
            )
            
            corr_matrix[i, j] = K_ij[1] / sqrt(K_ii[1] * K_jj[1])
        end
    end
    
    return corr_matrix
end