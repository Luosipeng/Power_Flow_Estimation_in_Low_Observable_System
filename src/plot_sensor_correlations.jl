"""
Plot sensor correlation heatmap
"""
function plot_sensor_correlations(result; max_sensors::Int=20)
    S = min(result.data.S, max_sensors)
    
    println("\nComputing sensor correlations (first $S sensors)...")
    
    # Compute correlation for subset
    corr_subset = zeros(Float32, S, S)
    t_ref = Float32[0.0]
    
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
            
            corr_subset[i, j] = K_ij[1] / sqrt(K_ii[1] * K_jj[1])
        end
    end
    
    # Create short labels
    short_names = [length(name) > 15 ? name[1:12]*"..." : name 
                   for name in result.data.sensor_names[1:S]]
    
    p = heatmap(corr_subset,
                xlabel = "Sensor Index",
                ylabel = "Sensor Index",
                title = "Learned Sensor Correlations",
                color = :RdBu,
                clims = (-1, 1),
                size = (800, 700),
                margin = 8Plots.mm,
                xticks = (1:S, short_names),
                yticks = (1:S, short_names),
                xrotation = 45,
                aspect_ratio = :equal)
    
    display(p)
    savefig(p, "multitask_gp_correlations.png")
    println("  ✓ Saved: multitask_gp_correlations.png")
    
    return corr_subset
end