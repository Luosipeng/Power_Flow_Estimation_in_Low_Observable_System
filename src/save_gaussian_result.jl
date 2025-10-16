function save_results(result, filename::String="multitask_gp_results.txt")
    open(filename, "w") do io
        println(io, "="^70)
        println(io, "Multi-task Gaussian Process Results")
        println(io, "="^70)
        println(io, "\nGlobal Hyperparameters:")
        println(io, "  σ_g = $(result.σ_g)")
        println(io, "  ℓ_g = $(result.ℓ_g) hours")
        
        println(io, "\nPer-Sensor Parameters:")
        println(io, "="^70)
        println(io, "Sensor Name | σ_s | ℓ_s (hrs) | σ_noise | SNR")
        println(io, "-"^70)
        
        snrs = result.σ_s ./ result.σ_noise
        for s in 1:result.data.S
            println(io, "$(rpad(result.data.sensor_names[s], 35)) | " *
                        "$(round(result.σ_s[s], digits=4)) | " *
                        "$(round(result.ℓ_s[s], digits=4)) | " *
                        "$(round(result.σ_noise[s], digits=4)) | " *
                        "$(round(snrs[s], digits=2))")
        end
        
        println(io, "\n" * "="^70)
        println(io, "Summary Statistics:")
        println(io, "  Total sensors: $(result.data.S)")
        println(io, "  Mean SNR: $(round(mean(snrs), digits=2))")
        println(io, "  Median SNR: $(round(median(snrs), digits=2))")
        println(io, "  Mean local ℓ_s: $(round(mean(result.ℓ_s), digits=4)) hours")
        # println(io, "  Final training loss: $(round(result.losses[end], digits=4))")
    end
    
    println("  ✓ Saved: $filename")
end
