#############################
# Multi-task Gaussian Process - Full Sensor Version
#############################

# include("../src/get_sample_from_ieee37.jl")
# include("../src/get_ieee37_multitask_data.jl")
include("../src/implement_data.jl")

using Flux
using LinearAlgebra
using Plots
using Statistics
using Random
using ProgressMeter
using DataFrames
using JLD2
using CSV

# Set plot defaults
gr(fontfamily="Arial", legendfontsize=7, guidefontsize=9, titlefontsize=9)


println("\n" * "="^70)
println("Starting Multi-task GP with Full Sensor Suite")
println("="^70)

Random.seed!(42)

println("="^70)
println("Multi-task Gaussian Process (MTGP) - Full Sensor Version")
println("="^70)

println("\n[1] Loading data...")
feeder_dir = "D:/luosipeng/Deep_Learning_in_Distribution_System/data"
res = time_series_ieee37(
    feeder_dir;
    dt_s=0.1,
    hours=24.0,
    sample_every=1,
    collect=[:voltage_bus, :total_power, :bus_injection],
    extract_ymatrix=true
)
ds = extract_requested_dataset(res)

println("\n[2] Building complete multi-sensor dataset...")
data = build_complete_multisensor_data(
    ds;
    max_points_per_sensor = 300
)

println("\n[3] Training multi-task GP...")
result = train_multitask_gp(
    data;
    num_epochs = 100,
    lr = 0.005,
    verbose = true,
    jitter = 1.0f-4
)

println("\n[4] Final hyperparameters:")
println("="^70)
println("Global parameters:")
println("  σ_g = $(round(result.σ_g, digits=4))")
println("  ℓ_g = $(round(result.ℓ_g, digits=4)) hours")
println("\nLocal parameters (top 10 sensors by SNR):")

# Sort sensors by SNR
snrs = result.σ_s ./ result.σ_noise
sorted_indices = sortperm(snrs, rev=true)

for i in 1:min(10, data.S)
    s = sorted_indices[i]
    println("  $(data.sensor_names[s]):")
    println("    σ_s = $(round(result.σ_s[s], digits=4))")
    println("    ℓ_s = $(round(result.ℓ_s[s], digits=4)) hours")
    println("    σ_noise = $(round(result.σ_noise[s], digits=4))")
    println("    SNR = $(round(snrs[s], digits=2))")
end

println("\n[5] Generating visualizations...")

# Select diverse sensors for visualization
# Pick 2 SCADA, 2 AMI, 1 PMU (if available)
scada_indices = findall(x -> x == :SCADA, data.sensor_types)
ami_indices = findall(x -> x == :AMI, data.sensor_types)
pmu_indices = findall(x -> x == :PMU, data.sensor_types)

selected_indices = Int[]
if length(scada_indices) >= 2
    append!(selected_indices, scada_indices[1:2])
end
if length(ami_indices) >= 2
    append!(selected_indices, ami_indices[1:2])
end
if length(pmu_indices) >= 1
    push!(selected_indices, pmu_indices[1])
end

# Limit to 9 plots maximum
selected_indices = selected_indices[1:min(9, length(selected_indices))]

plots_list = []

for s in selected_indices
    x_test = range(minimum(data.times[s]), maximum(data.times[s]), length=200)
    μ_pred, σ_pred = multitask_gp_predict(result, s, collect(Float32, x_test))
    
    p = plot(x_test, μ_pred,
                ribbon = 1.96 .* σ_pred,
                label = "Pred (95% CI)",
                xlabel = "Time (hours)",
                ylabel = "Value",
                title = data.sensor_names[s],
                linewidth = 2,
                fillalpha = 0.3,
                legend = :topright,
                size = (500, 350),
                margin = 4Plots.mm,
                titlefontsize = 8)
    
    scatter!(p, data.times[s], data.values[s],
                label = "Data",
                markersize = 1.5,
                alpha = 0.6,
                color = :red)
    
    push!(plots_list, p)
end

# Combined plot
n_plots = length(plots_list)
layout = (ceil(Int, n_plots/3), 3)
combined = plot(plots_list..., 
                layout = layout, 
                size = (1400, 320 * layout[1]),
                margin = 6Plots.mm)
display(combined)
savefig(combined, "multitask_gp_predictions_full.png")
println("  ✓ Saved: multitask_gp_predictions_full.png")

# Loss curve
valid_losses = filter(x -> isfinite(x) && x > 0, result.losses)
if !isempty(valid_losses)
    p_loss = plot(valid_losses, 
                    xlabel = "Epoch", 
                    ylabel = "Total NLL",
                    title = "Training Loss", 
                    linewidth = 2, 
                    legend = false,
                    yscale = :log10, 
                    size = (700, 450),
                    margin = 5Plots.mm)
    display(p_loss)
    savefig(p_loss, "multitask_gp_loss_full.png")
    println("  ✓ Saved: multitask_gp_loss_full.png")
end

# Hyperparameter evolution
if !isempty(result.ℓ_g_history) && !isempty(result.σ_g_history)
    p_hyper = plot(result.ℓ_g_history,
                    label = "Global length scale (ℓ_g)",
                    xlabel = "Epoch",
                    ylabel = "Value (normalized)",
                    title = "Hyperparameter Evolution",
                    linewidth = 2,
                    legend = :topright,
                    size = (700, 450),
                    margin = 5Plots.mm)
    plot!(p_hyper, result.σ_g_history,
            label = "Global signal variance (σ_g)",
                        linewidth = 2)
    display(p_hyper)
    savefig(p_hyper, "multitask_gp_hyperparams_full.png")
    println("  ✓ Saved: multitask_gp_hyperparams_full.png")
end

# SNR analysis plot
println("\n[6] Generating SNR analysis...")
snrs = result.σ_s ./ result.σ_noise
sorted_indices = sortperm(snrs, rev=true)

# Color by sensor type
colors = [data.sensor_types[i] == :SCADA ? :blue : 
            (data.sensor_types[i] == :AMI ? :green : :red) 
            for i in sorted_indices]

p_snr = bar(1:data.S, snrs[sorted_indices],
            xlabel = "Sensor Index (sorted by SNR)",
            ylabel = "Signal-to-Noise Ratio",
            title = "SNR by Sensor (Blue=SCADA, Green=AMI, Red=PMU)",
            legend = false,
            color = colors,
            size = (1000, 500),
            margin = 5Plots.mm,
            xticks = (1:5:data.S, string.(1:5:data.S)))
display(p_snr)
savefig(p_snr, "multitask_gp_snr_analysis.png")
println("  ✓ Saved: multitask_gp_snr_analysis.png")

# Length scale analysis
println("\n[7] Generating length scale analysis...")
sorted_ls_indices = sortperm(result.ℓ_s, rev=true)
colors_ls = [data.sensor_types[i] == :SCADA ? :blue : 
                (data.sensor_types[i] == :AMI ? :green : :red) 
                for i in sorted_ls_indices]

p_ls = bar(1:data.S, result.ℓ_s[sorted_ls_indices],
            xlabel = "Sensor Index (sorted by length scale)",
            ylabel = "Length Scale (hours)",
            title = "Local Length Scales (Blue=SCADA, Green=AMI, Red=PMU)",
            legend = false,
            color = colors_ls,
            size = (1000, 500),
            margin = 5Plots.mm,
            xticks = (1:5:data.S, string.(1:5:data.S)))
hline!(p_ls, [result.ℓ_g], 
        label = "Global ℓ_g", 
        linewidth = 2, 
        linestyle = :dash, 
        color = :black,
        legend = :topright)
display(p_ls)
savefig(p_ls, "multitask_gp_lengthscale_analysis.png")
println("  ✓ Saved: multitask_gp_lengthscale_analysis.png")

# Summary statistics
println("\n[8] Summary Statistics:")
println("="^70)
println("Signal-to-Noise Ratio:")
println("  Mean SNR: $(round(mean(snrs), digits=2))")
println("  Median SNR: $(round(median(snrs), digits=2))")
println("  Max SNR: $(round(maximum(snrs), digits=2)) ($(data.sensor_names[argmax(snrs)]))")
println("  Min SNR: $(round(minimum(snrs), digits=2)) ($(data.sensor_names[argmin(snrs)]))")

println("\nLength Scales:")
println("  Global ℓ_g: $(round(result.ℓ_g, digits=4)) hours")
println("  Mean local ℓ_s: $(round(mean(result.ℓ_s), digits=4)) hours")
println("  Median local ℓ_s: $(round(median(result.ℓ_s), digits=4)) hours")
println("  Max local ℓ_s: $(round(maximum(result.ℓ_s), digits=4)) hours")
println("  Min local ℓ_s: $(round(minimum(result.ℓ_s), digits=4)) hours")

println("\nNoise Levels:")
println("  Mean σ_noise: $(round(mean(result.σ_noise), digits=4))")
println("  Median σ_noise: $(round(median(result.σ_noise), digits=4))")

# Sensor type breakdown
scada_snrs = snrs[data.sensor_types .== :SCADA]
ami_snrs = snrs[data.sensor_types .== :AMI]
pmu_snrs = snrs[data.sensor_types .== :PMU]

println("\nSNR by Sensor Type:")
if !isempty(scada_snrs)
    println("  SCADA - Mean: $(round(mean(scada_snrs), digits=2)), " *
            "Median: $(round(median(scada_snrs), digits=2))")
end
if !isempty(ami_snrs)
    println("  AMI   - Mean: $(round(mean(ami_snrs), digits=2)), " *
            "Median: $(round(median(ami_snrs), digits=2))")
end
if !isempty(pmu_snrs)
    println("  PMU   - Mean: $(round(mean(pmu_snrs), digits=2)), " *
            "Median: $(round(median(pmu_snrs), digits=2))")
end

println("\n" * "="^70)
println("Complete!")
println("="^70)

if result !== nothing
unified_result = generate_1min_resolution_predictions(result)
end
