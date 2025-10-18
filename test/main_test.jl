#############################
# Multi-task Gaussian Process - Full Sensor Version
#############################
include("../src/implement_data.jl")
include("../src/generate_series_data.jl")
include("../ios/read_mat.jl")
include("../src/extract_requested_dataset_multibatch.jl")
include("../src/build_complete_multisensor_data.jl")
include("../src/data_processing.jl")
include("../src/multi_task_gaussian.jl")
include("../src/gaussian_prediction.jl")

using Flux
using LinearAlgebra
using Plots
using Statistics
using Random
using ProgressMeter
using DataFrames
using JLD2
using MAT
using CSV

# Set plot defaults
gr(fontfamily="Arial", legendfontsize=7, guidefontsize=9, titlefontsize=9)

println("\n" * "="^70)
println("Starting Multi-task GP with Full Sensor Suite")
println("="^70)

# Random.seed!(42)

println("\n[1] Loading data...")

(batch_data_1, batch_data_2, batch_data_3, batch_data_4,
            batch_data_5, batch_data_6, batch_data_7, batch_data_8,
            batch_data_9, batch_data_10, batch_data_11, batch_data_12,
            batch_data_13, batch_data_14, batch_data_15, batch_data_16, 
            batch_data_17, batch_data_18) = read_mat()

ds = extract_requested_dataset_multibatch(
    (batch_data_1, batch_data_2, batch_data_3, batch_data_4,
    batch_data_5, batch_data_6, batch_data_7, batch_data_8,
    batch_data_9, batch_data_10, batch_data_11, batch_data_12,
    batch_data_13, batch_data_14, batch_data_15, batch_data_16,
    batch_data_17, batch_data_18)
)


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
    sensor_name = data.sensor_names[s]
    sensor_type = data.sensor_types[s]

    if sensor_type == :PMU
        # PMU：使用原始高频时间戳进行预测与绘图
        x_test = data.times[s]  # 原始0.1s时间戳（小时单位）
        μ_pred, σ_pred = multitask_gp_predict(result, s, x_test)

        p = plot(x_test, μ_pred,
                 ribbon = 1.96 .* σ_pred,
                 label = "GP Pred (95% CI)",
                 xlabel = "Time (hours)",
                 ylabel = "Value",
                 title = sensor_name * " (PMU, 0.1s)",
                 linewidth = 1.5,
                 fillalpha = 0.25,
                 legend = :topright,
                 size = (600, 350),
                 margin = 4Plots.mm)

        # 高频原始测量点：用细线或小点以避免淹没
        scatter!(p, data.times[s], data.values[s],
                 label = "Data (0.1s)",
                 markersize = 1.0,
                 alpha = 0.5,
                 color = :red,
                 markerstrokewidth = 0)

        push!(plots_list, p)
    else
        # 非PMU：保持原逻辑（200个均匀点）
        x_test = range(minimum(data.times[s]), maximum(data.times[s]), length=200)
        μ_pred, σ_pred = multitask_gp_predict(result, s, collect(Float32, x_test))

        p = plot(x_test, μ_pred,
                 ribbon = 1.96 .* σ_pred,
                 label = "Pred (95% CI)",
                 xlabel = "Time (hours)",
                 ylabel = "Value",
                 title = sensor_name,
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
