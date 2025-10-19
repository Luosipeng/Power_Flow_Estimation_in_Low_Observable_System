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

println("\n[3] Training multi-task GP (ICM/LMC)...")
result = train_icm_mtgp(data; num_epochs = 200, lr = 0.01, verbose = true)

println("\n[4] Final hyperparameters (reported in original scale):")
println("="^70)
println("Global time kernel:")
println("  σ_time = $(round(result.σ_time, digits=4))")
println("  ℓ_time = $(round(result.ℓ_time, digits=4)) hours")
println("\nLocal per-sensor kernel (top 10 by σ_locals/σ_noise):")

snrs = (result.σ_locals) ./ (result.σ_noise .+ 1e-12)
sorted_indices = sortperm(snrs, rev=true)
for i in 1:min(10, data.S)
    s = sorted_indices[i]
    println("  $(data.sensor_names[s]):")
    println("    σ_local = $(round(result.σ_locals[s], digits=4))")
    println("    ℓ_local = $(round(result.ℓ_locals[s], digits=4)) hours")
    println("    σ_noise = $(round(result.σ_noise[s], digits=4))")
    println("    SNR = $(round(snrs[s], digits=2))")
end

println("\n[5] Visualizations...")
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
selected_indices = selected_indices[1:min(9, length(selected_indices))]

plots_list = []
for s in selected_indices
    x_test = range(minimum(data.times[s]), maximum(data.times[s]), length=200)
    μ_pred, σ_pred = icm_predict(result, s, collect(Float32, x_test))
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

n_plots = length(plots_list)
if n_plots > 0
    layout = (ceil(Int, n_plots/3), 3)
    combined = plot(plots_list..., 
                    layout = layout, 
                    size = (1400, 320 * layout[1]),
                    margin = 6Plots.mm)
    display(combined)
    savefig(combined, "mtgp_icm_predictions.png")
    println("  ✓ Saved: mtgp_icm_predictions.png")
end

valid_losses = filter(x -> isfinite(x) && x > 0, result.losses)
if !isempty(valid_losses)
    p_loss = plot(valid_losses, 
                    xlabel = "Epoch", 
                    ylabel = "Total NLL",
                    title = "Training Loss (Joint LML)", 
                    linewidth = 2, 
                    legend = false,
                    yscale = :log10, 
                    size = (700, 450),
                    margin = 5Plots.mm)
    display(p_loss)
    savefig(p_loss, "mtgp_icm_loss.png")
    println("  ✓ Saved: mtgp_icm_loss.png")
end

if data.S > 0
    sorted_indices = sortperm(snrs, rev=true)
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
    savefig(p_snr, "mtgp_icm_snr.png")
    println("  ✓ Saved: mtgp_icm_snr.png")
end

println("\n[6] Summary:")
println("="^70)
println("  Mean SNR: $(round(mean(snrs), digits=2))")
println("  Median SNR: $(round(median(snrs), digits=2))")
println("  Max SNR: $(round(maximum(snrs), digits=2)) ($(data.sensor_names[argmax(snrs)]))")
println("  Min SNR: $(round(minimum(snrs), digits=2)) ($(data.sensor_names[argmin(snrs)]))")
println("  Global ℓ_time: $(round(result.ℓ_time, digits=4)) hours")
println("  Mean ℓ_local: $(round(mean(result.ℓ_locals), digits=4)) hours")
println("  Mean σ_noise: $(round(mean(result.σ_noise), digits=4))")

println("\n" * "="^70)
println("Complete!")
println("="^70)


# ============================================
# Entry
# ============================================

println("\n" * "="^70)
println("Starting Multi-task GP (ICM/LMC) with Full Sensor Suite - No In-place")
println("="^70)


if result !== nothing
unified_result = generate_1min_resolution_predictions(result)
end