# FAD 配置
# 10% FAD 配置：PMU：3， SCADA：8,12,18,33,34， AMI：15,22,25,36,37
# 20% FAD 配置: PMU: 3, SCADA: 8,12,18,21,26,33,34, AMI：4,5,6,9,15,19,22,23,25,30,31,36,37
# 30% FAD 配置: PMU: 3, SCADA: 8,12,18,21,26,33,34, AMI：2,4,5,6,7,9,10,13,15,16,19,21,22,23,24,25,28,30,31,32,36,37
# 40% FAD 配置: PMU: 3, SCADA: 8,12,16,18,26,33,34, AMI:2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,21,22,23,24,25,26,27,28,29,30,31,32,36,37
# 50% FAD 配置: PMU: 3,4, SCADA: 2,5,6,8,12,13,14,15,16,18,20,22,24,26,30,31,33,34, AMI:2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29,30,31,32,33,36,37
# 60% FAD 配置: PMU: 3,4,9,12,16,18,20,32, SCADA: 2,5,6,8,11,13,14,15,17,19,21,22,24,26,30,31,33,34, AMI:2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,31,32,33,36,37
# 70% FAD 配置: PMU: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, AMI:2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 31, 32, 33, 35, 36, 37, 38
# 80% FAD 配置: PMU: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 AMI:2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 31, 32, 33, 35, 36, 37, 38
# 90% FAD 配置: PMU: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, AMI: 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 35, 36, 37, 38
include("../src/implement_data.jl")
include("../src/generate_series_data.jl")
include("../ios/read_mat.jl")
include("../src/extract_requested_dataset_multibatch.jl")
include("../src/build_complete_multisensor_data.jl")
include("../src/data_processing.jl")
include("../src/multi_task_gaussian.jl")
include("../src/gaussian_prediction.jl")
include("../src/linear_imputation.jl")
include("../src/missing_data_evaluation.jl")
include("../src/build_observed_matrix_z.jl")
include("../src/matrix_completion.jl")
include("../src/ac_dc_power_flow.jl")
include("../src/build_noise_precision_beta.jl")
include("../src/GPBMC.jl")
include("../src/LNBMC.jl")

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

println("\n[1] 📂 Loading data...")
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

println("\n[2] 🏗️ Building complete multi-sensor dataset...")
data = build_complete_multisensor_data(
    ds;
    max_points_per_sensor = 300
)

noise_levels = [0.01, 0.10]  # 1% and 10% of signal standard deviation
missing_percentages = [0.6]
mtgp_epochs = 200
tolerance = 1e-6
c = 1e-7
d = 1e-7
max_iter = 400

root_bus = 1
Vref = 1.0
inv_bus = 18
rec_bus = 1
eta = 0.9
nac = 33
ndc = 4
n_noise = length(noise_levels)
n_missing = length(missing_percentages)
FAD = 0.5

function add_gaussian_noise(data::MultiSensorData, noise_level::Float64; seed::Int=42)
    Random.seed!(seed)
    
    # Create a copy of the original data
    noisy_data = MultiSensorData(
        data.S,
        copy.(data.times),
        copy.(data.values),
        copy(data.sensor_names),
        copy(data.sensor_types)
    )
    
    # Add Gaussian noise to each sensor's values
    for s in 1:data.S
        if !isempty(data.values[s])
            # Calculate noise standard deviation as percentage of signal standard deviation
            signal_std = std(data.values[s])
            noise_std = noise_level * signal_std
            
            # Add noise
            noise = randn(length(data.values[s])) .* noise_std
            noisy_data.values[s] .+= noise
        end
    end
    
    return noisy_data
end

for (i, noise_level) in enumerate(noise_levels)
    println("\n" * "="^60)
    println("🔊 NOISE LEVEL: $(Int(noise_level*100))% of signal standard deviation")
    println("="^60)
    
    # Add noise to ALL sensor data (not just target sensors)
    noisy_data = add_gaussian_noise(data, noise_level, seed=42)
    
    for (j, missing_pct) in enumerate(missing_percentages)
        println("\n[Noise $(Int(noise_level*100))% | Missing $(Int(missing_pct*100))%] Running ...")
        
        missing_noisy_data, removed_times, removed_values = create_missing_data(noisy_data, missing_pct, seed=42)
        
        # Train MTGP using ALL sensors for multi-task learning
        mtgp_result = train_icm_mtgp(missing_noisy_data; num_epochs=mtgp_epochs, lr=0.01, verbose=false)
        
        # Train Linear Interpolation using ALL sensors
        linear_result = train_linear_interpolation(missing_noisy_data; verbose=false)

        GP_miae_power, GP_miae_theta, GP_mape_voltage = mtgpbmc(mtgp_result; max_iter, tolerance, c, d, nac, ndc, root_bus, inv_bus, rec_bus, eta, Vref, noise_level)
        LN_miae_power, LN_miae_theta, LN_mape_voltage = linear_sbmc(linear_result; max_iter, tolerance, c, d, nac, ndc, root_bus, inv_bus, rec_bus, eta, Vref, FAD, noise_level)
        println("Noise Level = $(Int(noise_level*100))% | GP MIAE Power = $(GP_miae_power) %")
        println("Noise Level = $(Int(noise_level*100))% | GP MIAE Theta = $(GP_miae_theta) %")
        println("Noise Level = $(Int(noise_level*100))% | GP MAPE Voltage = $(GP_mape_voltage) %")
        println("Noise Level = $(Int(noise_level*100))% | LN MIAE Power = $(LN_miae_power) %")
        println("Noise Level = $(Int(noise_level*100))% | LN MIAE Theta = $(LN_miae_theta) %")
        println("Noise Level = $(Int(noise_level*100))% | LN MAPE Voltage = $(LN_mape_voltage) %")
    end
end