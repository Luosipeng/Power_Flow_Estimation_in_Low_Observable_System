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
include("../src/build_noise_precision_beta.jl")
include("../src/GPBMC.jl")
include("../src/LNBMC.jl")
include("../ios/read_mat_opendss.jl")
include("../src/lack_phase_analysis.jl")
include("../data/sensor_location.jl")

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

pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases, pmu_sensors, scada_sensors, ami_sensors = FAD70_config()
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
    batch_data_17, batch_data_18); pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases
)

println("\n[2] 🏗️ Building complete multi-sensor dataset...")
data = build_complete_multisensor_data(
    ds;
    max_points_per_sensor = 300,
    pmu_sensors, scada_sensors, ami_sensors
)

noise_levels = [0.01, 0.05]  # 1% and 5% of signal standard deviation
missing_percentages = [0.6]
mtgp_epochs = 200
# tolerance = 1e-6
c = 1e-7
d = 1e-7
FAD = 0.7  # Base MVA for FAD10 system

# Load phase-loss configuration
println("\n[3] Loading phase-loss configuration...")
(voltage_mag_a, voltage_mag_b, voltage_mag_c,
 voltage_ang_a, voltage_ang_b, voltage_ang_c,
 power_p_a, power_p_b, power_p_c,
 power_q_a, power_q_b, power_q_c) = read_all_opendss_data()
lack_a, lack_b, lack_c = lack_phase_analysis(voltage_mag_a, voltage_mag_b, voltage_mag_c)

function add_gaussian_noise(data::MultiSensorData, noise_level::Float64; seed::Int=42)
    Random.seed!(seed)
    
    # Create a copy of the original data
    noisy_data = MultiSensorData(
        data.S,
        copy.(data.times),
        copy.(data.values),
        copy(data.sensor_names),
        copy(data.sensor_types),
        copy(data.is_zero_load)
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
    
    # Add noise to ALL sensor data
    noisy_data = add_gaussian_noise(data, noise_level, seed=42)
    
    for (j, missing_pct) in enumerate(missing_percentages)
        println("\n[Noise $(Int(noise_level*100))% | Missing $(Int(missing_pct*100))%] Running ...")
        
        missing_noisy_data, removed_times, removed_values = create_missing_data(noisy_data, missing_pct, seed=42)
        
        # 1. Train MTGP
        println("  > Training MTGP...")
        mtgp_result = train_icm_mtgp(missing_noisy_data; num_epochs=mtgp_epochs, lr=0.01, verbose=false)
        
        # 2. Train Linear Interpolation
        println("  > Training Linear Interpolation...")
        linear_result = train_linear_interpolation(missing_noisy_data; verbose=false)

        # 3. Evaluation (SBMC Stage)
        # 🔥 注意：这里传入了 noise_level 参数
        println("  > Running SBMC Correction (Adaptive Rank)...")
        
        GP_result = mtgpbmc(mtgp_result, noise_level, lack_a, lack_b, lack_c)
        GP_miae_power = GP_result.miae_power
        GP_miae_theta = GP_result.miae_theta
        GP_mape_voltage = GP_result.mape_voltage
        LN_miae_power, LN_miae_theta, LN_mape_voltage = linear_sbmc(linear_result, FAD, noise_level, lack_a, lack_b, lack_c)
        
        # 4. Print Results
        println("-"^40)
        println("Noise Level = $(Int(noise_level*100))% | GP MIAE Power    = $(GP_miae_power)%")
        println("Noise Level = $(Int(noise_level*100))% | GP MIAE Theta    = $(GP_miae_theta)%")
        println("Noise Level = $(Int(noise_level*100))% | GP MAPE Voltage  = $(GP_mape_voltage) %")
        println("-"^20)
        println("Noise Level = $(Int(noise_level*100))% | LN MIAE Power    = $(LN_miae_power)%")
        println("Noise Level = $(Int(noise_level*100))% | LN MIAE Theta    = $(LN_miae_theta)%")
        println("Noise Level = $(Int(noise_level*100))% | LN MAPE Voltage  = $(LN_mape_voltage) %")
        println("-"^40)
    end
end