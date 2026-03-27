# ============================================
# MTGP 训练时间基准测试脚本
# 测试不同状态数量对MTGP训练时间的影响
# ============================================

include("../src/implement_data.jl")
include("../src/generate_series_data.jl")
include("../ios/read_mat_opendss.jl")
include("../src/extract_requested_dataset_opendss.jl")
include("../src/build_complete_multisensor_data.jl")
include("../src/data_processing.jl")
include("../src/multi_task_gaussian.jl")
include("../src/gaussian_prediction.jl")
include("../src/linear_imputation.jl")
include("../src/missing_data_evaluation.jl")
include("../data/sensor_location_123.jl")
include("../src/lack_phase_analysis.jl")

using Flux
using LinearAlgebra
using Statistics
using Random
using ProgressMeter
using DataFrames
using CSV
using Dates

# ============================================
# 配置参数
# ============================================
const MTGP_EPOCHS = 50           # 训练轮数（固定以便公平比较）
const NOISE_LEVEL = 0.01         # 噪声水平
const MISSING_PCT = 0.3          # 缺失数据比例
const RANDOM_SEED = 42           # 随机种子
const NUM_TRIALS = 3             # 每个配置运行的次数（取平均）

# ============================================
# 辅助函数：添加高斯噪声
# ============================================
function add_gaussian_noise(data::MultiSensorData, noise_level::Float64; seed::Int=42)
    Random.seed!(seed)
    
    noisy_data = MultiSensorData(
        data.S,
        copy.(data.times),
        copy.(data.values),
        copy(data.sensor_names),
        copy(data.sensor_types),
        copy(data.is_zero_load)
    )
    
    for s in 1:data.S
        if !data.is_zero_load[s] && !isempty(data.values[s])
            signal_std = std(data.values[s])
            noise_std = noise_level * signal_std
            noise = randn(length(data.values[s])) .* noise_std
            noisy_data.values[s] .+= noise
        end
    end
    
    return noisy_data
end

# ============================================
# 辅助函数：创建子数据集（选择部分传感器）
# ============================================
function create_subset_data(data::MultiSensorData, num_sensors::Int; seed::Int=42)
    Random.seed!(seed)
    
    # 找出非零负荷传感器的索引
    active_indices = findall(.!data.is_zero_load)
    
    if num_sensors > length(active_indices)
        num_sensors = length(active_indices)
        println("  ⚠️ Requested sensors exceed active sensors, using all $(num_sensors)")
    end
    
    # 随机选择传感器
    selected_indices = sort(active_indices[randperm(length(active_indices))[1:num_sensors]])
    
    return MultiSensorData(
        num_sensors,
        data.times[selected_indices],
        data.values[selected_indices],
        data.sensor_names[selected_indices],
        data.sensor_types[selected_indices],
        data.is_zero_load[selected_indices]
    )
end

# ============================================
# 辅助函数：获取活跃传感器数量
# ============================================
function count_active_sensors(data::MultiSensorData)
    return sum(.!data.is_zero_load)
end

# ============================================
# 基准测试函数
# ============================================
function benchmark_mtgp_training(data::MultiSensorData, num_sensors::Int; 
                                  num_epochs::Int=MTGP_EPOCHS, 
                                  lr::Float64=0.01,
                                  seed::Int=RANDOM_SEED)
    # 创建子数据集
    subset_data = create_subset_data(data, num_sensors, seed=seed)
    
    # 添加噪声
    noisy_data = add_gaussian_noise(subset_data, NOISE_LEVEL, seed=seed)
    
    # 创建缺失数据
    missing_data, _, _ = create_missing_data(noisy_data, MISSING_PCT, seed=seed)
    
    # 计时训练
    start_time = time()
    result = train_icm_mtgp(missing_data; num_epochs=num_epochs, lr=lr, verbose=false)
    end_time = time()
    
    training_time = end_time - start_time
    final_loss = result.losses[end]
    
    return training_time, final_loss, count_active_sensors(missing_data)
end

# ============================================
# 主函数
# ============================================
function run_mtgp_time_benchmark()
    println("\n" * "="^70)
    println("🚀 MTGP 训练时间基准测试")
    println("="^70)
    println("配置:")
    println("  - MTGP Epochs: $MTGP_EPOCHS")
    println("  - Noise Level: $(Int(NOISE_LEVEL*100))%")
    println("  - Missing Percentage: $(Int(MISSING_PCT*100))%")
    println("  - Number of Trials: $NUM_TRIALS")
    println("="^70)
    
    # 加载完整数据集
    println("\n[1] 📂 Loading data using FAD20_config_ieee123...")
    pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases, 
        pmu_sensors, scada_sensors, ami_sensors = FAD20_config_ieee123()
    
    (voltage_mag_a, voltage_mag_b, voltage_mag_c,
     voltage_ang_a, voltage_ang_b, voltage_ang_c,
     power_p_a, power_p_b, power_p_c,
     power_q_a, power_q_b, power_q_c) = read_all_opendss_data()
    
    lack_a, lack_b, lack_c = lack_phase_analysis(voltage_mag_a, voltage_mag_b, voltage_mag_c)
    
    ds = extract_requested_dataset_opendss(
        voltage_mag_a, voltage_mag_b, voltage_mag_c,
        voltage_ang_a, voltage_ang_b, voltage_ang_c,
        power_p_a, power_p_b, power_p_c,
        power_q_a, power_q_b, power_q_c;
        pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases
    )
    
    println("\n[2] 🏗️ Building complete multi-sensor dataset...")
    full_data = build_complete_multisensor_data(
        ds;
        max_points_per_sensor = 200,
        pmu_sensors, scada_sensors, ami_sensors
    )
    
    total_sensors = full_data.S
    active_sensors = count_active_sensors(full_data)
    
    println("\n[3] 📊 Dataset Summary:")
    println("  - Total Sensors: $total_sensors")
    println("  - Active Sensors (non-zero load): $active_sensors")
    
    # 定义要测试的状态数量
    # 根据活跃传感器数量动态生成测试点
    max_test_sensors = min(active_sensors, 200)
    sensor_counts = unique(vcat(
        [5, 10, 15, 20, 25, 30],           # 小规模
        [40, 50, 60, 70, 80, 90, 100],     # 中规模
        [120, 150, 180, 200]               # 大规模（如果可用）
    ))
    sensor_counts = filter(x -> x <= max_test_sensors, sensor_counts)
    
    println("\n[4] 🧪 Running benchmarks for sensor counts: $sensor_counts")
    println("-"^70)
    
    # 结果存储
    results = DataFrame(
        num_sensors = Int[],
        actual_active = Int[],
        trial = Int[],
        training_time_sec = Float64[],
        final_loss = Float32[]
    )
    
    summary_results = DataFrame(
        num_sensors = Int[],
        actual_active = Int[],
        mean_time_sec = Float64[],
        std_time_sec = Float64[],
        min_time_sec = Float64[],
        max_time_sec = Float64[],
        mean_loss = Float32[]
    )
    
    for num_sensors in sensor_counts
        println("\n📊 Testing with $num_sensors sensors...")
        
        trial_times = Float64[]
        trial_losses = Float32[]
        actual_active = 0
        
        for trial in 1:NUM_TRIALS
            println("  Trial $trial/$NUM_TRIALS...")
            
            # 使用不同的种子进行每次试验
            trial_seed = RANDOM_SEED + trial * 100
            
            training_time, final_loss, active = benchmark_mtgp_training(
                full_data, num_sensors;
                num_epochs=MTGP_EPOCHS,
                seed=trial_seed
            )
            
            actual_active = active
            push!(trial_times, training_time)
            push!(trial_losses, final_loss)
            
            push!(results, (num_sensors, actual_active, trial, training_time, final_loss))
            
            println("    ✅ Time: $(round(training_time, digits=2))s, Loss: $(round(final_loss, digits=4))")
        end
        
        # 计算统计量
        mean_time = mean(trial_times)
        std_time = std(trial_times)
        min_time = minimum(trial_times)
        max_time = maximum(trial_times)
        mean_loss = mean(trial_losses)
        
        push!(summary_results, (num_sensors, actual_active, mean_time, std_time, min_time, max_time, mean_loss))
        
        println("  📈 Summary: Mean=$(round(mean_time, digits=2))s ± $(round(std_time, digits=2))s")
    end
    
    # 保存结果
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    
    detail_file = "../mtgp_benchmark_details_$timestamp.csv"
    summary_file = "../mtgp_benchmark_summary_$timestamp.csv"
    
    CSV.write(detail_file, results)
    CSV.write(summary_file, summary_results)
    
    println("\n" * "="^70)
    println("📊 BENCHMARK RESULTS SUMMARY")
    println("="^70)
    println("\n| Sensors | Active | Mean Time (s) | Std (s) | Min (s) | Max (s) |")
    println("|---------|--------|---------------|---------|---------|---------|")
    
    for row in eachrow(summary_results)
        @printf("| %7d | %6d | %13.2f | %7.2f | %7.2f | %7.2f |\n",
                row.num_sensors, row.actual_active, 
                row.mean_time_sec, row.std_time_sec,
                row.min_time_sec, row.max_time_sec)
    end
    
    println("\n📁 Results saved to:")
    println("  - Detail: $detail_file")
    println("  - Summary: $summary_file")
    
    # 计算时间复杂度估计
    println("\n📐 Time Complexity Analysis:")
    if nrow(summary_results) >= 3
        n = summary_results.num_sensors
        t = summary_results.mean_time_sec
        
        # 估计 t = a * n^b 中的 b (时间复杂度指数)
        log_n = log.(n)
        log_t = log.(max.(t, 1e-6))  # 避免 log(0)
        
        # 简单线性回归: log(t) = log(a) + b * log(n)
        mean_log_n = mean(log_n)
        mean_log_t = mean(log_t)
        
        b_estimate = sum((log_n .- mean_log_n) .* (log_t .- mean_log_t)) / 
                     sum((log_n .- mean_log_n).^2)
        
        println("  Estimated time complexity: O(n^$(round(b_estimate, digits=2)))")
        println("  (Based on empirical fit t = a * n^b)")
    end
    
    println("\n✅ Benchmark completed!")
    
    return results, summary_results
end

# ============================================
# 运行基准测试
# ============================================
using Printf

println("🔧 Starting MTGP Time Benchmark Script...")
results, summary = run_mtgp_time_benchmark()
