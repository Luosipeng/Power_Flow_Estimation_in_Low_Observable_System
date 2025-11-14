#############################
# Multi-task Gaussian Process - Corrected Targeted Sensor Analysis
# 训练：使用所有传感器  |  评估：只计算指定传感器的RMSE
#############################
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

"""
Add Gaussian noise to the sensor data
"""
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

"""
修正后的评估函数 - MTGP用所有传感器训练，但只评估指定传感器

关键修正：
1. MTGP训练：使用所有传感器数据进行多任务学习
2. 线性插值训练：使用所有传感器数据
3. RMSE评估：只计算指定目标传感器的RMSE
4. 数据处理：对所有传感器添加噪声和缺失数据
"""
function evaluate_noise_and_missing_data_impact_corrected(data::MultiSensorData; 
                                              noise_levels::Vector{Float64}=[0.01, 0.10],
                                              missing_percentages::Vector{Float64}=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                              mtgp_epochs::Int=200,
                                              verbose::Bool=true,
                                              target_sensors::Union{Vector{Int}, Vector{String}, Nothing}=nothing,
                                              sensor_types_filter::Union{Vector{Symbol}, Nothing}=nothing,
                                              include_sensor_details::Bool=true)
    
    # 确定要评估RMSE的传感器（不影响训练）
    eval_sensor_indices = target_sensors
    # 确保索引有效
    eval_sensor_indices = filter(x -> 1 <= x <= data.S, eval_sensor_indices)
    
    println("🔧 TRAINING STRATEGY:")
    println("   MTGP: Uses ALL $(data.S) sensors for multi-task learning")
    println("   Linear: Uses ALL $(data.S) sensors for interpolation")
    println("📊 EVALUATION STRATEGY:")
    println("   RMSE calculated on $(length(eval_sensor_indices)) target sensors:")
    for (i, s) in enumerate(eval_sensor_indices[1:min(10, end)])
        println("     $i. $(data.sensor_names[s]) ($(data.sensor_types[s]))")
    end
    if length(eval_sensor_indices) > 10
        println("     ... and $(length(eval_sensor_indices) - 10) more sensors")
    end
    
    results = Dict{String, Any}()
    results["noise_levels"] = noise_levels
    results["missing_percentages"] = missing_percentages
    results["eval_sensor_names"] = data.sensor_names[eval_sensor_indices]  # 只评估的传感器
    results["eval_sensor_indices"] = eval_sensor_indices
    results["eval_sensor_types"] = data.sensor_types[eval_sensor_indices]
    results["all_sensor_names"] = data.sensor_names  # 所有传感器（用于训练）
    results["training_sensor_count"] = data.S  # 训练使用的传感器数量
    
    # Initialize result arrays: [noise_level, missing_percentage, trial]
    n_noise = length(noise_levels)
    n_missing = length(missing_percentages)

    results["mtgp_rmse"] = zeros(Float64, n_noise, n_missing)
    results["linear_rmse"] = zeros(Float64, n_noise, n_missing)
    results["mtgp_rmse_by_sensor"] = Array{Vector{Vector{Float64}}}(undef, n_noise, n_missing)
    results["linear_rmse_by_sensor"] = Array{Vector{Vector{Float64}}}(undef, n_noise, n_missing)
    
    if include_sensor_details
        results["sensor_level_results"] = Dict{String, Any}()
    end
    
    println("\n" * "="^80)
    println("CORRECTED NOISE & MISSING DATA EVALUATION: MTGP vs Linear")
    println("="^80)
    println("🎯 Training: ALL $(data.S) sensors | Evaluation: $(length(eval_sensor_indices)) target sensors")
    println("🔊 Noise levels: ", [Int(nl*100) for nl in noise_levels], "% of signal std")
    println("❌ Missing percentages: ", [Int(mp*100) for mp in missing_percentages], "%")
    println("📈 Total experiments: ", n_noise * n_missing )
    
    for (i, noise_level) in enumerate(noise_levels)
        println("\n" * "="^60)
        println("🔊 NOISE LEVEL: $(Int(noise_level*100))% of signal standard deviation")
        println("="^60)
        
        # Add noise to ALL sensor data (not just target sensors)
        noisy_data = add_gaussian_noise(data, noise_level, seed=42)
        
        for (j, missing_pct) in enumerate(missing_percentages)
            println("\n[Noise $(Int(noise_level*100))% | Missing $(Int(missing_pct*100))%] Running ...")
            
            mtgp_sensor_rmses = []
            linear_sensor_rmses = []
            
            # 存储每个目标传感器的详细结果
            if include_sensor_details
                sensor_detailed_results = Dict{String, Dict{String, Any}}()
            end
            
                missing_noisy_data, removed_times, removed_values = create_missing_data(
        noisy_data, missing_pct, seed=42)
            
            # Train MTGP using ALL sensors for multi-task learning
            if verbose
                println("    🧠 Training MTGP with ALL $(data.S) sensors...")
            end
            mtgp_result = train_icm_mtgp(missing_noisy_data; num_epochs=mtgp_epochs, lr=0.01, verbose=false)
            
            # Train Linear Interpolation using ALL sensors
            if verbose
                println("    📏 Training Linear Interpolation with ALL $(data.S) sensors...")
            end
            linear_result = train_linear_interpolation(missing_noisy_data; verbose=false)
            
            # Evaluate ONLY on target sensors
            mtgp_rmses_sensor = Float64[]
            linear_rmses_sensor = Float64[]
            
            if verbose
                println("    📊 Evaluating on $(length(eval_sensor_indices)) target sensors...")
            end
            
            for s in eval_sensor_indices  # 只评估目标传感器
                if !isempty(removed_times[s])
                    # Get original clean values at removed time points
                    original_clean_values = Float32[]
                    for t in removed_times[s]
                        # Find closest time point in original data
                        time_diffs = abs.(data.times[s] .- t)
                        closest_idx = argmin(time_diffs)
                        if time_diffs[closest_idx] < 0.01  # Within 0.01 time units
                            push!(original_clean_values, data.values[s][closest_idx])
                        end
                    end
                    
                    if !isempty(original_clean_values)
                        # MTGP predictions (trained on all sensors, predict for sensor s)
                        μ_mtgp, σ_mtgp = icm_predict(mtgp_result, s, removed_times[s][1:length(original_clean_values)])
                        mtgp_rmse = sqrt(mean((μ_mtgp .- original_clean_values).^2))
                        
                        # Linear predictions (trained on all sensors, predict for sensor s)
                        μ_linear, σ_linear = linear_predict(linear_result, s, removed_times[s][1:length(original_clean_values)])
                        linear_rmse = sqrt(mean((μ_linear .- original_clean_values).^2))
                        
                        push!(mtgp_rmses_sensor, mtgp_rmse)
                        push!(linear_rmses_sensor, linear_rmse)
                        
                        # 存储单个传感器的详细结果
                        if include_sensor_details
                            sensor_name = data.sensor_names[s]
                            if !haskey(sensor_detailed_results, sensor_name)
                                sensor_detailed_results[sensor_name] = Dict{String, Any}(
                                    "mtgp_rmse" => Float64[],
                                    "linear_rmse" => Float64[],
                                    "improvement" => Float64[],
                                    "sensor_type" => data.sensor_types[s]
                                )
                            end
                            push!(sensor_detailed_results[sensor_name]["mtgp_rmse"], mtgp_rmse)
                            push!(sensor_detailed_results[sensor_name]["linear_rmse"], linear_rmse)
                            improvement = (linear_rmse - mtgp_rmse) / linear_rmse * 100
                            push!(sensor_detailed_results[sensor_name]["improvement"], improvement)
                        end
                    end
                end
            end
            
            # Store results (average over target sensors only)
            if !isempty(mtgp_rmses_sensor)
                results["mtgp_rmse"][i, j] = mtgp_rmses_sensor[1]
                results["linear_rmse"][i, j] = linear_rmses_sensor[1]

                push!(mtgp_sensor_rmses, mtgp_rmses_sensor)
                push!(linear_sensor_rmses, linear_rmses_sensor)
            end
                    
            # Store sensor-level results
            results["mtgp_rmse_by_sensor"][i, j] = mtgp_sensor_rmses
            results["linear_rmse_by_sensor"][i, j] = linear_sensor_rmses
            
            # 存储传感器详细结果
            if include_sensor_details
                key = "noise_$(Int(noise_level*100))_missing_$(Int(missing_pct*100))"
                results["sensor_level_results"][key] = sensor_detailed_results
            end
            
            # Print summary for this combination (only for target sensors)
            
            println("  📊 Results (averaged over $(length(eval_sensor_indices)) target sensors):")
            println("    🧠 MTGP RMSE: $(round(mtgp_rmses_sensor[1] * 100, digits=4))%")
            println("    📏 Linear RMSE: $(round(linear_rmses_sensor[1] * 100, digits=4))%")
        end
    end
    
    return results
end

"""
修正后的传感器级别报告生成函数
"""
function generate_sensor_level_report_corrected(results::Dict{String, Any})
    if !haskey(results, "sensor_level_results")
        println("No sensor-level details available. Set include_sensor_details=true when running evaluation.")
        return Dict{String, Dict{String, Float64}}()
    end
    
    println("\n" * "="^80)
    println("📊 SENSOR-LEVEL DETAILED ANALYSIS")
    println("="^80)
    println("🔧 Training: MTGP used ALL $(results["training_sensor_count"]) sensors")
    println("📊 Evaluation: RMSE calculated on $(length(results["eval_sensor_indices"])) target sensors")
    
    sensor_names = results["eval_sensor_names"]
    sensor_types = results["eval_sensor_types"]
    noise_levels = results["noise_levels"]
    missing_percentages = results["missing_percentages"]
    
    # 创建传感器性能摘要
    sensor_performance_summary = Dict{String, Dict{String, Float64}}()
    
    # 为每个目标传感器生成报告
    for (idx, sensor_name) in enumerate(sensor_names)
        println("\n" * "="^60)
        println("🎯 TARGET SENSOR: $sensor_name ($(sensor_types[idx]))")
        println("="^60)
        
        sensor_summary = Dict{String, Vector{Float64}}()
        sensor_summary["mtgp_rmse"] = Float64[]
        sensor_summary["linear_rmse"] = Float64[]
        sensor_summary["improvement"] = Float64[]
        
        # 收集该传感器在所有条件下的结果
        for (i, noise_level) in enumerate(noise_levels)
            for (j, missing_pct) in enumerate(missing_percentages)
                key = "noise_$(Int(noise_level*100))_missing_$(Int(missing_pct*100))"
                if haskey(results["sensor_level_results"], key) && 
                   haskey(results["sensor_level_results"][key], sensor_name)
                    
                    sensor_data = results["sensor_level_results"][key][sensor_name]
                    if !isempty(sensor_data["mtgp_rmse"])
                        append!(sensor_summary["mtgp_rmse"], sensor_data["mtgp_rmse"])
                        append!(sensor_summary["linear_rmse"], sensor_data["linear_rmse"])
                        append!(sensor_summary["improvement"], sensor_data["improvement"])
                        
                        println("🔊 Noise $(Int(noise_level*100))% | ❌ Missing $(Int(missing_pct*100))%:")
                        println("  🧠 MTGP RMSE: $(round(mean(sensor_data["mtgp_rmse"]), digits=4)) ± $(round(std(sensor_data["mtgp_rmse"]), digits=4))")
                        println("  📏 Linear RMSE: $(round(mean(sensor_data["linear_rmse"]), digits=4)) ± $(round(std(sensor_data["linear_rmse"]), digits=4))")
                        println("  🎯 Improvement: $(round(mean(sensor_data["improvement"]), digits=2))%")
                    end
                end
            end
        end
        
        # 传感器总体统计
        if !isempty(sensor_summary["improvement"])
            avg_improvement = mean(sensor_summary["improvement"])
            best_improvement = maximum(sensor_summary["improvement"])
            worst_improvement = minimum(sensor_summary["improvement"])
            
            println("\n📈 Overall Statistics for $sensor_name:")
            println("  🧠 Average MTGP RMSE: $(round(mean(sensor_summary["mtgp_rmse"]), digits=4))")
            println("  📏 Average Linear RMSE: $(round(mean(sensor_summary["linear_rmse"]), digits=4))")
            println("  🎯 Average Improvement: $(round(avg_improvement, digits=2))%")
            println("  🏆 Best Improvement: $(round(best_improvement, digits=2))%")
            println("  📉 Worst Improvement: $(round(worst_improvement, digits=2))%")
            
            # 存储到性能摘要
            sensor_performance_summary[sensor_name] = Dict(
                "avg_improvement" => avg_improvement,
                "best_improvement" => best_improvement,
                "worst_improvement" => worst_improvement,
                "avg_mtgp_rmse" => mean(sensor_summary["mtgp_rmse"]),
                "avg_linear_rmse" => mean(sensor_summary["linear_rmse"])
            )
        end
    end
    
    # 生成排名报告
    generate_sensor_ranking_report_corrected(sensor_performance_summary, results)
    
    return sensor_performance_summary
end

"""
修正后的传感器性能排名报告
"""
function generate_sensor_ranking_report_corrected(performance_summary::Dict{String, Dict{String, Float64}}, 
                                                results::Dict{String, Any})
    println("\n" * "="^80)
    println("🏆 TARGET SENSOR PERFORMANCE RANKING")
    println("="^80)
    println("📝 Note: MTGP was trained on ALL $(results["training_sensor_count"]) sensors")
    println("📊      RMSE calculated on $(length(results["eval_sensor_indices"])) target sensors")
    
    # 按平均改进率排序
    sorted_sensors = sort(collect(performance_summary), by=x->x[2]["avg_improvement"], rev=true)
    
    println("\n🏆 Top $(min(10, length(sorted_sensors))) Target Sensors by Average MTGP Improvement:")
    println("-"^60)
    for (i, (sensor_name, stats)) in enumerate(sorted_sensors[1:min(10, end)])
        println("$i. $sensor_name")
        println("   🎯 Average Improvement: $(round(stats["avg_improvement"], digits=2))%")
        println("   🏆 Best Case: $(round(stats["best_improvement"], digits=2))%")
        println("   🧠 MTGP RMSE: $(round(stats["avg_mtgp_rmse"], digits=4))")
        println()
    end
    
    # 按传感器类型统计
    println("\n📊 Performance by Target Sensor Type:")
    println("-"^60)
    
    if haskey(results, "eval_sensor_names") && haskey(results, "eval_sensor_types")
        sensor_names = results["eval_sensor_names"]
        sensor_types = results["eval_sensor_types"]
        
        type_stats = Dict{Symbol, Vector{Float64}}()
        
        for (sensor_name, stats) in performance_summary
            # 找到传感器在目标列表中的位置
            sensor_idx = findfirst(x -> x == sensor_name, sensor_names)
            if sensor_idx !== nothing
                sensor_type = sensor_types[sensor_idx]
                
                if !haskey(type_stats, sensor_type)
                    type_stats[sensor_type] = Float64[]
                end
                push!(type_stats[sensor_type], stats["avg_improvement"])
            end
        end
        
        for (sensor_type, improvements) in type_stats
            if !isempty(improvements)
                println("🔧 $sensor_type sensors (target evaluation):")
                println("  📊 Count: $(length(improvements))")
                println("  🎯 Average Improvement: $(round(mean(improvements), digits=2))%")
                println("  🏆 Best Improvement: $(round(maximum(improvements), digits=2))%")
                println("  📉 Worst Improvement: $(round(minimum(improvements), digits=2))%")
                println()
            end
        end
    else
        println("Target sensor type information not available in results.")
    end
end

"""
修正后的目标传感器分析函数
"""
function run_targeted_sensor_analysis_corrected(data::MultiSensorData)
    println("\n" * "="^80)
    println("🎯 TARGETED SENSOR ANALYSIS - CORRECTED VERSION")
    println("="^80)
    println("🔧 Strategy: Train MTGP/Linear on ALL sensors, evaluate RMSE on target sensors")
    
    # 显示可用的传感器信息
    println("\n📋 Available sensors (ALL will be used for training):")
    for i in 1:min(20, data.S)
        println("  $i. $(data.sensor_names[i]) ($(data.sensor_types[i]))")
    end
    if data.S > 20
        println("  ... and $(data.S - 20) more sensors")
    end
    
    # 显示传感器类型统计
    type_counts = Dict{Symbol, Int}()
    for sensor_type in data.sensor_types
        type_counts[sensor_type] = get(type_counts, sensor_type, 0) + 1
    end
    println("\n📊 Sensor type distribution:")
    for (sensor_type, count) in type_counts
        println("  $sensor_type: $count sensors")
    end
    
    # 示例1: 指定传感器索引进行RMSE评估
    target_sensor_indices = [8, 9, 12, 20]
    target_sensor_indices = filter(x -> x <= data.S, target_sensor_indices)
    
    println("\n[Example 1] 🔧 Training on ALL sensors, 📊 evaluating RMSE on specific sensors:")
    println("🔧 Training sensors: ALL $(data.S) sensors")
    println("📊 Evaluation sensors: ", target_sensor_indices)
    println("🔊 Noise levels: 1% and 10% only")
    
    # 运行分析 - 使用修正后的函数
    results_indices = evaluate_noise_and_missing_data_impact_corrected(
        data;
        noise_levels = [0.01, 0.10],
        missing_percentages = [0.2, 0.4, 0.6],
        n_trials = 2,
        mtgp_epochs = 200,
        verbose = true,
        target_sensors = target_sensor_indices,
        include_sensor_details = true
    )
    
    # 示例2: 按传感器类型过滤进行RMSE评估
    available_types = collect(keys(type_counts))
    if length(available_types) >= 2
        target_sensor_types = available_types[1:2]
        
        println("\n[Example 2] 🔧 Training on ALL sensors, 📊 evaluating RMSE on sensor types:")
        println("🔧 Training sensors: ALL $(data.S) sensors")
        println("📊 Evaluation sensor types: ", target_sensor_types)
        
        results_types = evaluate_noise_and_missing_data_impact_corrected(
            data;
            noise_levels = [0.01, 0.10],
            missing_percentages = [0.3, 0.5],
            n_trials = 2,
            mtgp_epochs = 200,
            verbose = false,
            sensor_types_filter = target_sensor_types,
            include_sensor_details = true
        )
    end
    
    # 生成传感器级别报告
    println("\n[Report Generation] 📊 Creating detailed target sensor report...")
    sensor_performance = generate_sensor_level_report_corrected(results_indices)
    
    return results_indices, sensor_performance
end

"""
快速测试特定传感器的函数 - 修正版
"""
function quick_sensor_test_corrected(data::MultiSensorData, sensor_indices::Vector{Int})
    println("\n" * "="^60)
    println("⚡ QUICK SENSOR TEST - CORRECTED")
    println("="^60)
    
    # 验证传感器索引
    valid_indices = filter(x -> 1 <= x <= data.S, sensor_indices)
    println("🔧 Training: ALL $(data.S) sensors")
    println("📊 Evaluating $(length(valid_indices)) target sensors:")
    for (i, s) in enumerate(valid_indices)
        println("  $i. $(data.sensor_names[s]) ($(data.sensor_types[s]))")
    end
    
    # 运行快速测试
    results = evaluate_noise_and_missing_data_impact_corrected(
        data;
        noise_levels = [0.05],  # 单一噪声水平
        missing_percentages = [0.3],  # 单一缺失率
        n_trials = 1,  # 单次试验
        mtgp_epochs = 200,  # 较少的训练轮次
        verbose = true,
        target_sensors = valid_indices,
        include_sensor_details = true
    )
    
    return results
end

# ============================================
# Main Execution - 修正版
# ============================================

println("\n" * "="^80)
println("🚀 Starting Corrected Targeted Sensor Analysis")
println("="^80)
println("🔧 MTGP: Multi-task learning with ALL sensors")
println("📊 Evaluation: RMSE on selected target sensors")

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
    max_points_per_sensor = 800
)


println("\n[4] 📊 Optional: Running comprehensive analysis...")
comprehensive_results = evaluate_noise_and_missing_data_impact_corrected(
    data;
    noise_levels = [0.01, 0.10],  # 1% and 10% of signal standard deviation
    missing_percentages = [0.1, 0.2, 0.4, 0.6],
    target_sensors = [9],  # 指定目标传感器索引
    mtgp_epochs = 200,  # MTGP training epochs
    verbose = false,  # 减少输出
    include_sensor_details = false  # 不包含详细信息以节省内存
)

println("\n[5] 📈 Generating visualizations...")
plot_noise_missing_results(comprehensive_results)

println("\n[6] 📋 Generating comprehensive report...")
# generate_noise_missing_report(comprehensive_results)

# Save results for future analysis

println("\n" * "="^80)
println("🎉 Corrected Targeted Analysis Complete!")
println("="^80)
println("✅ MTGP trained on ALL sensors for multi-task learning")
println("✅ RMSE evaluated on selected target sensors only")
println("✅ Proper utilization of sensor correlations")
println("\n🛠️ Available functions:")
println("- evaluate_noise_and_missing_data_impact_corrected()")
println("- generate_sensor_level_report_corrected()")
println("- run_targeted_sensor_analysis_corrected()")
println("- quick_sensor_test_corrected()")

