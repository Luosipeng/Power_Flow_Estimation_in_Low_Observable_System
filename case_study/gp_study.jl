#############################
# Multi-task Gaussian Process - Corrected Targeted Sensor Analysis
# 训练：使用所有传感器  |  评估：只计算指定传感器的RMSE
# 修复版：包含动态随机种子和多轮试验平均
#############################
include("../src/implement_data.jl")
include("../ios/read_mat.jl")
include("../src/extract_requested_dataset_multibatch.jl")
include("../src/build_complete_multisensor_data.jl")
include("../src/data_processing.jl")
include("../src/multi_task_gaussian.jl")
include("../src/gaussian_prediction.jl")
include("../src/linear_imputation.jl")
include("../src/missing_data_evaluation.jl")
# include("../data/sensor_location.jl")
include("../data/sensor_location_123.jl")

using Flux
using LinearAlgebra
using Statistics
using Random
using ProgressMeter
using DataFrames
using JLD2
using MAT
using CSV

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

"""
修正后的评估函数 - MTGP用所有传感器训练，但只评估指定传感器

关键修正：
1. MTGP训练：使用所有传感器数据进行多任务学习
2. 线性插值训练：使用所有传感器数据
3. RMSE评估：只计算指定目标传感器的RMSE
4. 数据处理：对所有传感器添加噪声和缺失数据
5. 【NEW】多轮试验平均：消除随机性，使用动态种子
"""
function evaluate_noise_and_missing_data_impact_corrected(data::MultiSensorData; 
                                              noise_levels::Vector{Float64}=[0.01, 0.10],
                                              missing_percentages::Vector{Float64}=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                              n_trials::Int=10,  # 默认进行10轮试验取平均
                                              mtgp_epochs::Int=200,
                                              verbose::Bool=true,
                                              target_sensors::Union{Vector{Int}, Vector{String}, Nothing}=nothing,
                                              sensor_types_filter::Union{Vector{Symbol}, Nothing}=nothing,
                                              include_sensor_details::Bool=true)
    
    # 确定要评估RMSE的传感器（不影响训练）
    if target_sensors !== nothing
        eval_sensor_indices = target_sensors
    elseif sensor_types_filter !== nothing
        # 根据传感器类型过滤
        eval_sensor_indices = findall(x -> x in sensor_types_filter, data.sensor_types)
    else
        eval_sensor_indices = collect(1:data.S)
    end
    # 确保索引有效
    eval_sensor_indices = filter(x -> 1 <= x <= data.S, eval_sensor_indices)
    
    if verbose
        println(" TRAINING STRATEGY:")
        println("   MTGP: Uses ALL $(data.S) sensors for multi-task learning")
        println("   Linear: Uses ALL $(data.S) sensors for interpolation")
        println(" EVALUATION STRATEGY:")
        println("   RMSE calculated on $(length(eval_sensor_indices)) target sensors")
        println("   Averaging over $n_trials independent trials per configuration")
        
        for (idx_show, s) in enumerate(eval_sensor_indices[1:min(5, end)])
            println("     $idx_show. $(data.sensor_names[s]) ($(data.sensor_types[s]))")
        end
        if length(eval_sensor_indices) > 5
            println("     ... and $(length(eval_sensor_indices) - 5) more sensors")
        end
    end
    
    results = Dict{String, Any}()
    results["noise_levels"] = noise_levels
    results["missing_percentages"] = missing_percentages
    results["eval_sensor_names"] = data.sensor_names[eval_sensor_indices]
    results["eval_sensor_indices"] = eval_sensor_indices
    results["eval_sensor_types"] = data.sensor_types[eval_sensor_indices]
    results["all_sensor_names"] = data.sensor_names
    results["training_sensor_count"] = data.S
    results["n_trials"] = n_trials
    
    # Initialize result arrays
    n_noise = length(noise_levels)
    n_missing = length(missing_percentages)

    results["mtgp_rmse"] = zeros(Float64, n_noise, n_missing)
    results["linear_rmse"] = zeros(Float64, n_noise, n_missing)
    results["mtgp_rmse_std"] = zeros(Float64, n_noise, n_missing) # 存储标准差
    results["linear_rmse_std"] = zeros(Float64, n_noise, n_missing)
    
    # 存储结构的初始化略微复杂，因为现在包含了多次试验
    results["mtgp_rmse_by_sensor"] = Array{Vector{Vector{Float64}}}(undef, n_noise, n_missing)
    results["linear_rmse_by_sensor"] = Array{Vector{Vector{Float64}}}(undef, n_noise, n_missing)
    
    if include_sensor_details
        results["sensor_level_results"] = Dict{String, Any}()
    end
    
    if verbose
        println("\n" * "="^80)
        println("CORRECTED NOISE & MISSING DATA EVALUATION: MTGP vs Linear")
        println("="^80)
    end
    
    for (i, noise_level) in enumerate(noise_levels)
        if verbose
            println("\n" * "="^60)
            println(" NOISE LEVEL: $(Int(noise_level*100))% of signal standard deviation")
            println("="^60)
        end
        
        # Add noise to ALL sensor data
        # 注意：这里噪声是固定的，但下面的缺失掩码是随机的
        noisy_data = add_gaussian_noise(data, noise_level, seed=42)
        
        for (j, missing_pct) in enumerate(missing_percentages)
            println("[Noise $(Int(noise_level*100))% | Missing $(Int(missing_pct*100))%] $n_trials trials × $(Threads.nthreads()) threads")
            
            # === 预生成所有trial的缺失数据（顺序执行，确保RNG安全） ===
            trial_missing_data = Vector{Tuple{MultiSensorData, Vector{Vector{Float32}}, Vector{Vector{Float32}}}}(undef, n_trials)
            for t in 1:n_trials
                current_seed = 42 + t + (i * 100) + (j * 1000)
                trial_missing_data[t] = create_missing_data(noisy_data, missing_pct, seed=current_seed)
            end
            
            # 预分配线程安全的结果数组（每个trial一个slot）
            TrialResult = NamedTuple{(:mtgp_rmse, :linear_rmse, :mtgp_sensor, :linear_sensor, :sensor_details),
                Tuple{Float64, Float64, Vector{Float64}, Vector{Float64}, Dict{String, Dict{String, Any}}}}
            trial_results = Vector{Union{Nothing, TrialResult}}(nothing, n_trials)
            
            # === 多线程并行试验 ===
            completed = Threads.Atomic{Int}(0)
            Threads.@threads for t in 1:n_trials
                missing_noisy_data, removed_times, removed_values = trial_missing_data[t]
                
                # Train MTGP (verbose=false to avoid spam)
                mtgp_result = train_icm_mtgp(missing_noisy_data; num_epochs=mtgp_epochs, lr=0.01, verbose=false)
                
                # Train Linear
                linear_result = train_linear_interpolation(missing_noisy_data; verbose=false)
                
                # Build mapping from original sensor index to active sensor index
                orig_to_active_mtgp = Dict(orig => active for (active, orig) in enumerate(mtgp_result.active_indices))
                orig_to_active_linear = Dict(orig => active for (active, orig) in enumerate(linear_result.active_indices))
                
                # Evaluate ONLY on target sensors
                current_trial_mtgp_sensor_rmses = Float64[]
                current_trial_linear_sensor_rmses = Float64[]
                local_sensor_details = Dict{String, Dict{String, Any}}()
                
                for s in eval_sensor_indices
                    # Skip zero-load sensors and sensors not in active set
                    if !haskey(orig_to_active_mtgp, s) || !haskey(orig_to_active_linear, s)
                        continue
                    end
                    s_mtgp = orig_to_active_mtgp[s]
                    s_linear = orig_to_active_linear[s]
                    
                    if !isempty(removed_times[s])
                        # Get original clean values
                        original_clean_values = Float32[]
                        valid_removed_indices = Int[] # Track which removed points we actually found
                        
                        for (k, time_point) in enumerate(removed_times[s])
                            time_diffs = abs.(data.times[s] .- time_point)
                            closest_idx = argmin(time_diffs)
                            if time_diffs[closest_idx] < 0.01
                                push!(original_clean_values, data.values[s][closest_idx])
                                push!(valid_removed_indices, k)
                            end
                        end
                        
                        if !isempty(original_clean_values)
                            # Predict using active sensor indices
                            query_times = removed_times[s][valid_removed_indices]
                            
                            μ_mtgp, _ = icm_predict(mtgp_result, s_mtgp, query_times)
                            mtgp_rmse = sqrt(mean((μ_mtgp .- original_clean_values).^2))
                            
                            μ_linear, _ = linear_predict(linear_result, s_linear, query_times)
                            linear_rmse = sqrt(mean((μ_linear .- original_clean_values).^2))
                            
                            push!(current_trial_mtgp_sensor_rmses, mtgp_rmse)
                            push!(current_trial_linear_sensor_rmses, linear_rmse)
                            
                            # 收集详细数据
                            if include_sensor_details
                                sensor_name = data.sensor_names[s]
                                if !haskey(local_sensor_details, sensor_name)
                                    local_sensor_details[sensor_name] = Dict{String, Any}(
                                        "mtgp_rmse" => Float64[],
                                        "linear_rmse" => Float64[],
                                        "improvement" => Float64[],
                                        "sensor_type" => data.sensor_types[s]
                                    )
                                end
                                push!(local_sensor_details[sensor_name]["mtgp_rmse"], mtgp_rmse)
                                push!(local_sensor_details[sensor_name]["linear_rmse"], linear_rmse)
                                improvement = (linear_rmse - mtgp_rmse) / linear_rmse * 100
                                push!(local_sensor_details[sensor_name]["improvement"], improvement)
                            end
                        end
                    end
                end
                
                # 计算本次试验所有目标传感器的平均RMSE
                if !isempty(current_trial_mtgp_sensor_rmses)
                    trial_results[t] = (
                        mtgp_rmse = mean(current_trial_mtgp_sensor_rmses),
                        linear_rmse = mean(current_trial_linear_sensor_rmses),
                        mtgp_sensor = current_trial_mtgp_sensor_rmses,
                        linear_sensor = current_trial_linear_sensor_rmses,
                        sensor_details = local_sensor_details
                    )
                end
                
                Threads.atomic_add!(completed, 1)
                done = completed[]
                print("\r  Progress: $done / $n_trials trials completed")
            end # End trials loop
            println()  # 换行
            
            # === 汇总所有trial结果（单线程） ===
            trial_mtgp_rmses = Float64[]
            trial_linear_rmses = Float64[]
            trial_mtgp_sensor_details = []
            trial_linear_sensor_details = []
            sensor_detailed_results = Dict{String, Dict{String, Any}}()
            
            for t_agg in 1:n_trials
                r = trial_results[t_agg]
                r === nothing && continue
                push!(trial_mtgp_rmses, r.mtgp_rmse)
                push!(trial_linear_rmses, r.linear_rmse)
                push!(trial_mtgp_sensor_details, r.mtgp_sensor)
                push!(trial_linear_sensor_details, r.linear_sensor)
                
                if include_sensor_details
                    for (sname, sdata) in r.sensor_details
                        if !haskey(sensor_detailed_results, sname)
                            sensor_detailed_results[sname] = Dict{String, Any}(
                                "mtgp_rmse" => Float64[],
                                "linear_rmse" => Float64[],
                                "improvement" => Float64[],
                                "sensor_type" => sdata["sensor_type"]
                            )
                        end
                        append!(sensor_detailed_results[sname]["mtgp_rmse"], sdata["mtgp_rmse"])
                        append!(sensor_detailed_results[sname]["linear_rmse"], sdata["linear_rmse"])
                        append!(sensor_detailed_results[sname]["improvement"], sdata["improvement"])
                    end
                end
            end
            
            # Store averaged results
            if !isempty(trial_mtgp_rmses)
                avg_mtgp = mean(trial_mtgp_rmses)
                avg_linear = mean(trial_linear_rmses)
                std_mtgp = std(trial_mtgp_rmses)
                std_linear = std(trial_linear_rmses)
                
                results["mtgp_rmse"][i, j] = avg_mtgp
                results["linear_rmse"][i, j] = avg_linear
                results["mtgp_rmse_std"][i, j] = std_mtgp
                results["linear_rmse_std"][i, j] = std_linear
                
                results["mtgp_rmse_by_sensor"][i, j] = trial_mtgp_sensor_details
                results["linear_rmse_by_sensor"][i, j] = trial_linear_sensor_details
                
                println("  → MTGP  RMSE: $(round(avg_mtgp * 100, digits=4))% ± $(round(std_mtgp*100, digits=4))%")
                println("  → Linear RMSE: $(round(avg_linear * 100, digits=4))% ± $(round(std_linear*100, digits=4))%")
            end
            
            if include_sensor_details
                key = "noise_$(Int(noise_level*100))_missing_$(Int(missing_pct*100))"
                results["sensor_level_results"][key] = sensor_detailed_results
            end
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
    println(" SENSOR-LEVEL DETAILED ANALYSIS")
    println("="^80)
    println(" Training: MTGP used ALL $(results["training_sensor_count"]) sensors")
    println(" Evaluation: RMSE calculated on $(length(results["eval_sensor_indices"])) target sensors")
    
    sensor_names = results["eval_sensor_names"]
    sensor_types = results["eval_sensor_types"]
    noise_levels = results["noise_levels"]
    missing_percentages = results["missing_percentages"]
    
    # 创建传感器性能摘要
    sensor_performance_summary = Dict{String, Dict{String, Float64}}()
    
    # 为每个目标传感器生成报告
    for (idx, sensor_name) in enumerate(sensor_names)
        println("\n" * "="^60)
        println(" TARGET SENSOR: $sensor_name ($(sensor_types[idx]))")
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
                        # 这里 sensor_data["mtgp_rmse"] 已经是该条件下所有 trials 的列表了
                        append!(sensor_summary["mtgp_rmse"], sensor_data["mtgp_rmse"])
                        append!(sensor_summary["linear_rmse"], sensor_data["linear_rmse"])
                        append!(sensor_summary["improvement"], sensor_data["improvement"])
                        
                        println(" Noise $(Int(noise_level*100))% | ❌ Missing $(Int(missing_pct*100))%:")
                        println("  易 MTGP RMSE: $(round(mean(sensor_data["mtgp_rmse"]), digits=4)) ± $(round(std(sensor_data["mtgp_rmse"]), digits=4))")
                        println("   Linear RMSE: $(round(mean(sensor_data["linear_rmse"]), digits=4)) ± $(round(std(sensor_data["linear_rmse"]), digits=4))")
                        println("   Improvement: $(round(mean(sensor_data["improvement"]), digits=2))%")
                    end
                end
            end
        end
        
        # 传感器总体统计
        if !isempty(sensor_summary["improvement"])
            avg_improvement = mean(sensor_summary["improvement"])
            best_improvement = maximum(sensor_summary["improvement"])
            worst_improvement = minimum(sensor_summary["improvement"])
            
            println("\n Overall Statistics for $sensor_name:")
            println("  易 Average MTGP RMSE: $(round(mean(sensor_summary["mtgp_rmse"]), digits=4))")
            println("   Average Linear RMSE: $(round(mean(sensor_summary["linear_rmse"]), digits=4))")
            println("   Average Improvement: $(round(avg_improvement, digits=2))%")
            println("   Best Improvement: $(round(best_improvement, digits=2))%")
            println("   Worst Improvement: $(round(worst_improvement, digits=2))%")
            
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
    println(" TARGET SENSOR PERFORMANCE RANKING")
    println("="^80)
    println(" Note: MTGP was trained on ALL $(results["training_sensor_count"]) sensors")
    println("      RMSE calculated on $(length(results["eval_sensor_indices"])) target sensors")
    
    # 按平均改进率排序
    sorted_sensors = sort(collect(performance_summary), by=x->x[2]["avg_improvement"], rev=true)
    
    println("\n Top $(min(10, length(sorted_sensors))) Target Sensors by Average MTGP Improvement:")
    println("-"^60)
    for (i, (sensor_name, stats)) in enumerate(sorted_sensors[1:min(10, end)])
        println("$i. $sensor_name")
        println("    Average Improvement: $(round(stats["avg_improvement"], digits=2))%")
        println("    Best Case: $(round(stats["best_improvement"], digits=2))%")
        println("   易 MTGP RMSE: $(round(stats["avg_mtgp_rmse"], digits=4))")
        println()
    end
    
    # 按传感器类型统计
    println("\n Performance by Target Sensor Type:")
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
                println(" $sensor_type sensors (target evaluation):")
                println("   Count: $(length(improvements))")
                println("   Average Improvement: $(round(mean(improvements), digits=2))%")
                println("   Best Improvement: $(round(maximum(improvements), digits=2))%")
                println("   Worst Improvement: $(round(minimum(improvements), digits=2))%")
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
    println(" TARGETED SENSOR ANALYSIS - CORRECTED VERSION")
    println("="^80)
    println(" Strategy: Train MTGP/Linear on ALL sensors, evaluate RMSE on target sensors")
    
    # 显示可用的传感器信息
    println("\n Available sensors (ALL will be used for training):")
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
    println("\n Sensor type distribution:")
    for (sensor_type, count) in type_counts
        println("  $sensor_type: $count sensors")
    end
    
    # 示例1: 指定传感器索引进行RMSE评估
    target_sensor_indices = [8, 9, 12, 20]
    target_sensor_indices = filter(x -> x <= data.S, target_sensor_indices)
    
    println("\n[Example 1]  Training on ALL sensors,  evaluating RMSE on specific sensors:")
    println(" Training sensors: ALL $(data.S) sensors")
    println(" Evaluation sensors: ", target_sensor_indices)
    println(" Noise levels: 1% and 10% only")
    
    # 运行分析 - 使用修正后的函数
    results_indices = evaluate_noise_and_missing_data_impact_corrected(
        data;
        noise_levels = [0.01, 0.10],
        missing_percentages = [0.2, 0.4, 0.6],
        n_trials = 5, # 增加试验次数
        mtgp_epochs = 200,
        verbose = true,
        target_sensors = target_sensor_indices,
        include_sensor_details = true
    )
    
    # 示例2: 按传感器类型过滤进行RMSE评估
    available_types = collect(keys(type_counts))
    if length(available_types) >= 2
        target_sensor_types = available_types[1:2]
        
        println("\n[Example 2]  Training on ALL sensors,  evaluating RMSE on sensor types:")
        println(" Training sensors: ALL $(data.S) sensors")
        println(" Evaluation sensor types: ", target_sensor_types)
        
        results_types = evaluate_noise_and_missing_data_impact_corrected(
            data;
            noise_levels = [0.01, 0.10],
            missing_percentages = [0.3, 0.5],
            n_trials = 5, # 增加试验次数
            mtgp_epochs = 200,
            verbose = false,
            sensor_types_filter = target_sensor_types,
            include_sensor_details = true
        )
    end
    
    # 生成传感器级别报告
    println("\n[Report Generation]  Creating detailed target sensor report...")
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
    println(" Training: ALL $(data.S) sensors")
    println(" Evaluating $(length(valid_indices)) target sensors:")
    for (i, s) in enumerate(valid_indices)
        println("  $i. $(data.sensor_names[s]) ($(data.sensor_types[s]))")
    end
    
    # 运行快速测试
    results = evaluate_noise_and_missing_data_impact_corrected(
        data;
        noise_levels = [0.05],  # 单一噪声水平
        missing_percentages = [0.3],  # 单一缺失率
        n_trials = 5,  # 至少跑5次取平均
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

# === 手动设置线程数 ===
# Julia 的 Threads.nthreads() 在启动后不可更改，需要通过环境变量在启动前设置。
# 如果当前线程数为 1，则设置 JULIA_NUM_THREADS 并提示用户重启。
if Threads.nthreads() == 1
    desired_threads = min(Sys.CPU_THREADS, 8)  # 使用 CPU 核心数，上限为 8
    ENV["JULIA_NUM_THREADS"] = string(desired_threads)
    @warn """当前 Julia 仅使用 1 个线程！
    已设置 ENV[\"JULIA_NUM_THREADS\"] = $desired_threads,但需要重启 Julia 才能生效。
    请使用以下方式启动：
      julia -t $desired_threads case_study/gp_study.jl
    或设置环境变量后重启：
      \$env:JULIA_NUM_THREADS = $desired_threads   # PowerShell
      export JULIA_NUM_THREADS=$desired_threads     # Linux/macOS
    """
end
println(" Julia threads: $(Threads.nthreads()) / $(Sys.CPU_THREADS) CPU cores")

println("\n" * "="^80)
println(" Starting Corrected Targeted Sensor Analysis")
println("="^80)
println(" MTGP: Multi-task learning with ALL sensors")
println(" Evaluation: RMSE on selected target sensors")

pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases, pmu_sensors, scada_sensors, ami_sensors = FAD10_config_ieee123()

println("\n[1]  Loading data...")
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

println("\n[2] ️ Building complete multi-sensor dataset...")
data = build_complete_multisensor_data(
    ds;
    max_points_per_sensor = 200,
    pmu_sensors, scada_sensors, ami_sensors
)


println("\n[4]  Optional: Running comprehensive analysis...")
# 注意：这里设置 n_trials = 10，运行时间会比之前长10倍，但结果会非常漂亮
comprehensive_results = evaluate_noise_and_missing_data_impact_corrected(
    data;
    noise_levels = [0.01, 0.05],  # 1% and 5% of signal standard deviation
    missing_percentages = [0.1, 0.2, 0.4, 0.6],
    target_sensors = [104],  # 指定目标传感器索引
    n_trials = 5,          # <--- 关键：设置试验次数
    mtgp_epochs = 100,      # MTGP training epochs
    verbose = false,        # 减少输出
    include_sensor_details = false  # 不包含详细信息以节省内存
)

println("\n[5]  Generating visualizations...")
# plot_noise_missing_results(comprehensive_results) # 确保你有这个函数的定义

println("\n[6]  Generating comprehensive report...")
# generate_noise_missing_report(comprehensive_results)

# Save results for future analysis

println("\n" * "="^80)
println(" Corrected Targeted Analysis Complete!")
println("="^80)
println("✅ MTGP trained on ALL sensors for multi-task learning")
println("✅ RMSE evaluated on selected target sensors only")
println("✅ Proper utilization of sensor correlations")
println("✅ Averaged over multiple trials with dynamic seeds")
println("\n️ Available functions:")
println("- evaluate_noise_and_missing_data_impact_corrected()")
println("- generate_sensor_level_report_corrected()")
println("- run_targeted_sensor_analysis_corrected()")
println("- quick_sensor_test_corrected()")
