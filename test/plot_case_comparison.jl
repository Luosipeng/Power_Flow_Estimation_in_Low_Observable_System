#############################
# 简化的MTGP vs 线性插值对比脚本
# 专门针对：1%噪声 + 60%缺失数据
#############################

# 导入必要的包和函数
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

# 导入可视化函数
include("../src/sensor_prediction_visualization.jl")

using Flux
using LinearAlgebra
using Plots
pyplot()
using Statistics
using Random
using ProgressMeter
using DataFrames
using JLD2
using MAT
using CSV
using Dates


# 设置绘图参数
default(fontfamily="Times New Roman", legendfontsize=10, guidefontsize=12, titlefontsize=14)
# default(fontfamily = "Times New Roman")

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
简化的对比分析函数
专门针对1%噪声、60%缺失数据的场景
"""
function run_simple_comparison(;
                              target_sensors::Union{Vector{Int}, Nothing} = nothing,
                              max_points_per_sensor::Int = 800,
                              mtgp_epochs::Int = 200,
                              time_step_minutes::Float64 = 1.0/60,
                              output_dir::String = "./simple_comparison_results/",
                              save_results::Bool = true)
    
    # 固定的分析参数
    NOISE_LEVEL = 0.01      # 1% 噪声
    MISSING_PERCENTAGE = 0.6 # 60% 缺失数据
    
    println("\n" * "="^80)
    println("🎯 SIMPLE MTGP vs LINEAR INTERPOLATION COMPARISON")
    println("="^80)
    println("📊 Analysis Conditions:")
    println("   🔊 Noise Level: $(NOISE_LEVEL*100)%")
    println("   ❌ Missing Data: $(MISSING_PERCENTAGE*100)%")
    println("   🧠 MTGP Epochs: $mtgp_epochs")
    println("   ⏱️  Time Step: $time_step_minutes minutes")
    println("="^80)
    
    # 创建输出目录
    if save_results
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        output_dir = joinpath(output_dir, "comparison_$timestamp")
        mkpath(output_dir)
        println("📁 Results will be saved to: $output_dir")
    end
    
    # ============================================
    # 第1步：数据加载
    # ============================================
    println("\n[1] 📂 Loading data...")
    
    # 读取数据
    (batch_data_1, batch_data_2, batch_data_3, batch_data_4,
     batch_data_5, batch_data_6, batch_data_7, batch_data_8,
     batch_data_9, batch_data_10, batch_data_11, batch_data_12,
     batch_data_13, batch_data_14, batch_data_15, batch_data_16, 
     batch_data_17, batch_data_18) = read_mat()
    
    # 提取数据集
    ds = extract_requested_dataset_multibatch(
        (batch_data_1, batch_data_2, batch_data_3, batch_data_4,
         batch_data_5, batch_data_6, batch_data_7, batch_data_8,
         batch_data_9, batch_data_10, batch_data_11, batch_data_12,
         batch_data_13, batch_data_14, batch_data_15, batch_data_16,
         batch_data_17, batch_data_18)
    )
    
    # 构建完整数据
    data = build_complete_multisensor_data(
        ds;
        max_points_per_sensor = max_points_per_sensor
    )
    
    println("✅ Data loaded successfully!")
    println("   📊 Total sensors: $(data.S)")
    println("   📏 Max points per sensor: $max_points_per_sensor")
    
    # ============================================
    # 第2步：选择目标传感器
    # ============================================
    println("\n[2] 🎯 Selecting target sensors...")
    
    if target_sensors === nothing
        # 自动选择一些有代表性的传感器
        target_sensors = [1, 2, 3, 4, 5, 6]  # 前6个传感器
        target_sensors = filter(x -> x <= data.S, target_sensors)
    else
        # 验证用户指定的传感器
        target_sensors = filter(x -> 1 <= x <= data.S, target_sensors)
    end
    
    println("   🎯 Selected sensors for analysis:")
    for (i, sensor_idx) in enumerate(target_sensors)
        sensor_name = length(data.sensor_names) >= sensor_idx ? data.sensor_names[sensor_idx] : "Sensor_$sensor_idx"
        sensor_type = length(data.sensor_types) >= sensor_idx ? data.sensor_types[sensor_idx] : :unknown
        println("      $i. Sensor $sensor_idx: $sensor_name ($sensor_type)")
    end
    
    # ============================================
    # 第3步：运行对比分析
    # ============================================
    println("\n[3] 🔬 Running comparison analysis...")
    println("   ⏳ This may take a few minutes...")
    
    # 使用可视化函数进行对比分析
    comparison_results = train_and_visualize_sensor_predictions(
        data;
        target_sensor_indices = target_sensors,
        noise_level = NOISE_LEVEL,
        missing_percentage = MISSING_PERCENTAGE,
        mtgp_epochs = mtgp_epochs,
        time_step_minutes = time_step_minutes,
        save_results = save_results,
        output_dir = output_dir
    )
    
    # ============================================
    # 第4步：汇总结果
    # ============================================
    println("\n[4] 📊 Summarizing results...")
    
    # 收集所有传感器的结果
    sensor_results = []
    total_mtgp_rmse = 0.0
    total_linear_rmse = 0.0
    valid_sensors = 0
    
    for (key, result) in comparison_results
        if startswith(key, "sensor_") && haskey(result, "mtgp_rmse")
            sensor_name = result["sensor_name"]
            mtgp_rmse = result["mtgp_rmse"]
            linear_rmse = result["linear_rmse"]
            improvement = result["improvement_percent"]
            
            push!(sensor_results, (
                name = sensor_name,
                mtgp_rmse = mtgp_rmse,
                linear_rmse = linear_rmse,
                improvement = improvement
            ))
            
            total_mtgp_rmse += mtgp_rmse
            total_linear_rmse += linear_rmse
            valid_sensors += 1
            
            println("   📈 $sensor_name:")
            println("      🧠 MTGP RMSE: $(round(mtgp_rmse, digits=4))")
            println("      📏 Linear RMSE: $(round(linear_rmse, digits=4))")
            println("      🎯 Improvement: $(round(improvement, digits=2))%")
            println()
        end
    end
    
    # 计算平均性能
    if valid_sensors > 0
        avg_mtgp_rmse = total_mtgp_rmse / valid_sensors
        avg_linear_rmse = total_linear_rmse / valid_sensors
        avg_improvement = (avg_linear_rmse - avg_mtgp_rmse) / avg_linear_rmse * 100
        
        println("\n" * "="^60)
        println("📊 OVERALL COMPARISON RESULTS")
        println("="^60)
        println("🧠 Average MTGP RMSE: $(round(avg_mtgp_rmse, digits=4))")
        println("📏 Average Linear RMSE: $(round(avg_linear_rmse, digits=4))")
        println("🎯 Average MTGP Improvement: $(round(avg_improvement, digits=2))%")
        println("📊 Number of sensors analyzed: $valid_sensors")
        println("="^60)
    end
    
    # ============================================
    # 第5步：保存汇总报告
    # ============================================
    if save_results
        println("\n[5] 💾 Saving summary report...")
        
        # 保存CSV汇总
        csv_path = joinpath(output_dir, "comparison_summary.csv")
        df = DataFrame(sensor_results)
        CSV.write(csv_path, df)
        
        # 保存文本报告
        report_path = joinpath(output_dir, "comparison_report.txt")
        open(report_path, "w") do f
            write(f, "MTGP vs Linear Interpolation Comparison Report\n")
            write(f, "="^50 * "\n")
            write(f, "Analysis Date: $(now())\n")
            write(f, "Noise Level: $(NOISE_LEVEL*100)%\n")
            write(f, "Missing Data: $(MISSING_PERCENTAGE*100)%\n")
            write(f, "MTGP Epochs: $mtgp_epochs\n")
            write(f, "Time Step: $time_step_minutes minutes\n\n")
            
            write(f, "Individual Sensor Results:\n")
            write(f, "-"^30 * "\n")
            for result in sensor_results
                write(f, "$(result.name):\n")
                write(f, "  MTGP RMSE: $(round(result.mtgp_rmse, digits=4))\n")
                write(f, "  Linear RMSE: $(round(result.linear_rmse, digits=4))\n")
                write(f, "  Improvement: $(round(result.improvement, digits=2))%\n\n")
            end
            
            if valid_sensors > 0
                write(f, "Overall Summary:\n")
                write(f, "-"^20 * "\n")
                write(f, "Average MTGP RMSE: $(round(avg_mtgp_rmse, digits=4))\n")
                write(f, "Average Linear RMSE: $(round(avg_linear_rmse, digits=4))\n")
                write(f, "Average Improvement: $(round(avg_improvement, digits=2))%\n")
                write(f, "Sensors Analyzed: $valid_sensors\n")
            end
        end
        
        println("   💾 CSV summary saved: $csv_path")
        println("   💾 Text report saved: $report_path")
        
        # 保存完整结果
        results_path = joinpath(output_dir, "complete_results.jld2")
        JLD2.save(results_path,
                 "parameters", Dict(
                     "noise_level" => NOISE_LEVEL,
                     "missing_percentage" => MISSING_PERCENTAGE,
                     "mtgp_epochs" => mtgp_epochs,
                     "target_sensors" => target_sensors,
                     "time_step_minutes" => time_step_minutes
                 ),
                 "data_info", Dict(
                     "total_sensors" => data.S,
                     "sensor_names" => data.sensor_names,
                     "sensor_types" => data.sensor_types
                 ),
                 "results", comparison_results,
                 "summary", Dict(
                     "sensor_results" => sensor_results,
                     "avg_mtgp_rmse" => valid_sensors > 0 ? avg_mtgp_rmse : NaN,
                     "avg_linear_rmse" => valid_sensors > 0 ? avg_linear_rmse : NaN,
                     "avg_improvement" => valid_sensors > 0 ? avg_improvement : NaN,
                     "valid_sensors" => valid_sensors
                 ))
        
        println("   💾 Complete results saved: $results_path")
    end
    
    # ============================================
    # 第6步：最终总结
    # ============================================
    println("\n" * "="^80)
    println("🎉 COMPARISON ANALYSIS COMPLETED!")
    println("="^80)
    println("📅 Completion time: $(now())")
    if save_results
        println("📁 All results saved in: $output_dir")
        println("📊 Key files:")
        println("   🖼️  Individual sensor plots: $(output_dir)/sensor_*_prediction.png")
        println("   🎨 Combined plot: $(output_dir)/combined_sensor_predictions.png")
        println("   📋 Summary CSV: $(output_dir)/comparison_summary.csv")
        println("   📄 Text report: $(output_dir)/comparison_report.txt")
    end
    println("="^80)
    
    return comparison_results, sensor_results, data
end

# ============================================
# 快速启动函数
# ============================================

"""
自定义传感器分析
"""
function run_custom_sensors(sensor_indices::Vector{Int})
    return run_simple_comparison(
        target_sensors = sensor_indices,
        mtgp_epochs = 200,
        time_step_minutes = 1.0/60
    )
end


# ============================================
# 脚本执行入口
# ============================================

    println("\n🎯 Simple MTGP vs Linear Interpolation Comparison")
    println("📊 Conditions: 1% Noise + 60% Missing Data")
    sensor_indices = [14]
    println("🎯 分析传感器: $sensor_indices")
    results, summary, data = run_custom_sensors(sensor_indices)  
    println("\n✅ 分析完成！")


