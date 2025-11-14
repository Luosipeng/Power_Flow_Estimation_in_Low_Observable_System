# Missing Data Evaluation Framework for MTGP vs Linear Interpolation
using Random
using Statistics
using LinearAlgebra
using Plots
using StatsBase

"""
Create missing data by randomly removing a percentage of observations from each sensor
"""
function create_missing_data(data::MultiSensorData, missing_percentage::Float64; seed::Int=42)
  Random.seed!(seed)
  
  # Create a copy of the original data - 修复构造函数参数顺序
  missing_data = MultiSensorData(
      data.S,                    # 传感器数量应该是第一个参数
      copy.(data.times),         # 时间数据
      copy.(data.values),        # 值数据
      copy(data.sensor_names),   # 传感器名称
      copy(data.sensor_types)    # 传感器类型
  )
  
  # Store removed data for evaluation
  removed_times = Vector{Vector{Float32}}(undef, data.S)
  removed_values = Vector{Vector{Float32}}(undef, data.S)
  
  for s in 1:data.S
      n_points = length(data.times[s])
      n_remove = Int(round(n_points * missing_percentage))
      
      if n_remove >= n_points
          # If we're removing all points, keep at least 2 for interpolation
          n_remove = max(0, n_points - 2)
      end
      
      if n_remove > 0
          # Randomly select indices to remove
          remove_indices = sample(1:n_points, n_remove, replace=false)
          keep_indices = setdiff(1:n_points, remove_indices)
          
          # Store removed data
          removed_times[s] = data.times[s][remove_indices]
          removed_values[s] = data.values[s][remove_indices]
          
          # Update missing_data with remaining points
          missing_data.times[s] = data.times[s][keep_indices]
          missing_data.values[s] = data.values[s][keep_indices]
      else
          # No data removed
          removed_times[s] = Float32[]
          removed_values[s] = Float32[]
      end
  end
  
  return missing_data, removed_times, removed_values
end

"""
Comprehensive evaluation of both methods across different missing data percentages
"""
function evaluate_missing_data_impact(data::MultiSensorData; 
                                  missing_percentages::Vector{Float64}=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                  n_trials::Int=5,
                                  mtgp_epochs::Int=100,
                                  verbose::Bool=true)
  
  results = Dict{String, Any}()
  results["missing_percentages"] = missing_percentages
  results["mtgp_rmse"] = zeros(Float64, length(missing_percentages), n_trials)
  results["linear_rmse"] = zeros(Float64, length(missing_percentages), n_trials)
  results["mtgp_rmse_by_sensor"] = []
  results["linear_rmse_by_sensor"] = []
  results["sensor_names"] = data.sensor_names
  
  println("\n" * "="^70)
  println("Evaluating Missing Data Impact: MTGP vs Linear Interpolation")
  println("="^70)
  println("Missing percentages: ", missing_percentages)
  println("Number of trials per percentage: ", n_trials)
  println("Total experiments: ", length(missing_percentages) * n_trials)
  
  for (i, missing_pct) in enumerate(missing_percentages)
      println("\n[Missing $(Int(missing_pct*100))%] Running $n_trials trials...")
      
      mtgp_rmses_trial = Float64[]
      linear_rmses_trial = Float64[]
      mtgp_sensor_rmses = []
      linear_sensor_rmses = []
      
      for trial in 1:n_trials
          if verbose
              println("  Trial $trial/$n_trials")
          end
          
          # Create missing data
          missing_data, removed_times, removed_values = create_missing_data(data, missing_pct, seed=42+trial)
          
          # Train both methods
          try
              # MTGP
              if verbose
                  println("    Training MTGP...")
              end
              mtgp_result = train_icm_mtgp(missing_data; num_epochs=mtgp_epochs, lr=0.01, verbose=false)
              
              # Linear Interpolation
              if verbose
                  println("    Training Linear Interpolation...")
              end
              linear_result = train_linear_interpolation(missing_data; verbose=false)
              
              # Evaluate on removed data
              mtgp_rmses_sensor = Float64[]
              linear_rmses_sensor = Float64[]
              
              for s in 1:data.S
                  if !isempty(removed_times[s])
                      # MTGP predictions
                      μ_mtgp, σ_mtgp = icm_predict(mtgp_result, s, removed_times[s])
                      mtgp_rmse = sqrt(mean((μ_mtgp .- removed_values[s]).^2))
                      
                      # Linear predictions
                      μ_linear, σ_linear = linear_predict(linear_result, s, removed_times[s])
                      linear_rmse = sqrt(mean((μ_linear .- removed_values[s]).^2))
                      
                      push!(mtgp_rmses_sensor, mtgp_rmse)
                      push!(linear_rmses_sensor, linear_rmse)
                  end
              end
              
              # Store results
              if !isempty(mtgp_rmses_sensor)
                  results["mtgp_rmse"][i, trial] = mean(mtgp_rmses_sensor)
                  results["linear_rmse"][i, trial] = mean(linear_rmses_sensor)
                  push!(mtgp_rmses_trial, mean(mtgp_rmses_sensor))
                  push!(linear_rmses_trial, mean(linear_rmses_sensor))
                  push!(mtgp_sensor_rmses, mtgp_rmses_sensor)
                  push!(linear_sensor_rmses, linear_rmses_sensor)
              end
              
          catch e
              println("    Error in trial $trial: $e")
              results["mtgp_rmse"][i, trial] = NaN
              results["linear_rmse"][i, trial] = NaN
          end
      end
      
      # Print summary for this missing percentage
      if !isempty(mtgp_rmses_trial)
          mtgp_mean = mean(filter(!isnan, mtgp_rmses_trial))
          linear_mean = mean(filter(!isnan, linear_rmses_trial))
          mtgp_std = std(filter(!isnan, mtgp_rmses_trial))
          linear_std = std(filter(!isnan, linear_rmses_trial))
          
          println("  Results for $(Int(missing_pct*100))% missing data:")
          println("    MTGP RMSE: $(round(mtgp_mean, digits=4)) ± $(round(mtgp_std, digits=4))")
          println("    Linear RMSE: $(round(linear_mean, digits=4)) ± $(round(linear_std, digits=4))")
          println("    Improvement: $(round((linear_mean - mtgp_mean)/linear_mean * 100, digits=2))%")
      end
      
      push!(results["mtgp_rmse_by_sensor"], mtgp_sensor_rmses)
      push!(results["linear_rmse_by_sensor"], linear_sensor_rmses)
  end
  
  return results
end

"""
Visualize the results of missing data evaluation
"""
function plot_missing_data_results(results::Dict{String, Any})
  missing_percentages = results["missing_percentages"]
  mtgp_rmse = results["mtgp_rmse"]
  linear_rmse = results["linear_rmse"]
  
  # Calculate means and standard deviations
  mtgp_means = [mean(filter(!isnan, mtgp_rmse[i, :])) for i in 1:length(missing_percentages)]
  linear_means = [mean(filter(!isnan, linear_rmse[i, :])) for i in 1:length(missing_percentages)]
  mtgp_stds = [std(filter(!isnan, mtgp_rmse[i, :])) for i in 1:length(missing_percentages)]
  linear_stds = [std(filter(!isnan, linear_rmse[i, :])) for i in 1:length(missing_percentages)]
  
  # Main comparison plot
  p1 = plot(missing_percentages .* 100, mtgp_means,
            ribbon = mtgp_stds,
            label = "MTGP",
            xlabel = "Missing Data Percentage (%)",
            ylabel = "RMSE",
            title = "RMSE vs Missing Data Percentage",
            linewidth = 3,
            fillalpha = 0.3,
            color = :blue,
            size = (800, 600),
            legend = :topleft,
            margin = 5Plots.mm)
  
  plot!(p1, missing_percentages .* 100, linear_means,
        ribbon = linear_stds,
        label = "Linear Interpolation",
        linewidth = 3,
        fillalpha = 0.3,
        color = :red)
  
  # Improvement percentage plot
  improvements = [(linear_means[i] - mtgp_means[i])/linear_means[i] * 100 
                 for i in 1:length(missing_percentages)]
  
  p2 = plot(missing_percentages .* 100, improvements,
            label = "MTGP Improvement over Linear",
            xlabel = "Missing Data Percentage (%)",
            ylabel = "Improvement (%)",
            title = "MTGP Performance Improvement",
            linewidth = 3,
            color = :green,
            size = (800, 400),
            legend = :topright,
            margin = 5Plots.mm)
  hline!(p2, [0], color = :black, linestyle = :dash, alpha = 0.5, label = "No Improvement")
  
  # Combined plot
  combined = plot(p1, p2, layout = (2, 1), size = (800, 1000))
  display(combined)
  savefig(combined, "missing_data_evaluation.png")
  println("\n✓ Saved: missing_data_evaluation.png")
  
  return p1, p2
end

"""
Generate detailed report of the evaluation results
"""
function generate_evaluation_report(results::Dict{String, Any})
  missing_percentages = results["missing_percentages"]
  mtgp_rmse = results["mtgp_rmse"]
  linear_rmse = results["linear_rmse"]
  
  println("\n" * "="^70)
  println("DETAILED EVALUATION REPORT")
  println("="^70)
  
  println("\nSummary Statistics:")
  println("-"^50)
  
  for (i, pct) in enumerate(missing_percentages)
      mtgp_vals = filter(!isnan, mtgp_rmse[i, :])
      linear_vals = filter(!isnan, linear_rmse[i, :])
      
      if !isempty(mtgp_vals) && !isempty(linear_vals)
          mtgp_mean = mean(mtgp_vals)
          linear_mean = mean(linear_vals)
          improvement = (linear_mean - mtgp_mean) / linear_mean * 100
          
          println("Missing $(Int(pct*100))%:")
          println("  MTGP RMSE: $(round(mtgp_mean, digits=4)) ± $(round(std(mtgp_vals), digits=4))")
          println("  Linear RMSE: $(round(linear_mean, digits=4)) ± $(round(std(linear_vals), digits=4))")
          println("  MTGP Improvement: $(round(improvement, digits=2))%")
          println()
      end
  end
  
  # Find best and worst cases
  all_improvements = Float64[]
  for (i, pct) in enumerate(missing_percentages)
      mtgp_vals = filter(!isnan, mtgp_rmse[i, :])
      linear_vals = filter(!isnan, linear_rmse[i, :])
      if !isempty(mtgp_vals) && !isempty(linear_vals)
          improvement = (mean(linear_vals) - mean(mtgp_vals)) / mean(linear_vals) * 100
          push!(all_improvements, improvement)
      end
  end
  
  if !isempty(all_improvements)
      best_idx = argmax(all_improvements)
      worst_idx = argmin(all_improvements)
      
      println("Best MTGP Performance: $(Int(missing_percentages[best_idx]*100))% missing data")
      println("  Improvement: $(round(all_improvements[best_idx], digits=2))%")
      println("Worst MTGP Performance: $(Int(missing_percentages[worst_idx]*100))% missing data")
      println("  Improvement: $(round(all_improvements[worst_idx], digits=2))%")
      println()
      println("Average MTGP Improvement: $(round(mean(all_improvements), digits=2))%")
  end
end

"""
Quick evaluation function for testing
"""
function quick_missing_data_test(data::MultiSensorData; missing_pct::Float64=0.3)
  println("\n" * "="^70)
  println("Quick Missing Data Test ($(Int(missing_pct*100))% missing)")
  println("="^70)
  
  # Create missing data
  missing_data, removed_times, removed_values = create_missing_data(data, missing_pct)
  
  println("Original data points per sensor: ", [length(data.times[s]) for s in 1:min(5, data.S)])
  println("Remaining data points per sensor: ", [length(missing_data.times[s]) for s in 1:min(5, data.S)])
  println("Removed data points per sensor: ", [length(removed_times[s]) for s in 1:min(5, data.S)])
  
  # Train both methods
  println("\nTraining MTGP...")
  mtgp_result = train_icm_mtgp(missing_data; num_epochs=50, lr=0.01, verbose=false)
  
  println("Training Linear Interpolation...")
  linear_result = train_linear_interpolation(missing_data; verbose=false)
  
  # Evaluate
  mtgp_rmses = Float64[]
  linear_rmses = Float64[]
  
  for s in 1:data.S
      if !isempty(removed_times[s])
          # MTGP predictions
          μ_mtgp, σ_mtgp = icm_predict(mtgp_result, s, removed_times[s])
          mtgp_rmse = sqrt(mean((μ_mtgp .- removed_values[s]).^2))
          
          # Linear predictions  
          μ_linear, σ_linear = linear_predict(linear_result, s, removed_times[s])
          linear_rmse = sqrt(mean((μ_linear .- removed_values[s]).^2))
          
          push!(mtgp_rmses, mtgp_rmse)
          push!(linear_rmses, linear_rmse)
      end
  end
  
  println("\nResults:")
  println("  MTGP RMSE: $(round(mean(mtgp_rmses), digits=4)) ± $(round(std(mtgp_rmses), digits=4))")
  println("  Linear RMSE: $(round(mean(linear_rmses), digits=4)) ± $(round(std(linear_rmses), digits=4))")
  println("  MTGP Improvement: $(round((mean(linear_rmses) - mean(mtgp_rmses))/mean(linear_rmses) * 100, digits=2))%")
  
  return mtgp_result, linear_result, removed_times, removed_values
end


function plot_noise_missing_results(results::Dict; 
                                   title_prefix::String="",
                                   save_path::Union{String, Nothing}=nothing)
    """
    绘制噪声和缺失数据影响的结果
    """
    
    try
        # 提取结果数据
        noise_levels = results["noise_levels"]
        missing_percentages = results["missing_percentages"] 
        mtgp_rmse = results["mtgp_rmse"]
        linear_rmse = results["linear_rmse"]
        
        println("📊 Plotting results...")
        println("  Noise levels: $noise_levels")
        println("  Missing percentages: $missing_percentages")
        println("  MTGP RMSE shape: $(size(mtgp_rmse))")
        println("  Linear RMSE shape: $(size(linear_rmse))")
        
        # 使用简化的绘图方法
        return simple_plot_results(mtgp_rmse, linear_rmse, noise_levels, missing_percentages)
        
    catch e
        println("⚠️  Error in plot_noise_missing_results: $e")
        println("📊 Attempting fallback visualization...")
        
        # 简单的统计输出作为后备
        if haskey(results, "mtgp_rmse") && haskey(results, "linear_rmse")
            mtgp_avg = mean(results["mtgp_rmse"])
            linear_avg = mean(results["linear_rmse"])
            improvement = (linear_avg - mtgp_avg) / linear_avg * 100
            
            println("📈 Summary Statistics:")
            println("  Average MTGP RMSE: $(round(mtgp_avg, digits=4))")
            println("  Average Linear RMSE: $(round(linear_avg, digits=4))")
            println("  MTGP Improvement: $(round(improvement, digits=2))%")
        end
        
        return nothing
    end
end

function simple_plot_results(mtgp_rmse, linear_rmse, noise_levels, missing_percentages)
    """
    简化版的结果可视化函数
    """
    
    try
        println("🎨 Creating visualization...")
        
        # 创建基本的比较图
        p = plot(title="MTGP vs Linear Interpolation Performance",
                xlabel="Test Condition",
                ylabel="RMSE",
                legend=:topright,
                size=(800, 500))
        
        # 计算平均值用于比较
        mtgp_avg = mean(mtgp_rmse)
        linear_avg = mean(linear_rmse)
        
        # 创建柱状图比较
        methods = ["MTGP", "Linear"]
        values = [mtgp_avg, linear_avg]
        
        bar_plot = bar(methods, values,
                      title="Average RMSE Comparison",
                      ylabel="RMSE",
                      color=[:blue :red],
                      alpha=0.7,
                      size=(600, 400))
        
        # 添加数值标签
        for (i, v) in enumerate(values)
            annotate!(bar_plot, [(i, v + max(values...)*0.05, 
                                text("$(round(v, digits=4))", 10, :center))])
        end
        
        display(bar_plot)
        
        # 计算并显示改进百分比
        improvement = (linear_avg - mtgp_avg) / linear_avg * 100
        println("📊 Results Summary:")
        println("  MTGP RMSE: $(round(mtgp_avg, digits=4))")
        println("  Linear RMSE: $(round(linear_avg, digits=4))")
        println("  MTGP Improvement: $(round(improvement, digits=2))%")
        
        if improvement < 0
            println("⚠️  Linear interpolation performs better than MTGP!")
        else
            println("✅ MTGP performs better than linear interpolation!")
        end
        
        return bar_plot
        
    catch e
        println("⚠️  Error in simple_plot_results: $e")
        println("📊 Showing basic statistics only:")
        
        println("MTGP RMSE statistics:")
        println("  Mean: $(mean(mtgp_rmse))")
        println("  Std: $(std(mtgp_rmse))")
        println("  Min: $(minimum(mtgp_rmse))")
        println("  Max: $(maximum(mtgp_rmse))")
        
        println("Linear RMSE statistics:")
        println("  Mean: $(mean(linear_rmse))")
        println("  Std: $(std(linear_rmse))")
        println("  Min: $(minimum(linear_rmse))")
        println("  Max: $(maximum(linear_rmse))")
        
        return nothing
    end
end

# ============================================
# 2. 传感器诊断可视化函数
# ============================================

function create_diagnostic_plots(data::MultiSensorData, sensor_indices::Vector{Int})
    """
    创建传感器数据的诊断图
    """
    
    try
        valid_indices = filter(x -> x <= data.S, sensor_indices)
        n_sensors = length(valid_indices)
        
        if n_sensors == 0
            println("❌ No valid sensors to plot")
            return nothing
        end
        
        println("📈 Creating diagnostic plots for $n_sensors sensors...")
        
        plots_array = []
        
        for (i, s) in enumerate(valid_indices)
            times = data.times[s]
            values = data.values[s]
            
            # 获取传感器信息
            sensor_name = s <= length(data.sensor_names) ? data.sensor_names[s] : "Sensor $s"
            sensor_type = s <= length(data.sensor_types) ? string(data.sensor_types[s]) : "Unknown"
            
            # 创建时间序列图
            p = plot(times, values,
                    title="$sensor_name ($sensor_type)",
                    xlabel="Time", 
                    ylabel="Value",
                    linewidth=2,
                    alpha=0.8,
                    titlefontsize=9,
                    guidefontsize=7,
                    legend=false,
                    grid=true)
            
            # 添加统计信息
            mean_val = mean(values)
            std_val = std(values)
            range_val = maximum(values) - minimum(values)
            
            # 在图上添加文本注释
            stats_text = "μ=$(round(mean_val,digits=4))\nσ=$(round(std_val,digits=4))\nΔ=$(round(range_val,digits=4))"
            
            # 找一个合适的位置放置文本
            x_pos = times[1] + (times[end] - times[1]) * 0.05
            y_pos = minimum(values) + (maximum(values) - minimum(values)) * 0.8
            
            annotate!(p, [(x_pos, y_pos, text(stats_text, 7, :left))])
            
            push!(plots_array, p)
        end
        
        # 根据传感器数量决定布局
        if n_sensors == 1
            layout = (1, 1)
        elseif n_sensors == 2
            layout = (1, 2)
        elseif n_sensors <= 4
            layout = (2, 2)
        elseif n_sensors <= 6
            layout = (2, 3)
        else
            layout = (3, 3)
            plots_array = plots_array[1:9]  # 最多显示9个
        end
        
        final_plot = plot(plots_array...,
                         layout=layout,
                         size=(min(800, 400*layout[2]), min(600, 300*layout[1])),
                         margin=3Plots.mm)
        
        display(final_plot)
        return final_plot
        
    catch e
        println("⚠️  Error in create_diagnostic_plots: $e")
        
        # 提供基本的数据统计作为后备
        println("📊 Basic sensor statistics:")
        valid_indices = filter(x -> x <= data.S, sensor_indices)
        
        for (i, s) in enumerate(valid_indices)
            values = data.values[s]
            sensor_name = s <= length(data.sensor_names) ? data.sensor_names[s] : "Sensor $s"
            
            println("  $sensor_name:")
            println("    Mean: $(round(mean(values), digits=4))")
            println("    Std: $(round(std(values), digits=4))")
            println("    Range: $(round(maximum(values) - minimum(values), digits=4))")
            println("    Points: $(length(values))")
        end
        
        return nothing
    end
end

# ============================================
# 3. 快速修复函数 - 替换原有的可视化调用
# ============================================

function safe_visualization_call(results::Dict, title::String="Results")
    """
    安全的可视化调用，带有错误处理
    """
    
    println("\n📊 Attempting to create visualization: $title")
    
    try
        # 尝试使用主要的可视化函数
        plot_result = plot_noise_missing_results(results, title_prefix=title)
        
        if plot_result !== nothing
            println("✅ Visualization created successfully!")
            return plot_result
        else
            println("⚠️  Visualization returned nothing, showing summary instead")
        end
        
    catch e
        println("❌ Visualization failed: $e")
    end
    
    # 后备方案：显示数值结果
    println("\n📈 Numerical Results Summary:")
    for (key, value) in results
        if isa(value, AbstractArray) && length(value) > 0
            println("  $key: mean=$(round(mean(value), digits=4)), std=$(round(std(value), digits=4))")
        elseif isa(value, Number)
            println("  $key: $(round(value, digits=4))")
        else
            println("  $key: $value")
        end
    end
    
    return nothing
end