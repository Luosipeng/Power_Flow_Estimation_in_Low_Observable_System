
"""
单次训练并可视化指定传感器的预测结果
包含MTGP、线性插值结果和95%置信区间
"""
function train_and_visualize_sensor_predictions(data::MultiSensorData;
                                               target_sensor_indices::Vector{Int}=[1, 2],
                                               noise_level::Float64=0.05,
                                               missing_percentage::Float64=0.3,
                                               mtgp_epochs::Int=200,
                                               time_step_minutes::Float64=1.0,
                                               seed::Int=42,
                                               save_results::Bool=true,
                                               save_pdf::Bool=true,        # 新增：是否保存PDF
                                               save_png::Bool=true,        # 新增：是否保存PNG
                                               output_dir::String="./results/")
    
    println("\n" * "="^80)
    println("🎯 SINGLE TRAINING & VISUALIZATION")
    println("="^80)
    println("🔧 Training: ALL $(data.S) sensors")
    println("📊 Visualization: $(length(target_sensor_indices)) target sensors")
    println("🔊 Noise level: $(Int(noise_level*100))%")
    println("❌ Missing data: $(Int(missing_percentage*100))%")
    println("⏱️  Time step: $(time_step_minutes) minutes")
    println("💾 Save PNG: $save_png, Save PDF: $save_pdf")  # 新增：显示保存选项
    
    # 验证传感器索引
    valid_indices = filter(x -> 1 <= x <= data.S, target_sensor_indices)
    if length(valid_indices) != length(target_sensor_indices)
        println("⚠️  Some sensor indices are invalid. Using valid ones: $valid_indices")
    end
    target_sensor_indices = valid_indices
    
    # 创建输出目录
    if save_results && !isdir(output_dir)
        mkpath(output_dir)
    end
    
    # Step 1: 添加噪声到所有传感器
    println("\n[1] 🔊 Adding noise to all sensors...")
    noisy_data = add_gaussian_noise(data, noise_level, seed=seed)
    
    # Step 2: 创建缺失数据
    println("[2] ❌ Creating missing data...")
    missing_noisy_data, removed_times, removed_values = create_missing_data(
        noisy_data, missing_percentage, seed=seed)
    
    # Step 3: 训练MTGP（使用所有传感器）
    println("[3] 🧠 Training MTGP with ALL $(data.S) sensors...")
    mtgp_result = train_icm_mtgp(missing_noisy_data; num_epochs=mtgp_epochs, lr=0.01, verbose=true)
    
    # Step 4: 训练线性插值（使用所有传感器）
    println("[4] 📏 Training Linear Interpolation with ALL $(data.S) sensors...")
    linear_result = train_linear_interpolation(missing_noisy_data; verbose=true)
    
    # Step 5: 为每个目标传感器生成预测和可视化
    println("[5] 📊 Generating predictions and visualizations...")
    
    all_results = Dict{String, Any}()
    plots_array = []
    
    for (plot_idx, sensor_idx) in enumerate(target_sensor_indices)
        sensor_name = data.sensor_names[sensor_idx]
        sensor_type = data.sensor_types[sensor_idx]
        
        println("\n  📈 Processing sensor $plot_idx/$(length(target_sensor_indices)): $sensor_name")
        
        # 获取传感器的时间范围
        original_times = data.times[sensor_idx]
        original_values = data.values[sensor_idx]
        noisy_times = missing_noisy_data.times[sensor_idx]
        noisy_values = missing_noisy_data.values[sensor_idx]
        
        if isempty(original_times)
            println("    ⚠️  No data for sensor $sensor_idx, skipping...")
            continue
        end
        
        # 创建统一的时间网格（每分钟一个点）
        time_min = minimum(original_times)
        time_max = maximum(original_times)
        time_grid = collect(time_min:time_step_minutes:time_max)
        
        println("    ⏱️  Time range: $(round(time_min, digits=2)) to $(round(time_max, digits=2))")
        println("    📏 Grid points: $(length(time_grid))")
        
        # 获取真实值在时间网格上的插值
        true_values_interp = Float32[]
        for t in noisy_times
            # 找到最近的原始数据点
            time_diffs = abs.(original_times .- t)
            closest_idx = argmin(time_diffs)
            if time_diffs[closest_idx] < time_step_minutes/2  # 在半个时间步长内
                push!(true_values_interp, original_values[closest_idx])
            end
        end
        
        # MTGP预测
        println("    🧠 MTGP prediction...")
        time_grid = Float32.(time_grid)  # 转换为Float32
        μ_mtgp, σ_mtgp = icm_predict(mtgp_result, sensor_idx, time_grid)
        
        # 线性插值预测
        println("    📏 Linear interpolation prediction...")
        μ_linear, σ_linear = linear_predict(linear_result, sensor_idx, time_grid)
        
        # 计算95%置信区间
        confidence_upper = μ_mtgp .+ 1.96 .* σ_mtgp
        confidence_lower = μ_mtgp .- 1.96 .* σ_mtgp
        
        # 计算RMSE（仅在有真实值的点上）
        valid_indices_for_rmse = findall(i -> !isnan(true_values_interp[i]), 1:length(true_values_interp))
        if !isempty(valid_indices_for_rmse)
            mtgp_rmse = sqrt(mean((μ_mtgp[valid_indices_for_rmse] .- true_values_interp[valid_indices_for_rmse]).^2))
            linear_rmse = sqrt(mean((μ_linear[valid_indices_for_rmse] .- true_values_interp[valid_indices_for_rmse]).^2))
            improvement = (linear_rmse - mtgp_rmse) / linear_rmse * 100
        else
            mtgp_rmse = NaN
            linear_rmse = NaN
            improvement = NaN
        end
        
        # 存储结果
        sensor_results = Dict(
            "sensor_name" => sensor_name,
            "sensor_type" => sensor_type,
            "sensor_index" => sensor_idx,
            "time_grid" => time_grid,
            "true_values" => true_values_interp,
            "mtgp_mean" => μ_mtgp,
            "mtgp_std" => σ_mtgp,
            "mtgp_upper_95" => confidence_upper,
            "mtgp_lower_95" => confidence_lower,
            "linear_mean" => μ_linear,
            "linear_std" => σ_linear,
            "original_times" => original_times,
            "original_values" => original_values,
            "noisy_times" => noisy_times,
            "noisy_values" => noisy_values,
            "removed_times" => removed_times[sensor_idx],
            "removed_values" => removed_values[sensor_idx],
            "mtgp_rmse" => mtgp_rmse,
            "linear_rmse" => linear_rmse,
            "improvement_percent" => improvement
        )
        
        all_results["sensor_$(sensor_idx)"] = sensor_results
        
        # 创建可视化
        println("    🎨 Creating visualization...")
        p = plot(
                xlabel="Time (hours)",
                ylabel="Reactive Power (kVAR)",
                size=(1200, 400),
                legend=:topright,
                grid=true,
                gridwidth=1,
                gridcolor=:lightgray,
                margin=5Plots.mm,
                fontfamily = "Times New Roman")
        
        # 绘制95%置信区间（填充区域）
        plot!(p, time_grid, -confidence_upper.*1000,
              fillto=-confidence_lower.*1000,
              fillalpha=0.2,
              fillcolor=:blue,
              line=:transparent,
              label="MTGP 95% CI",
              fontfamily = "Times New Roman")
        
        # 绘制真实值
        scatter!(p, noisy_times, -true_values_interp.*1000,
                markersize=3,
                color=:black,
                alpha=1.0,
                label="True Values",
                fontfamily = "Times New Roman")
        
        # 绘制MTGP预测
        plot!(p, time_grid, -μ_mtgp.*1000,
              linewidth=2,
              color=:blue,
              label="MTGP Prediction",
              linestyle=:solid,
              fontfamily = "Times New Roman")
        
        # 绘制线性插值预测
        plot!(p, time_grid, -μ_linear.*1000,
              linewidth=2,
              color=:red,
              label="Linear Interpolation",
              linestyle=:dash,
              fontfamily = "Times New Roman")
        
        # 添加到图形数组
        push!(plots_array, p)
        
        # 保存单个传感器的图
        if save_results
            # 保存PNG格式
            if save_png
                sensor_plot_path_png = joinpath(output_dir, "sensor_$(sensor_idx)_$(sensor_name)_prediction.png")
                savefig(p, sensor_plot_path_png)
                println("    💾 Saved PNG: $sensor_plot_path_png")
            end
            
            # 保存PDF格式
            if save_pdf
                sensor_plot_path_pdf = joinpath(output_dir, "sensor_$(sensor_idx)_$(sensor_name)_prediction.pdf")
                savefig(p, sensor_plot_path_pdf)
                println("    📄 Saved PDF: $sensor_plot_path_pdf")
            end
        end
    end
    
    return all_results
end