    using ProgressMeter
function generate_1min_resolution_predictions(result)
    println("\n[9] Generating 1-minute resolution predictions for 24 hours...")
    println("="^70)

    # 统一1分钟网格（保留，用于下游；对PMU我们也会算一份分钟级，但评估不用它）
    dt_unified = 1.0f0 / 60.0f0
    t_min = minimum([minimum(result.data.times[s]) for s in 1:result.data.S])
    t_max = maximum([maximum(result.data.times[s]) for s in 1:result.data.S])
    t_unified = collect(Float32, t_min:dt_unified:t_max)

    println("  Time grid configuration:")
    println("    Sampling interval: 1 minute")
    println("    Time range: $(round(t_min, digits=2)) ~ $(round(t_max, digits=2)) hours")
    println("    Number of time points: $(length(t_unified))")

    println("\n  Generating GP predictions for all sensors on 1-min grid (for downstream use)...")
    unified_predictions = Dict{String, Any}()
    unified_predictions["time_hours"] = t_unified
    unified_predictions["time_minutes"] = t_unified .* 60.0f0
    unified_predictions["sampling_interval_minutes"] = 1.0
    unified_predictions["num_timepoints"] = length(t_unified)
    unified_predictions["sensors"] = Dict{String, Any}()

    @showprogress for s in 1:result.data.S
        sensor_name = result.data.sensor_names[s]
        sensor_type = result.data.sensor_types[s]
        t_meas = result.data.times[s]
        v_meas = result.data.values[s]

        # 计算原始采样率字符串（修正：针对PMU显示0.1s）
        orig_dt = length(t_meas) > 1 ? mean(diff(sort(t_meas))) : NaN
        orig_rate_str = if isnan(orig_dt)
            "single point"
        elseif orig_dt ≤ (0.1/3600 + 1e-6) # 0.1秒（小时）
            "0.1s"
        elseif orig_dt < 0.01
            "$(round(orig_dt * 3600, digits=1))s"
        elseif orig_dt < 1.0
            "$(round(orig_dt * 60, digits=1))min"
        else
            "$(round(orig_dt, digits=2))h"
        end

        # 1分钟统一网格上的预测（保持，为下游使用）
        μ_pred_1min, σ_pred_1min = multitask_gp_predict(result, s, t_unified)

        unified_predictions["sensors"][sensor_name] = Dict(
            "type" => String(sensor_type),
            "prediction_mean" => μ_pred_1min,
            "prediction_std" => σ_pred_1min,
            "prediction_ci_lower" => μ_pred_1min .- 1.96f0 .* σ_pred_1min,
            "prediction_ci_upper" => μ_pred_1min .+ 1.96f0 .* σ_pred_1min,
            "measurement_times" => t_meas,
            "measurement_values" => v_meas,
            "original_sampling_rate" => orig_rate_str,
            "num_measurements" => length(t_meas),
            "num_predictions" => length(t_unified)
        )
    end

    println("  ✓ Successfully generated 1-min predictions for $(length(unified_predictions["sensors"])) sensors")

    # 评估：在原始测量点上评估；PMU使用0.1s高频点，不用1分钟网格
    println("\n  Computing prediction quality metrics at original sampling points...")
    metrics = DataFrame(
        Sensor = String[],
        Type = String[],
        Original_Rate = String[],
        Num_Measurements = Int[],
        RMSE = Float32[],
        MAE = Float32[],
        MAPE = Float32[],
        R2 = Float32[],
        Coverage_95 = Float32[],
        Mean_Uncertainty = Float32[]
    )

    # MAPE 分母安全阈值（避免 V_imag 近零爆炸）；可调
    eps_mape_den = 1e-6

    for (sensor_name, sensor_data) in unified_predictions["sensors"]
        t_meas = sensor_data["measurement_times"]
        v_meas = sensor_data["measurement_values"]
        s_idx = findfirst(==(sensor_name), result.data.sensor_names)

        # 在“原始测量点”上做GP预测（PMU=0.1s；其他=其本来频率）
        μ_at_meas, σ_at_meas = multitask_gp_predict(result, s_idx, t_meas)

        residuals = v_meas .- μ_at_meas
        rmse = sqrt(mean(residuals.^2))
        mae = mean(abs.(residuals))

        # 两种 MAPE 选项：默认保留你原公式，加入 eps；如需 sMAPE，可替换下一行
        mape = mean(abs.(residuals) ./ (abs.(v_meas) .+ eps_mape_den)) * 100
        # smape = mean(2 .* abs.(residuals) ./ (abs.(v_meas) .+ abs.(μ_at_meas) .+ eps_mape_den)) * 100

        ss_res = sum(residuals.^2)
        ss_tot = sum((v_meas .- mean(v_meas)).^2)
        r2 = 1 - ss_res / max(ss_tot, 1e-12)

        ci_lower = μ_at_meas .- 1.96f0 .* σ_at_meas
        ci_upper = μ_at_meas .+ 1.96f0 .* σ_at_meas
        coverage = mean((v_meas .>= ci_lower) .& (v_meas .<= ci_upper)) * 100

        mean_uncertainty = mean(σ_at_meas)

        push!(metrics, (
            sensor_name,
            sensor_data["type"],
            sensor_data["original_sampling_rate"],
            length(t_meas),
            rmse, mae, mape, r2, coverage, mean_uncertainty
        ))
    end

    # 排序与打印
    sort!(metrics, [:Type, order(:R2, rev=true)])
    println("\n" * "="^70)
    println("Prediction Quality Metrics (evaluated at original sampling points)")
    println("="^70)
    println(metrics)

    # 保存CSV：保持原有三个文件，但评估现在是高频口径（PMU=0.1s）
    try
        # (a) 1分钟预测长表（保持）
        println("  Saving gp_predictions_1min_timeseries.csv ...")
        df_predictions = DataFrame(
            Time_hours = Float32[],
            Time_minutes = Float32[],
            Sensor = String[],
            Type = String[],
            Prediction_Mean = Float32[],
            Prediction_Std = Float32[],
            CI_Lower = Float32[],
            CI_Upper = Float32[]
        )
        for (sensor_name, sensor_data) in unified_predictions["sensors"]
            n = length(t_unified)
            append!(df_predictions, DataFrame(
                Time_hours = t_unified,
                Time_minutes = t_unified .* 60.0f0,
                Sensor = fill(sensor_name, n),
                Type = fill(sensor_data["type"], n),
                Prediction_Mean = sensor_data["prediction_mean"],
                Prediction_Std = sensor_data["prediction_std"],
                CI_Lower = sensor_data["prediction_ci_lower"],
                CI_Upper = sensor_data["prediction_ci_upper"]
            ))
        end
        CSV.write("gp_predictions_1min_timeseries.csv", df_predictions)

        # (b) 原始测量数据（PMU=0.1s高频）保持
        println("  Saving original_measurements.csv ...")
        df_measurements = DataFrame(
            Time_hours = Float32[],
            Time_minutes = Float32[],
            Sensor = String[],
            Type = String[],
            Measurement = Float32[]
        )
        for (sensor_name, sensor_data) in unified_predictions["sensors"]
            n = length(sensor_data["measurement_times"])
            append!(df_measurements, DataFrame(
                Time_hours = sensor_data["measurement_times"],
                Time_minutes = sensor_data["measurement_times"] .* 60.0f0,
                Sensor = fill(sensor_name, n),
                Type = fill(sensor_data["type"], n),
                Measurement = sensor_data["measurement_values"]
            ))
        end
        CSV.write("original_measurements.csv", df_measurements)

        # (c) 指标
        println("  Saving prediction_quality_metrics.csv ...")
        CSV.write("prediction_quality_metrics.csv", metrics)

        # (d) 宽表（1分钟预测，保持）
        println("  Saving gp_predictions_1min_wide.csv ...")
        df_wide = DataFrame(Time_hours = t_unified, Time_minutes = t_unified .* 60.0f0)
        for (sensor_name, sensor_data) in unified_predictions["sensors"]
            df_wide[!, Symbol(sensor_name)] = sensor_data["prediction_mean"]
        end
        CSV.write("gp_predictions_1min_wide.csv", df_wide)
    catch e
        println("  CSV save failed: $e")
    end

    # 可视化：新增 PMU 高频图（可选）
    try
        println("\n  Generating high-frequency plots for PMU sensors...")
        pmu_names = [k for (k, v) in unified_predictions["sensors"] if v["type"] == "PMU"]
        plots_pmu = []
        for sensor_name in pmu_names
            s_idx = findfirst(==(sensor_name), result.data.sensor_names)
            t_meas = result.data.times[s_idx]
            v_meas = result.data.values[s_idx]
            μ_at_meas, σ_at_meas = multitask_gp_predict(result, s_idx, t_meas)

            p = plot(t_meas, μ_at_meas,
                     ribbon = 1.96f0 .* σ_at_meas,
                     label = "GP @0.1s (95% CI)",
                     xlabel = "Time (hours)",
                     ylabel = "Value",
                     title = sensor_name * " (PMU, 0.1s)",
                     linewidth = 1.0,
                     fillalpha = 0.2,
                     legend = :topright,
                     size = (600, 300),
                     margin = 4Plots.mm)
            scatter!(p, t_meas, v_meas,
                     label = "Data (0.1s)",
                     markersize = 1.0, alpha = 0.5, color = :red, markerstrokewidth = 0)
            push!(plots_pmu, p)
        end
        if !isempty(plots_pmu)
            layout_pmu = (ceil(Int, length(plots_pmu)/2), 2)
            combined_pmu = plot(plots_pmu..., layout = layout_pmu, size = (1200, 320 * layout_pmu[1]))
            display(combined_pmu)
            savefig(combined_pmu, "pmu_highfreq_plots.png")
            println("  ✓ Saved: pmu_highfreq_plots.png")
        end
    catch e
        println("  PMU plotting failed: $e")
    end

    # 其它汇总与柱状图：保持原逻辑（它们读取 metrics，现为高频评估口径）
    # ... 你原有的 R²、Coverage 柱状图代码可直接复用

    unified_result = (
        predictions = unified_predictions,
        metrics = metrics,
        time_grid = t_unified
    )
    return unified_result
end
