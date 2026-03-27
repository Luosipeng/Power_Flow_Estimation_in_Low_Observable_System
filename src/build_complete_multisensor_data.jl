# ============================================
# 1. 扩展 MultiSensorData 结构
# ============================================
struct MultiSensorData
    S::Int                              # 传感器总数
    times::Vector{Vector{Float32}}      # 每个传感器的时间序列
    values::Vector{Vector{Float32}}     # 每个传感器的值序列
    sensor_names::Vector{String}        # 传感器名称
    sensor_types::Vector{Symbol}        # 传感器类型 (:PMU, :SCADA, :AMI)
    is_zero_load::Vector{Bool}          # ✅ 新增：标记零负荷节点
end

# ============================================
# 2. 修改 build_complete_multisensor_data
# ============================================
function build_complete_multisensor_data(ds; max_points_per_sensor::Int=864000, 
                                         pmu_sensors, scada_sensors, ami_sensors)
    
    times_list = Vector{Vector{Float32}}()
    values_list = Vector{Vector{Float32}}()
    names_list = String[]
    types_list = Symbol[]
    zero_load_flags = Bool[]  # ✅ 新增
    
    println("="^70)
    println("Building Complete Multi-Sensor Dataset (Three-Phase)")
    println("="^70)
    
    # ============================================
    # Process PMU sensors
    # ============================================
    println("\n[PMU Sensors]")
    pmu_count = 0
    pmu_zero_load = 0
    
    for (bus, phase, measurement) in pmu_sensors
        sensor_name = "PMU-$bus-$phase-$(string(measurement))"
        
        if !haskey(ds[:PMU], bus)
            println("  ⚠️  Skip $sensor_name (bus not found)")
            continue
        end
        
        if !haskey(ds[:PMU][bus], phase)
            println("  ⚠️  Skip $sensor_name (phase not found)")
            continue
        end
        
        phase_data = ds[:PMU][bus][phase]
        
        if !haskey(phase_data, :Times) || !haskey(phase_data, measurement)
            println("  ⚠️  Skip $sensor_name (measurement not found)")
            continue
        end

        t_raw = phase_data[:Times]
        v_raw = phase_data[measurement]

        t_hours = t_raw ./ 36000.0
        t, v = smart_downsample(t_hours, v_raw, max_points_per_sensor)
        
        # ✅ 关键修改：区分"无效数据"和"零负荷"
        is_zero_load = (std(v) < 1e-6)
        is_invalid = !all(isfinite.(v))
        
        if is_invalid
            println("  ⚠️  Skip $sensor_name (invalid data: NaN/Inf)")
            continue
        end
        
        if is_zero_load
            # ✅ 保留零负荷传感器，但标记它
            push!(times_list, Float32[])      # 空时间序列
            push!(values_list, Float32[])     # 空值序列
            push!(names_list, sensor_name)
            push!(types_list, :PMU)
            push!(zero_load_flags, true)      # ✅ 标记为零负荷
            
            pmu_zero_load += 1
            println("  ������ $sensor_name: ZERO-LOAD (kept for topology)")
        else
            # 正常传感器
            push!(times_list, t)
            push!(values_list, v)
            push!(names_list, sensor_name)
            push!(types_list, :PMU)
            push!(zero_load_flags, false)
            
            pmu_count += 1
            println("  ✓ $sensor_name: $(length(v)) points " *
                    "(range: $(round(minimum(v), digits=4)) ~ $(round(maximum(v), digits=4)))")
        end
    end
    println("  Total PMU sensors: $pmu_count (+ $pmu_zero_load zero-load)")
    
    # ============================================
    # Process SCADA sensors
    # ============================================
    println("\n[SCADA Sensors]")
    scada_count = 0
    scada_zero_load = 0
    
    for (bus, phase, measurement) in scada_sensors
        sensor_name = "SCADA-$bus-$phase-$(string(measurement))"
        
        if !haskey(ds[:SCADA], bus)
            println("  ⚠️  Skip $sensor_name (bus not found)")
            continue
        end
        
        if !haskey(ds[:SCADA][bus], phase)
            println("  ⚠️  Skip $sensor_name (phase not found)")
            continue
        end
        
        phase_data = ds[:SCADA][bus][phase]
        
        if !haskey(phase_data, :Times) || !haskey(phase_data, measurement)
            println("  ⚠️  Skip $sensor_name (measurement not found)")
            continue
        end
            
        t_raw = phase_data[:Times]
        v_raw = phase_data[measurement]
        
        t_hours = t_raw ./ 36000.0
        t, v = smart_downsample(t_hours, v_raw, max_points_per_sensor)
        
        is_zero_load = (std(v) < 1e-6)
        is_invalid = !all(isfinite.(v))
        
        if is_invalid
            println("  ⚠️  Skip $sensor_name (invalid data: NaN/Inf)")
            continue
        end
        
        if is_zero_load
            push!(times_list, Float32[])
            push!(values_list, Float32[])
            push!(names_list, sensor_name)
            push!(types_list, :SCADA)
            push!(zero_load_flags, true)
            
            scada_zero_load += 1
            println("  ������ $sensor_name: ZERO-LOAD (kept for topology)")
        else
            push!(times_list, t)
            push!(values_list, v)
            push!(names_list, sensor_name)
            push!(types_list, :SCADA)
            push!(zero_load_flags, false)
            
            scada_count += 1
            println("  ✓ $sensor_name: $(length(v)) points " *
                    "(range: $(round(minimum(v), digits=4)) ~ $(round(maximum(v), digits=4)))")
        end
    end
    println("  Total SCADA sensors: $scada_count (+ $scada_zero_load zero-load)")
    
    # ============================================
    # Process AMI sensors
    # ============================================
    println("\n[AMI Sensors]")
    ami_count = 0
    ami_zero_load = 0
    
    for (bus, phase, measurement) in ami_sensors
        meas_name = measurement == :P_kW ? "P" : "Q"
        sensor_name = "AMI-$bus-$phase-$meas_name"
        
        if !haskey(ds[:AMI], bus)
            println("  ⚠️  Skip $sensor_name (bus not found)")
            continue
        end
        
        if !haskey(ds[:AMI][bus], phase)
            println("  ⚠️  Skip $sensor_name (phase not found)")
            continue
        end
        
        phase_data = ds[:AMI][bus][phase]
        
        if !haskey(phase_data, :Times) || !haskey(phase_data, measurement)
            println("  ⚠️  Skip $sensor_name (measurement not found)")
            continue
        end

        t_raw = phase_data[:Times]
        v_raw = phase_data[measurement]

        t_hours = t_raw ./ 36000.0
        t, v = smart_downsample(t_hours, v_raw, max_points_per_sensor)
        
        is_zero_load = (std(v) < 1e-5)
        is_invalid = !all(isfinite.(v))
        
        if is_invalid
            println("  ⚠️  Skip $sensor_name (invalid data: NaN/Inf)")
            continue
        end
        
        if is_zero_load
            push!(times_list, Float32[])
            push!(values_list, Float32[])
            push!(names_list, sensor_name)
            push!(types_list, :AMI)
            push!(zero_load_flags, true)
            
            ami_zero_load += 1
            println("  ������ $sensor_name: ZERO-LOAD (kept for topology)")
        else
            push!(times_list, t)
            push!(values_list, v)
            push!(names_list, sensor_name)
            push!(types_list, :AMI)
            push!(zero_load_flags, false)
            
            ami_count += 1
            println("  ✓ $sensor_name: $(length(v)) points " *
                    "(range: $(round(minimum(v), digits=4)) ~ $(round(maximum(v), digits=4)))")
        end
    end
    println("  Total AMI sensors: $ami_count (+ $ami_zero_load zero-load)")
    
    # ============================================
    # Summary
    # ============================================
    S = length(times_list)
    total_points = sum(length.(values_list))
    total_zero_load = pmu_zero_load + scada_zero_load + ami_zero_load
    
    println("\n" * "="^70)
    println("Dataset Built Successfully")
    println("="^70)
    println("  Total sensors: $S")
    println("    - PMU: $pmu_count (+ $pmu_zero_load zero-load)")
    println("    - SCADA: $scada_count (+ $scada_zero_load zero-load)")
    println("    - AMI: $ami_count (+ $ami_zero_load zero-load)")
    println("  Active data points: $total_points")
    println("  Zero-load sensors: $total_zero_load")
    println("  Estimated memory: ~$(round(total_points * 8 / 1024^2, digits=2)) MB")
    println("="^70)
    
    return MultiSensorData(S, times_list, values_list, names_list, types_list, zero_load_flags)
end

function smart_downsample(times::Vector, values::Vector, max_points::Int=500)
    n = length(times)
    if n <= max_points
        return Float32.(times), Float32.(values)
    end
    step = n ÷ max_points
    indices = 1:step:n
    return Float32.(times[indices]), Float32.(values[indices])
end
