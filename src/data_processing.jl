struct MultiSensorData
    S::Int
    times::Vector{Vector{Float32}}
    values::Vector{Vector{Float32}}
    sensor_names::Vector{String}
    sensor_types::Vector{Symbol}
end

struct NormalizationParams
    x_mean::Float32
    x_std::Float32
    y_means::Vector{Float32}
    y_stds::Vector{Float32}
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

"""
Build complete multi-sensor dataset with all available sensors
"""
function build_complete_multisensor_data(ds; max_points_per_sensor::Int=300)
    
    times_list = Vector{Vector{Float32}}()
    values_list = Vector{Vector{Float32}}()
    names_list = String[]
    types_list = Symbol[]
    
    println("="^70)
    println("Building Complete Multi-Sensor Dataset")
    println("="^70)
    
    # Define all sensors based on your specification
    scada_sensors = [
        # Bus 702: A, B, C phase voltage magnitude
        ("702", [:A, :B, :C], :Vmag),
        # Bus 703: A, B, C phase voltage magnitude
        ("703", [:A, :B, :C], :Vmag),
        # Bus 730: A, B, C phase voltage magnitude
        ("730", [:A, :B, :C], :Vmag),
    ]
    
    ami_sensors = [
        # Bus 701: A, B, C phase active and reactive power
        ("701", [:A, :B, :C], :P_kW),
        ("701", [:A, :B, :C], :Q_kvar),
        # Bus 744: A phase active and reactive power
        ("744", [:A], :P_kW),
        ("744", [:A], :Q_kvar),
        # Bus 728: A, B, C phase active and reactive power
        ("728", [:A, :B, :C], :P_kW),
        ("728", [:A, :B, :C], :Q_kvar),
        # Bus 729: A phase active and reactive power
        ("729", [:A], :P_kW),
        ("729", [:A], :Q_kvar),
        # Bus 736: B phase active and reactive power
        ("736", [:B], :P_kW),
        ("736", [:B], :Q_kvar),
        # Bus 727: C phase active and reactive power
        ("727", [:C], :P_kW),
        ("727", [:C], :Q_kvar),
    ]
    
    # Process SCADA sensors
    println("\n[SCADA Sensors]")
    scada_count = 0
    for (bus, phases, measurement) in scada_sensors
        if !haskey(ds[:SCADA], bus)
            println("  ⚠️  Skip SCADA-$bus (not found)")
            continue
        end
        
        for phase in phases
            if !haskey(ds[:SCADA][bus][measurement], phase)
                continue
            end
            
            t_raw = ds[:SCADA][bus][:times]
            v_raw = ds[:SCADA][bus][measurement][phase]
            
            t_hours = t_raw ./ 3600.0
            t, v = smart_downsample(t_hours, v_raw, max_points_per_sensor)
            
            if std(v) < 1e-6 || !all(isfinite.(v))
                println("  ⚠️  Skip SCADA-$bus-$phase-$measurement (invalid data)")
                continue
            end
            
            push!(times_list, t)
            push!(values_list, v)
            
            # Create descriptive name
            meas_name = measurement == :Vmag ? "Vmag" : string(measurement)
            push!(names_list, "SCADA-$bus-$phase-$meas_name")
            push!(types_list, :SCADA)
            
            scada_count += 1
            println("  ✓ SCADA-$bus-$phase-$meas_name: $(length(v)) points " *
                    "(range: $(round(minimum(v), digits=2)) ~ $(round(maximum(v), digits=2)))")
        end
    end
    println("  Total SCADA sensors: $scada_count")
    
    # Process AMI sensors
    println("\n[AMI Sensors]")
    ami_count = 0
    for (bus, phases, measurement) in ami_sensors
        if !haskey(ds[:AMI], bus)
            println("  ⚠️  Skip AMI-$bus (not found)")
            continue
        end
        
        for phase in phases
            if !haskey(ds[:AMI][bus][measurement], phase)
                continue
            end
            
            t_raw = ds[:AMI][bus][:times]
            v_raw = ds[:AMI][bus][measurement][phase]
            
            t_hours = t_raw ./ 3600.0
            t, v = smart_downsample(t_hours, v_raw, max_points_per_sensor)
            
            if std(v) < 1e-6 || !all(isfinite.(v))
                println("  ⚠️  Skip AMI-$bus-$phase-$measurement (invalid data)")
                continue
            end
            
            push!(times_list, t)
            push!(values_list, v)
            
            # Create descriptive name
            meas_name = measurement == :P_kW ? "P" : "Q"
            push!(names_list, "AMI-$bus-$phase-$meas_name")
            push!(types_list, :AMI)
            
            ami_count += 1
            println("  ✓ AMI-$bus-$phase-$meas_name: $(length(v)) points " *
                    "(range: $(round(minimum(v), digits=2)) ~ $(round(maximum(v), digits=2)))")
        end
    end
    println("  Total AMI sensors: $ami_count")
    
    # Process PMU sensor (if available)
    println("\n[PMU Sensors]")
    pmu_count = 0
    if haskey(ds, :PMU705)
        for phase in [:A, :B, :C]
            if !haskey(ds[:PMU705][:Vmag], phase)
                continue
            end
            
            t_raw = ds[:PMU705][:times]
            v_raw = ds[:PMU705][:Vmag][phase]
            
            t_hours = t_raw ./ 3600.0
            t, v = smart_downsample(t_hours, v_raw, max_points_per_sensor)
            
            if std(v) < 1e-6 || !all(isfinite.(v))
                println("  ⚠️  Skip PMU-705-$phase (invalid data)")
                continue
            end
            
            push!(times_list, t)
            push!(values_list, v)
            push!(names_list, "PMU-705-$phase-Vmag")
            push!(types_list, :PMU)
            
            pmu_count += 1
            println("  ✓ PMU-705-$phase-Vmag: $(length(v)) points (original: $(length(v_raw))) " *
                    "(range: $(round(minimum(v), digits=2)) ~ $(round(maximum(v), digits=2)))")
        end
    end
    println("  Total PMU sensors: $pmu_count")
    
    S = length(times_list)
    total_points = sum(length.(values_list))
    
    println("\n" * "="^70)
    println("Dataset Built Successfully")
    println("="^70)
    println("  Total sensors: $S")
    println("    - SCADA: $scada_count")
    println("    - AMI: $ami_count")
    println("    - PMU: $pmu_count")
    println("  Total data points: $total_points")
    println("  Estimated memory: ~$(round(total_points * 8 / 1024^2, digits=2)) MB")
    println("="^70)
    
    return MultiSensorData(S, times_list, values_list, names_list, types_list)
end

function normalize_multisensor_data(data::MultiSensorData)
    all_times = vcat(data.times...)
    x_mean = Float32(mean(all_times))
    x_std = Float32(std(all_times))
    
    y_means = Float32[]
    y_stds = Float32[]
    times_norm = Vector{Vector{Float32}}()
    values_norm = Vector{Vector{Float32}}()
    
    for s in 1:data.S
        t_norm = (data.times[s] .- x_mean) ./ x_std
        push!(times_norm, t_norm)
        
        y_mean = Float32(mean(data.values[s]))
        y_std = Float32(std(data.values[s]))
        v_norm = (data.values[s] .- y_mean) ./ y_std
        push!(values_norm, v_norm)
        
        push!(y_means, y_mean)
        push!(y_stds, y_std)
    end
    
    norm_params = NormalizationParams(x_mean, x_std, y_means, y_stds)
    norm_data = MultiSensorData(data.S, times_norm, values_norm, 
                                 data.sensor_names, data.sensor_types)
    
    return norm_data, norm_params
end