struct MultiSensorData
    S::Int
    times::Vector{Vector{Float32}}
    values::Vector{Vector{Float32}}
    sensor_names::Vector{String}
    sensor_types::Vector{Symbol}
end

"""
Build complete multi-sensor dataset with all available sensors
"""
function build_complete_multisensor_data(ds; max_points_per_sensor::Int=864000)
    
    times_list = Vector{Vector{Float32}}()
    values_list = Vector{Vector{Float32}}()
    names_list = String[]
    types_list = Symbol[]
    
    println("="^70)
    println("Building Complete Multi-Sensor Dataset")
    println("="^70)
    
    # Define all sensors based on your specification
    pmu_sensors = [
        # PMU-20: voltage magnitude
        ("3", :Vmag),
        # PMU-20: voltage real parts
        ("3", :V_real),
        # PMU-20: voltage imaginary parts
        ("3", :V_imag),

    ]

    scada_sensors = [
        # Bus 3: voltage magnitude
        ("8", :Vmag),
        # Bus 6: voltage magnitude
        ("15", :Vmag),
        # Bus 12: voltage magnitude
        ("12", :Vmag),
        # Bus 19: voltage magnitude
        ("32", :Vmag),
    ]
    
    ami_sensors = [
        # Bus 2: active and reactive power
        ("18", :P_kW),
        ("18", :Q_kVAR),
        # Bus 33: active and reactive power
        ("22", :P_kW),
        ("22", :Q_kVAR),
        # Bus 10: active and reactive power
        ("25", :P_kW),
        ("25", :Q_kVAR),
        # Bus 21: active and reactive power
        ("29", :P_kW),
        ("29", :Q_kVAR),
        # Bus 23: active and reactive power
        ("36", :P_kW),
        ("36", :Q_kVAR),
        # Bus 24: active and reactive power
        ("37", :P_kW),
        ("37", :Q_kVAR),
        # Bus 33: active and reactive power
        ("33", :P_kW),
        ("33", :Q_kVAR),
    ]
    
    # Process SCADA sensors
    println("\n[SCADA Sensors]")
    scada_count = 0
    for (bus, measurement) in scada_sensors
        if !haskey(ds[:SCADA], bus)
            println("  ⚠️  Skip SCADA-$bus (not found)")
            continue
        end
            
        t_raw = ds[:SCADA][bus][:Times]
        v_raw = ds[:SCADA][bus][measurement]
        
        t_hours = t_raw ./ 36000.0
        t, v = smart_downsample(t_hours, v_raw, max_points_per_sensor)
        
        if std(v) < 1e-6 || !all(isfinite.(v))
            println("  ⚠️  Skip SCADA-$bus-$measurement (invalid data)")
            continue
        end
        
        push!(times_list, t)
        push!(values_list, v)
        
        # Create descriptive name
        meas_name = measurement == :Vmag ? "Vmag" : string(measurement)
        push!(names_list, "SCADA-$bus-$meas_name")
        push!(types_list, :SCADA)
        
        scada_count += 1
        println("  ✓ SCADA-$bus-$meas_name: $(length(v)) points " *
                "(range: $(round(minimum(v), digits=2)) ~ $(round(maximum(v), digits=2)))")

    end
    println("  Total SCADA sensors: $scada_count")
    
    # Process AMI sensors
    println("\n[AMI Sensors]")
    ami_count = 0
    for (bus, measurement) in ami_sensors
        if !haskey(ds[:AMI], bus)
            println("  ⚠️  Skip AMI-$bus (not found)")
            continue
        end

        t_raw = ds[:AMI][bus][:Times]
        v_raw = ds[:AMI][bus][measurement]

        t_hours = t_raw ./ 36000.0
        t, v = smart_downsample(t_hours, v_raw, max_points_per_sensor)
        
        if std(v) < 1e-6 || !all(isfinite.(v))
            println("  ⚠️  Skip AMI-$bus-$measurement (invalid data)")
            continue
        end
        
        push!(times_list, t)
        push!(values_list, v)
        
        # Create descriptive name
        meas_name = measurement == :P_kW ? "P" : "Q"
        push!(names_list, "AMI-$bus-$meas_name")
        push!(types_list, :AMI)
        
        ami_count += 1
        println("  ✓ AMI-$bus-$meas_name: $(length(v)) points " *
                "(range: $(round(minimum(v), digits=2)) ~ $(round(maximum(v), digits=2)))")
    end
    println("  Total AMI sensors: $ami_count")
    
    println("\n[PMU Sensors]")
    pmu_count = 0
    for (bus, measurement) in pmu_sensors
        if !haskey(ds[:PMU], bus)
            println("  ⚠️  Skip PMU-$bus (not found)")
            continue
        end

        t_raw = ds[:PMU][bus][:Times]
        v_raw = ds[:PMU][bus][measurement]

        t_hours = t_raw ./ 36000.0
        t, v = smart_downsample(t_hours, v_raw, max_points_per_sensor)
        
        if std(v) < 1e-6 || !all(isfinite.(v))
            println("  ⚠️  Skip PMU-$bus-$measurement (invalid data)")
            continue
        end
        
        push!(times_list, t)
        push!(values_list, v)
        
        # Create descriptive name
        meas_name = measurement == :V_mag ? "V" : string(measurement)
        push!(names_list, "PMU-$bus-$meas_name")
        push!(types_list, :PMU)

        pmu_count += 1
        println("  ✓ PMU-$bus-$meas_name: $(length(v)) points " *
                "(range: $(round(minimum(v), digits=2)) ~ $(round(maximum(v), digits=2)))")
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


function smart_downsample(times::Vector, values::Vector, max_points::Int=500)
    n = length(times)
    if n <= max_points
        return Float32.(times), Float32.(values)
    end
    step = n ÷ max_points
    indices = 1:step:n
    return Float32.(times[indices]), Float32.(values[indices])
end