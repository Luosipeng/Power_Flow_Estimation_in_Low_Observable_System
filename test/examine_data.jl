function diagnose_sensor(data, sensor_idx)
    println("\n������ DIAGNOSING SENSOR $sensor_idx")
    if sensor_idx > data.S
        println("❌ Index out of bounds")
        return
    end
    
    vals = data.values[sensor_idx]
    name = data.sensor_names[sensor_idx]
    type = data.sensor_types[sensor_idx]
    
    println("  Name: $name")
    println("  Type: $type")
    println("  Length: $(length(vals))")
    println("  Mean: $(mean(vals))")
    println("  Std Dev: $(std(vals))") # 如果这里是 0，那就是问题所在
    println("  Min: $(minimum(vals))")
    println("  Max: $(maximum(vals))")
    println("  First 10 values: $(vals[1:min(10, end)])")
    
    if std(vals) < 1e-6
        println("⚠️ WARNING: This sensor signal is CONSTANT. Standard noise addition will add 0 noise.")
        println("⚠️ Interpolation on a constant line is mathematically perfect (Error = 0).")
    end
end

# 在主流程中调用
diagnose_sensor(data, 104)
