struct NormalizationParams
    x_mean::Float32
    x_std::Float32
    y_means::Vector{Float32}
    y_stds::Vector{Float32}
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