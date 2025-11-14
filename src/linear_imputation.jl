# Linear Interpolation Implementation for Multi-task GP Framework - FIXED VERSION v2
using LinearAlgebra
using Statistics
using Interpolations

"""
Linear interpolation result structure to match MTGP interface
"""
struct LinearInterpolationResult
  interpolators::Vector{Any}  # Store interpolation objects for each sensor
  data::MultiSensorData      # Original data
  norm_params::Any           # Normalization parameters
  sensor_names::Vector{String}
  losses::Vector{Float32}    # Dummy losses for compatibility
end

"""
Create linear interpolation for a single sensor
"""
function create_linear_interpolator(times::Vector{Float32}, values::Vector{Float32})
  if length(times) < 2
      # For single point, return constant interpolation
      return x -> fill(values[1], length(x))
  end
  
  # Sort by time to ensure monotonic interpolation
  sorted_indices = sortperm(times)
  sorted_times = times[sorted_indices]
  sorted_values = values[sorted_indices]
  
  # Create linear interpolation object
  itp = LinearInterpolation(sorted_times, sorted_values, extrapolation_bc=Interpolations.Line())
  
  return itp
end

"""
Train linear interpolation model (mimics MTGP interface)
"""
function train_linear_interpolation(data::MultiSensorData; verbose::Bool=true)
  println("\n" * "="^70)
  println("Training Linear Interpolation Model")
  println("="^70)
  
  # Normalize data (same as MTGP)
  norm_data, norm_params = normalize_multisensor_data(data)
  
  # Create interpolators for each sensor
  interpolators = Vector{Any}(undef, data.S)
  
  if verbose
      println("Creating linear interpolators for $(data.S) sensors...")
  end
  
  for s in 1:data.S
      interpolators[s] = create_linear_interpolator(norm_data.times[s], norm_data.values[s])
      if verbose && (s % 10 == 0 || s == 1)
          println("  Created interpolator for sensor $s: $(data.sensor_names[s])")
      end
  end
  
  # Dummy losses for compatibility
  losses = Float32[0.0]
  
  if verbose
      println("✓ Linear interpolation model ready!")
  end
  
  return LinearInterpolationResult(
      interpolators,
      data,
      norm_params,
      data.sensor_names,
      losses
  )
end

"""
Linear interpolation prediction - FIXED to handle both Float32 and Float64 without recursion
"""
function linear_predict(result::LinearInterpolationResult, sensor_idx::Int, x_test::Vector{T}) where T <: AbstractFloat
  # Convert to Float32 for consistency
  x_test_f32 = convert(Vector{Float32}, x_test)
  
  # Normalize test inputs
  x_test_norm = (x_test_f32 .- result.norm_params.x_mean) ./ result.norm_params.x_std
  
  # Get interpolator for this sensor
  interpolator = result.interpolators[sensor_idx]
  
  # Predict normalized values
  y_pred_norm = interpolator(x_test_norm)
  
  # Denormalize predictions
  y_pred = y_pred_norm .* result.norm_params.y_stds[sensor_idx] .+ result.norm_params.y_means[sensor_idx]
  
  # Linear interpolation has no uncertainty, so set small constant uncertainty
  σ_pred = fill(0.01f0 * result.norm_params.y_stds[sensor_idx], length(y_pred))
  
  return y_pred, σ_pred
end

"""
Generate 1-minute resolution predictions using linear interpolation - FIXED
"""
function generate_1min_resolution_predictions_linear(result::LinearInterpolationResult)
  println("\n[Linear] Generating 1-minute resolution predictions...")
  
  # Find global time range
  all_times = vcat(result.data.times...)
  t_min, t_max = extrema(all_times)
  
  # Create 1-minute resolution time grid (in hours) - Convert to Float32
  time_grid = Float32.(collect(t_min:(1/60):t_max))  # 1 minute = 1/60 hour
  
  println("  Time range: $(round(t_min, digits=2)) to $(round(t_max, digits=2)) hours")
  println("  Grid points: $(length(time_grid))")
  
  # Generate predictions for each sensor
  predictions = Dict{String, Tuple{Vector{Float32}, Vector{Float32}, Vector{Float32}}}()
  
  for s in 1:result.data.S
      sensor_name = result.sensor_names[s]
      μ_pred, σ_pred = linear_predict(result, s, time_grid)
      predictions[sensor_name] = (time_grid, μ_pred, σ_pred)
      
      if s % 20 == 0 || s == result.data.S
          println("  Completed predictions for $s/$(result.data.S) sensors")
      end
  end
  
  return predictions
end

"""
Visualize linear interpolation results (compatible with existing visualization code) - FIXED
"""
function visualize_linear_predictions(result::LinearInterpolationResult; max_sensors::Int=9)
  println("\n[Linear] Creating visualizations...")
  
  # Select sensors same way as MTGP code
  scada_indices = findall(x -> x == :SCADA, result.data.sensor_types)
  ami_indices = findall(x -> x == :AMI, result.data.sensor_types)
  pmu_indices = findall(x -> x == :PMU, result.data.sensor_types)
  
  selected_indices = Int[]
  if length(scada_indices) >= 2
      append!(selected_indices, scada_indices[1:2])
  end
  if length(ami_indices) >= 2
      append!(selected_indices, ami_indices[1:2])
  end
  if length(pmu_indices) >= 1
      push!(selected_indices, pmu_indices[1])
  end
  selected_indices = selected_indices[1:min(max_sensors, length(selected_indices))]
  
  plots_list = []
  for s in selected_indices
      x_test = range(minimum(result.data.times[s]), maximum(result.data.times[s]), length=200)
      μ_pred, σ_pred = linear_predict(result, s, collect(Float32, x_test))  # Explicitly convert to Float32
      
      p = plot(x_test, μ_pred,
                  ribbon = 1.96 .* σ_pred,
                  label = "Linear Pred",
                  xlabel = "Time (hours)",
                  ylabel = "Value",
                  title = result.sensor_names[s],
                  linewidth = 2,
                  fillalpha = 0.2,
                  legend = :topright,
                  size = (500, 350),
                  margin = 4Plots.mm,
                  titlefontsize = 8)
      scatter!(p, result.data.times[s], result.data.values[s],
                  label = "Data",
                  markersize = 1.5,
                  alpha = 0.6,
                  color = :red)
      push!(plots_list, p)
  end
  
  if !isempty(plots_list)
      n_plots = length(plots_list)
      layout = (ceil(Int, n_plots/3), 3)
      combined = plot(plots_list..., 
                      layout = layout, 
                      size = (1400, 320 * layout[1]),
                      margin = 6Plots.mm)
      display(combined)
      savefig(combined, "linear_interpolation_predictions.png")
      println("  ✓ Saved: linear_interpolation_predictions.png")
  end
  
  return plots_list
end

"""
Compare linear interpolation with MTGP results - FIXED
"""
function compare_linear_vs_mtgp(linear_result::LinearInterpolationResult, mtgp_result, sensor_indices::Vector{Int}=[1,2,3])
  println("\n[Comparison] Linear vs MTGP...")
  
  plots_list = []
  for s in sensor_indices[1:min(length(sensor_indices), linear_result.data.S)]
      x_test = range(minimum(linear_result.data.times[s]), maximum(linear_result.data.times[s]), length=200)
      x_test_vec = collect(Float32, x_test)  # Convert to Float32 explicitly
      
      # Linear predictions
      μ_linear, σ_linear = linear_predict(linear_result, s, x_test_vec)
      
      # MTGP predictions
      μ_mtgp, σ_mtgp = icm_predict(mtgp_result, s, x_test_vec)
      
      p = plot(x_test, μ_linear,
                  label = "Linear",
                  xlabel = "Time (hours)",
                  ylabel = "Value",
                  title = "$(linear_result.sensor_names[s]) - Linear vs MTGP",
                  linewidth = 2,
                  color = :blue,
                  legend = :topright,
                  size = (600, 400),
                  margin = 4Plots.mm)
      
      plot!(p, x_test, μ_mtgp,
              label = "MTGP",
              linewidth = 2,
              color = :green)
      
      plot!(p, x_test, μ_mtgp,
              ribbon = 1.96 .* σ_mtgp,
              fillalpha = 0.2,
              color = :green,
              label = "MTGP 95% CI")
      
      scatter!(p, linear_result.data.times[s], linear_result.data.values[s],
                  label = "Data",
                  markersize = 2,
                  alpha = 0.7,
                  color = :red)
      
      push!(plots_list, p)
  end
  
  if !isempty(plots_list)
      combined = plot(plots_list..., 
                      layout = (length(plots_list), 1),
                      size = (800, 400 * length(plots_list)))
      display(combined)
      savefig(combined, "linear_vs_mtgp_comparison.png")
      println("  ✓ Saved: linear_vs_mtgp_comparison.png")
  end
  
  return plots_list
end

# Example usage function that can be added to your main script
"""
Example of how to use linear interpolation with your existing code - FIXED
"""
function run_linear_interpolation_example(data::MultiSensorData)
  println("\n" * "="^70)
  println("Running Linear Interpolation Example")
  println("="^70)
  
  # Train linear interpolation model
  linear_result = train_linear_interpolation(data; verbose=true)
  
  # Generate visualizations
  visualize_linear_predictions(linear_result; max_sensors=9)
  
  # Generate 1-minute resolution predictions
  predictions = generate_1min_resolution_predictions_linear(linear_result)
  
  # Print summary
  println("\n[Linear] Summary:")
  println("="^70)
  println("  Total sensors: $(linear_result.data.S)")
  println("  Sensor types: $(unique(linear_result.data.sensor_types))")
  println("  Time range: $(round(minimum(vcat(linear_result.data.times...)), digits=2)) to $(round(maximum(vcat(linear_result.data.times...)), digits=2)) hours")
  println("  1-min predictions generated for all sensors")
  
  return linear_result, predictions
end