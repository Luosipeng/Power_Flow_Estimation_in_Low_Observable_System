function compute_multitask_kernel(times_s::Vector{Float32},
                                   times_t::Vector{Float32},
                                   s::Int, t::Int,
                                   σ_g::Float32, ℓ_g::Float32,
                                   σ_s::Vector{Float32}, ℓ_s::Vector{Float32})
    
    Δt = times_s .- times_t'
    K_global = σ_g^2 .* exp.(-Δt.^2 ./ (2 * ℓ_g^2))
    
    if s == t
        K_local = σ_s[s]^2 .* exp.(-Δt.^2 ./ (2 * ℓ_s[s]^2))
        return K_global .+ K_local
    else
        return K_global
    end
end

function create_shared_mean_network(input_dim=1, hidden_dim=32)
    return Chain(
        Dense(input_dim, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim, relu),
        Dense(hidden_dim, 1)
    )
end

function compute_sensor_nll(times_s::Vector{Float32},
                            values_s::Vector{Float32},
                            s::Int,
                            mean_func,
                            σ_g::Float32, ℓ_g::Float32,
                            σ_s::Vector{Float32}, ℓ_s::Vector{Float32},
                            σ_noise::Vector{Float32};
                            jitter::Float32=1.0f-4)
    n = length(times_s)
    
    m = vec(mean_func(reshape(times_s, 1, :)))
    K = compute_multitask_kernel(times_s, times_s, s, s,
                                  σ_g, ℓ_g, σ_s, ℓ_s)
    K_noisy = K + (σ_noise[s]^2 + jitter) * I
    K_noisy = 0.5f0 .* (K_noisy .+ K_noisy')
    
    residual = values_s .- m
    L = cholesky(Hermitian(K_noisy)).L
    α = L' \ (L \ residual)
    
    nll = 0.5f0 * dot(residual, α) + 
          sum(log.(diag(L))) + 
          0.5f0 * n * log(2.0f0 * Float32(π))
    
    return nll
end

function train_multitask_gp(data::MultiSensorData;
                             num_epochs::Int=100,
                             lr::Float64=0.005,
                             verbose::Bool=true,
                             jitter::Float32=1.0f-4)
    
    S = data.S
    
    if verbose
        println("\nNormalizing data...")
    end
    norm_data, norm_params = normalize_multisensor_data(data)
    
    mean_func = create_shared_mean_network(1, 32)
    
    log_σ_g = Float32[log(1.0)]
    log_ℓ_g = Float32[log(0.5)]
    log_σ_s = Float32[log(0.3) for _ in 1:S]
    log_ℓ_s = Float32[log(0.3) for _ in 1:S]
    log_σ_noise = Float32[log(0.1) for _ in 1:S]
    
    if verbose
        println("\nInitial hyperparameters:")
        println("  Global: σ_g=$(round(exp(log_σ_g[1]), digits=4)), " *
                "ℓ_g=$(round(exp(log_ℓ_g[1]), digits=4))")
        println("  Jitter: $jitter")
        println("  Number of sensors: $S")
    end
    
    ps = Flux.params(mean_func, log_σ_g, log_ℓ_g, 
                     log_σ_s, log_ℓ_s, log_σ_noise)
    opt = Flux.Adam(lr)
    
    losses = Float32[]
    σ_g_history = Float32[]
    ℓ_g_history = Float32[]
    
    best_loss = Inf32
    patience = 20
    patience_counter = 0
    
    println("\n" * "="^70)
    println("Training Multi-task Gaussian Process")
    println("="^70)
    
    @showprogress for epoch in 1:num_epochs
        local total_loss
        
        try
            gs = Flux.gradient(ps) do
                σ_g = exp(log_σ_g[1])
                ℓ_g = exp(log_ℓ_g[1])
                σ_s = exp.(log_σ_s)
                ℓ_s = exp.(log_ℓ_s)
                σ_noise = exp.(log_σ_noise)
                
                total_nll = 0.0f0
                for s in 1:S
                    nll_s = compute_sensor_nll(
                        norm_data.times[s],
                        norm_data.values[s],
                        s, mean_func,
                        σ_g, ℓ_g, σ_s, ℓ_s, σ_noise;
                        jitter=jitter
                    )
                    total_nll += nll_s
                end
                
                total_loss = total_nll
                return total_loss
            end
            
            if !isnan(total_loss) && !isinf(total_loss) && total_loss > 0
                Flux.update!(opt, ps, gs)
                push!(losses, total_loss)
                push!(σ_g_history, exp(log_σ_g[1]))
                push!(ℓ_g_history, exp(log_ℓ_g[1]))
                
                if total_loss < best_loss
                    best_loss = total_loss
                    patience_counter = 0
                else
                    patience_counter += 1
                end
                
                if patience_counter >= patience && epoch > 30
                    if verbose
                        println("\nEarly stopping: no improvement for $patience epochs")
                    end
                    break
                end
            else
                if verbose
                    println("\nWarning: Invalid loss at epoch $epoch")
                end
                if !isempty(losses)
                    push!(losses, losses[end])
                    push!(σ_g_history, σ_g_history[end])
                    push!(ℓ_g_history, ℓ_g_history[end])
                end
            end
            
        catch e
            if verbose
                println("\nError at epoch $epoch: $e")
            end
            if !isempty(losses)
                push!(losses, losses[end])
                push!(σ_g_history, σ_g_history[end])
                push!(ℓ_g_history, ℓ_g_history[end])
            end
            continue
        end
        
        if verbose && (epoch % 10 == 0 || epoch == 1)
            println("\nEpoch $epoch/$num_epochs")
            println("  Loss: $(round(total_loss, digits=4))")
            println("  σ_g: $(round(exp(log_σ_g[1]), digits=4)), " *
                    "ℓ_g: $(round(exp(log_ℓ_g[1]), digits=4))")
        end
    end
    
    println("\n" * "="^70)
    println("Training Complete!")
    println("="^70)
    
    σ_g_final = exp(log_σ_g[1])
    ℓ_g_final = exp(log_ℓ_g[1]) * norm_params.x_std
    σ_s_final = exp.(log_σ_s) .* norm_params.y_stds
    ℓ_s_final = exp.(log_ℓ_s) .* norm_params.x_std
    σ_noise_final = exp.(log_σ_noise) .* norm_params.y_stds
    
    return (
        mean_func = mean_func,
        σ_g = σ_g_final,
        ℓ_g = ℓ_g_final,
        σ_s = σ_s_final,
        ℓ_s = ℓ_s_final,
        σ_noise = σ_noise_final,
        losses = losses,
        σ_g_history = σ_g_history,
        ℓ_g_history = ℓ_g_history,
        norm_params = norm_params,
        data = data,
        jitter = jitter
    )
end