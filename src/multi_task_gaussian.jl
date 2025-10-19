
# Shared mean network
function create_shared_mean_network(input_dim=1, hidden_dim=32)
    return Chain(
        Dense(input_dim, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim, relu),
        Dense(hidden_dim, 1)
    )
end

# Task correlation matrix B = L * L'
function build_task_correlation_matrix(L_params::Matrix{Float32})
    L = LowerTriangular(L_params)
    Matrix(L * L')
end

# RBF kernel
function rbf_kernel(t1::Vector{Float32}, t2::Vector{Float32}, σ::Float32, ℓ::Float32)
    Δ = t1 .- t2'
    @. (σ^2) * exp( - (Δ^2) / (2f0 * ℓ^2) )
end

# Construct joint covariance K_all and add per-task noise inside construction (non-inplace)
# K((s,x),(t,x′)) = B_{st} k_time(x,x′) + δ_{st} k_local_s(x,x′) + δ_{st} σ_noise[s]^2 I
function construct_joint_K_all(times::Vector{Vector{Float32}},
                               B::Matrix{Float32},
                               σ_time::Float32, ℓ_time::Float32,
                               σ_locals::Vector{Float32}, ℓ_locals::Vector{Float32},
                               σ_noise::Vector{Float32}, jitter::Float32)
    S = length(times)
    row_blocks = map(1:S) do s
        reduce(hcat, map(1:S) do t
            K_block = B[s, t] .* rbf_kernel(times[s], times[t], σ_time, ℓ_time)
            if s == t
                K_block = K_block .+ rbf_kernel(times[s], times[t], σ_locals[s], ℓ_locals[s])
                K_block = add_diag_copy(K_block, σ_noise[s]^2 + jitter)
            end
            K_block
        end)
    end
    reduce(vcat, row_blocks)
end

# Build joint dataset in normalized space
function build_joint_dataset(data::MultiSensorData)
    norm_data, norm_params = normalize_multisensor_data(data)
    x_all = vcat(norm_data.times...)
    y_all = vcat(norm_data.values...)
    n_per_task = length.(norm_data.times)
    offsets = cumsum(vcat(0, n_per_task[1:end-1]))
    return (; norm_data, norm_params, x_all, y_all, n_per_task, offsets)
end

# Mean forward
mean_forward(mean_func, x_all::Vector{Float32}) = vec(mean_func(reshape(x_all, 1, :)))

# Joint NLL
function joint_nll_icm(params, joint_pack; jitter::Float32=1f-5)
    mean_func    = params.mean_func
    L_params     = params.L_params
    log_σ_time   = params.log_σ_time
    log_ℓ_time   = params.log_ℓ_time
    log_σ_locals = params.log_σ_locals
    log_ℓ_locals = params.log_ℓ_locals
    log_σ_noise  = params.log_σ_noise

    norm_data   = joint_pack.norm_data
    x_all       = joint_pack.x_all
    y_all       = joint_pack.y_all

    S = norm_data.S

    B = build_task_correlation_matrix(L_params)
    σ_time = exp(log_σ_time[])
    ℓ_time = exp(log_ℓ_time[])
    σ_locals = exp.(log_σ_locals)
    ℓ_locals = exp.(log_ℓ_locals)
    σ_noise  = exp.(log_σ_noise)

    # Build K_all with noise inside (no in-place modifications)
    K_all = construct_joint_K_all(norm_data.times, B, σ_time, ℓ_time, σ_locals, ℓ_locals, σ_noise, jitter)

    m_all = mean_forward(mean_func, x_all)
    r = y_all .- m_all

    K_sym = Hermitian(0.5f0 .* (K_all .+ K_all'))
    L = cholesky(K_sym).L
    α = L' \ (L \ r)
    N = length(y_all)

    0.5f0 * dot(r, α) + sum(log.(diag(L))) + 0.5f0 * N * log(2f0 * Float32(pi))
end

# Train ICM MTGP
function train_icm_mtgp(data::MultiSensorData; num_epochs::Int=200, lr::Float64=0.01, verbose::Bool=true)
    joint_pack = build_joint_dataset(data)
    S = data.S

    mean_func = create_shared_mean_network(1, 32)

    # Initialize params
    L_params = Matrix{Float32}(I, S, S)          # B = I initially
    log_σ_time = Float32[log(1.0)]
    log_ℓ_time = Float32[log(0.5)]
    log_σ_locals = fill(Float32(log(0.3)), S)
    log_ℓ_locals = fill(Float32(log(0.3)), S)
    log_σ_noise  = fill(Float32(log(0.1)), S)

    params = (
        mean_func = mean_func,
        L_params = L_params,
        log_σ_time = log_σ_time,
        log_ℓ_time = log_ℓ_time,
        log_σ_locals = log_σ_locals,
        log_ℓ_locals = log_ℓ_locals,
        log_σ_noise = log_σ_noise
    )

    ps = Flux.params(mean_func, L_params, log_σ_time, log_ℓ_time, log_σ_locals, log_ℓ_locals, log_σ_noise)
    opt = Flux.Adam(lr)

    losses = Float32[]
    best = Inf32
    patience = 30
    stall = 0

    println("\n" * "="^70)
    println("Training ICM/LMC Multi-task GP (Joint LML)")
    println("="^70)

    @showprogress for epoch in 1:num_epochs
        gs = Flux.gradient(ps) do
            joint_nll_icm(params, joint_pack; jitter=1f-5)
        end
        Flux.update!(opt, ps, gs)
        loss = joint_nll_icm(params, joint_pack; jitter=1f-5)
        push!(losses, loss)

        if verbose && (epoch % 10 == 0 || epoch == 1)
            println("Epoch $epoch, NLL = $(round(loss, digits=4))")
        end

        if loss + 1e-5 < best
            best = loss
            stall = 0
        else
            stall += 1
            if stall >= patience && epoch > 50
                verbose && println("Early stopping at epoch $epoch")
                break
            end
        end
    end

    # For reporting in original scale (optional)
    σ_time_final = exp(log_σ_time[]) * mean(joint_pack.norm_params.y_stds)
    ℓ_time_final = exp(log_ℓ_time[]) * joint_pack.norm_params.x_std
    σ_locals_final = exp.(log_σ_locals) .* joint_pack.norm_params.y_stds
    ℓ_locals_final = exp.(log_ℓ_locals) .* joint_pack.norm_params.x_std
    σ_noise_final  = exp.(log_σ_noise)  .* joint_pack.norm_params.y_stds

    return (
        mean_func = mean_func,
        L_params = L_params,
        log_σ_time = log_σ_time,
        log_ℓ_time = log_ℓ_time,
        log_σ_locals = log_σ_locals,
        log_ℓ_locals = log_ℓ_locals,
        log_σ_noise  = log_σ_noise,
        σ_time = σ_time_final,
        ℓ_time = ℓ_time_final,
        σ_locals = σ_locals_final,
        ℓ_locals = ℓ_locals_final,
        σ_noise = σ_noise_final,
        losses = losses,
        norm_params = joint_pack.norm_params,
        joint_pack = joint_pack,
        data = data
    )
end

