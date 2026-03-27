#############################
# 128-Core Optimized Parallel Monte Carlo
# 针对128核CPU + 377GB内存优化
#############################

using Distributed

# 为128核CPU优化工作进程数量
if nprocs() == 1
    # 策略：使用100个工作进程（保留28核给系统和其他任务）
    # 每个工作进程约占用 ~500MB，总计 ~50GB，非常安全
    n_workers = 100
    addprocs(n_workers)
    println("✅ Added $n_workers worker processes for 128-core CPU")
    println("������ System Memory: 377 GB (Estimated usage: ~50-80 GB)")
    println("������ Total processes: $(nprocs()) (1 master + $n_workers workers)")
    println("������ Reserved cores for system: 28")
else
    println("ℹ️  Using existing $(nworkers()) worker processes")
end

# 显示系统信息
println("\n������ System Information:")
println("   Total CPU cores: 128")
println("   Active workers: $(nworkers())")
println("   Total RAM: 377 GB")
println("   Estimated memory usage: ~$(nworkers() * 0.5) GB")
println("   Memory safety margin: $(377 - nworkers() * 0.5) GB")
println("   Expected speedup: ~$(nworkers())x")

# 在所有进程上加载必要的包和文件
@everywhere begin
    include("../src/implement_data.jl")
    include("../src/generate_series_data.jl")
    include("../ios/read_mat.jl")
    include("../src/extract_requested_dataset_multibatch.jl")
    include("../src/build_complete_multisensor_data.jl")
    include("../src/data_processing.jl")
    include("../src/multi_task_gaussian.jl")
    include("../src/gaussian_prediction.jl")
    include("../src/linear_imputation.jl")
    include("../src/missing_data_evaluation.jl")
    include("../data/sensor_location.jl")
    include("../src/get_topology.jl")
    include("../src/build_admittance_matrix.jl")
    include("../src/physic_constraint.jl")
    include("../src/calculate_matrix_uncertainty.jl")
    include("../src/build_observed_matrix_z.jl")
    include("../src/build_noise_precision_beta.jl")
    include("../src/matrix_completion.jl")
    
    using Flux
    using LinearAlgebra
    using Statistics
    using Random
end

using Plots
using ProgressMeter
using DataFrames
using JLD2
using MAT
using CSV
using Dates

#############################
# 辅助函数
#############################

@everywhere function add_gaussian_noise(data::MultiSensorData, noise_level::Float64; seed::Int=42)
    Random.seed!(seed)
    noisy_data = MultiSensorData(
        data.S,
        copy.(data.times),
        copy.(data.values),
        copy(data.sensor_names),
        copy(data.sensor_types)
    )
    
    for s in 1:data.S
        if !isempty(data.values[s])
            signal_std = std(data.values[s])
            noise_std = noise_level * signal_std
            noise = randn(length(data.values[s])) .* noise_std
            noisy_data.values[s] .+= noise
        end
    end
    
    return noisy_data
end

@everywhere function run_stage2_test_topology_aware(banch_data, daily_predictions)
    Y_bus = build_three_phase_admittance_matrix(banch_data)
    w = build_w_matrix(Y_bus)
    abs_w = abs.(w)
    M = build_matrix_M(Y_bus, w)
    K_mat = build_matrix_K(M, w)
    
    observed_matrix_Z, observed_pairs, monitored_obs = build_observed_matrix_Z(daily_predictions)
    noise_precision_β = build_noise_precision_beta(daily_predictions)
    
    tolerance = 1e-6
    c_param = 1e-7
    d_param = 1e-7
    max_iter = 400
    baseMVA = 10.0
    
    idx_P, idx_Q = 1, 2
    idx_Vr, idx_Vi, idx_V = 3, 4, 5
    
    svd_res = svd(observed_matrix_Z)
    r = 5
    U_r = svd_res.U[:, 1:r]
    Σ_r = svd_res.S[1:r]
    Vt_r = svd_res.Vt[1:r, :]
    
    sqrtD = Diagonal(sqrt.(Σ_r))
    A_mean = Array{Float64}(U_r * sqrtD)
    B_mean = Array{Float64}(Vt_r' * sqrtD)
    
    α = 1e-3
    Σa0 = α .* Matrix{Float64}(I, r, r)
    Σb0 = α .* Matrix{Float64}(I, r, r)
    Σa_list = [copy(Σa0) for _ in 1:size(A_mean, 1)]
    Σb_list = [copy(Σb0) for _ in 1:size(B_mean, 1)]
    γ = fill(1.0, r)
    
    X_old = Array{Float64}(A_mean * B_mean')
    latent_dim = size(A_mean, 2)
    
    observed_rows_Vr = Set{Int}()
    observed_rows_Vi = Set{Int}()
    observed_rows_V  = Set{Int}()
    
    for (r, col) in observed_pairs
        if col == idx_Vr push!(observed_rows_Vr, r)
        elseif col == idx_Vi push!(observed_rows_Vi, r)
        elseif col == idx_V  push!(observed_rows_V, r)
        end
    end
    
    augmented_pairs = copy(observed_pairs)
    augmented_Z = copy(observed_matrix_Z) 
    augmented_beta = copy(noise_precision_β)
    missing_indices = Vector{Tuple{Int, Int}}()
    beta_phys = 1e4 
    
    for i in 1:size(observed_matrix_Z, 1)
        if !(i in observed_rows_Vr)
            push!(augmented_pairs, (i, idx_Vr))
            push!(missing_indices, (i, idx_Vr))
            augmented_beta[i, idx_Vr] = beta_phys
            augmented_Z[i, idx_Vr] = X_old[i, idx_Vr]
        end
        if !(i in observed_rows_Vi)
            push!(augmented_pairs, (i, idx_Vi))
            push!(missing_indices, (i, idx_Vi))
            augmented_beta[i, idx_Vi] = beta_phys
            augmented_Z[i, idx_Vi] = X_old[i, idx_Vi]
        end
        if !(i in observed_rows_V)
            push!(augmented_pairs, (i, idx_V))
            push!(missing_indices, (i, idx_V))
            augmented_beta[i, idx_V] = beta_phys
            augmented_Z[i, idx_V] = X_old[i, idx_V]
        end
    end
    
    for it in 1:max_iter
        if it > 1
            for (r, c) in missing_indices
                augmented_Z[r, c] = X_old[r, c]
            end
        end
        
        for i in 1:size(A_mean, 1)
            βBtB = cal_beta_BTB_i(i, B_mean, Σb_list, augmented_pairs, augmented_beta, latent_dim)
            Σa_list[i] = cal_sigma_a_i(βBtB, γ)
            A_mean[i, :] = cal_a_mean_i(i, B_mean, Σa_list[i], augmented_pairs, augmented_beta, augmented_Z)
        end
        
        for j in 1:size(B_mean, 1)
            βAtA = cal_beta_ATA_j(j, A_mean, Σa_list, augmented_pairs, augmented_beta, latent_dim)
            Σb_list[j] = cal_sigma_b_j(βAtA, γ)
            B_mean[j, :] = cal_b_mean_j(j, A_mean, Σb_list[j], augmented_pairs, augmented_beta, augmented_Z)
        end
        
        for k in 1:length(γ)
            aTa = cal_aTa_i(k, A_mean, Σa_list)
            bTb = cal_bTb_j(k, B_mean, Σb_list)
            γ[k] = clamp((2 * c_param + size(A_mean,1) + size(B_mean,1)) / (aTa + bTb + 2 * d_param), 1e-6, 1e6)
        end
        
        X_new = Array{Float64}(A_mean * B_mean')
        
        try
            P_est = X_new[:, idx_P] ./ baseMVA
            Q_est = X_new[:, idx_Q] ./ baseMVA
            
            if length(P_est) * 2 == size(K_mat, 2)
                PQ_vec = vcat(P_est, Q_est)
                v_complex_phys = w + M * PQ_vec
                v_mag_phys = abs_w + K_mat * PQ_vec
                
                for i in 1:size(X_new, 1)
                    if !(i in observed_rows_Vr)
                        X_new[i, idx_Vr] = real(v_complex_phys[i])
                    end
                    if !(i in observed_rows_Vi)
                        X_new[i, idx_Vi] = imag(v_complex_phys[i])
                    end
                    if !(i in observed_rows_V)
                        X_new[i, idx_V] = v_mag_phys[i]
                    end
                end
            end
        catch e
        end
        
        numerator = norm(X_new - X_old)
        denominator = max(norm(X_old), 1e-12)
        rel = numerator / denominator
        
        X_old = copy(X_new)
        
        if rel < tolerance
            break
        end
    end
    
    log_likelihood_proxy = 0.0
    for (r, c) in observed_pairs
        z_val = observed_matrix_Z[r, c]
        x_val = X_old[r, c]
        beta_val = noise_precision_β[r, c]
        log_likelihood_proxy -= 0.5 * beta_val * (z_val - x_val)^2
    end
    
    return (log_likelihood = log_likelihood_proxy, X = X_old)
end

@everywhere function run_single_trial(
    trial::Int,
    data::MultiSensorData,
    noise_level::Float64,
    missing_pct::Float64,
    mtgp_epochs::Int,
    branch_list,
    prob_list,
    correct_topology_idx::Int,
    top_k::Int,
    seed_offset::Int
)
    trial_seed = 42 + trial * 1000 + seed_offset
    worker_id = myid()
    
    noisy_data = add_gaussian_noise(data, noise_level, seed=trial_seed)
    missing_noisy_data, removed_times, removed_values = 
        create_missing_data(noisy_data, missing_pct, seed=trial_seed + 1)
    
    mtgp_result = train_icm_mtgp(
        missing_noisy_data; 
        num_epochs=mtgp_epochs, 
        lr=0.01, 
        verbose=false
    )
    
    daily_predictions = generate_daily_predictions(mtgp_result, 40000, 1)
    
    n_topologies = length(branch_list)
    log_likelihoods = zeros(Float64, n_topologies)
    
    for k in eachindex(branch_list)
        branch_data = branch_list[k]
        result_bmc = run_stage2_test_topology_aware(branch_data, daily_predictions)
        log_likelihoods[k] = result_bmc.log_likelihood
    end
    
    max_ll = maximum(log_likelihoods)
    unnormalized_probs = prob_list .* exp.(log_likelihoods .- max_ll)
    posterior_probs = unnormalized_probs ./ sum(unnormalized_probs)
    
    sorted_indices = sortperm(posterior_probs, rev=true)
    correct_rank = findfirst(x -> x == correct_topology_idx, sorted_indices)
    is_success = correct_rank <= top_k
    
    return Dict(
        "trial" => trial,
        "worker_id" => worker_id,
        "rank" => correct_rank,
        "posterior" => posterior_probs[correct_topology_idx],
        "max_posterior" => maximum(posterior_probs),
        "max_posterior_idx" => argmax(posterior_probs),
        "is_success" => is_success,
        "top_k_indices" => sorted_indices[1:top_k],
        "top_k_posteriors" => posterior_probs[sorted_indices[1:top_k]]
    )
end

#############################
# 128核优化的并行蒙特卡洛评估
#############################

function evaluate_topology_identification_monte_carlo_parallel(
    data::MultiSensorData;
    noise_levels::Vector{Float64}=[0.01, 0.05],
    missing_percentages::Vector{Float64}=[0.1, 0.2, 0.4, 0.6],
    mtgp_epochs::Int=200,
    branch_list,
    prob_list,
    correct_topology_idx::Int=24,
    n_trials::Int=20,
    top_k_percent::Float64=0.1,
    verbose::Bool=true
)
    n_noise = length(noise_levels)
    n_missing = length(missing_percentages)
    n_topologies = length(branch_list)
    top_k = max(1, Int(ceil(n_topologies * top_k_percent)))
    
    results = Dict{String, Any}()
    results["success_rate"] = zeros(Float64, n_noise, n_missing)
    results["mean_rank"] = zeros(Float64, n_noise, n_missing)
    results["mean_posterior"] = zeros(Float64, n_noise, n_missing)
    results["trial_details"] = Array{Vector{Dict}}(undef, n_noise, n_missing)
    results["timing"] = Dict{String, Float64}()
    
    println("\n" * "="^80)
    println("������ 128-CORE PARALLEL MONTE CARLO SIMULATION")
    println("="^80)
    println("������ Correct Topology Index: $correct_topology_idx")
    println("������ Candidate topologies: $n_topologies")
    println("������ Trials per condition: $n_trials")
    println("������ Worker processes: $(nworkers())")
    println("������ Hardware: 128 cores, 377 GB RAM")
    println("⚡ Expected speedup: ~$(nworkers())x")
    
    # 修正时间估算
    estimated_minutes = Int(ceil(n_noise * n_missing * n_trials * 30 / nworkers() / 60))
    println("⏱️  Estimated time: ~$estimated_minutes minutes (~$(round(estimated_minutes/60, digits=1)) hours)")
    println("="^80)
    
    total_start_time = time()
    
    for (i, noise_level) in enumerate(noise_levels)
        println("\n" * "="^70)
        noise_pct = Int(noise_level * 100)
        println("������ NOISE LEVEL: $(noise_pct)%")
        println("="^70)
        
        for (j, missing_pct) in enumerate(missing_percentages)
            println("\n" * "-"^70)
            missing_pct_int = Int(missing_pct * 100)
            println("❌ Missing Data: $(missing_pct_int)%")
            println("-"^70)
            
            seed_offset = i * 100 + j * 10
            condition_start_time = time()
            
            println("  ������ Launching $n_trials parallel trials...")
            prog = Progress(n_trials, desc="  ������ Progress: ", barlen=50)
            
            trial_results = pmap(1:n_trials, batch_size=1) do trial
                result = run_single_trial(
                    trial, data, noise_level, missing_pct,
                    mtgp_epochs, branch_list, prob_list,
                    correct_topology_idx, top_k, seed_offset
                )
                next!(prog)
                return result
            end
            
            condition_elapsed = time() - condition_start_time
            
            success_count = count(r -> r["is_success"], trial_results)
            ranks = [r["rank"] for r in trial_results]
            posteriors = [r["posterior"] for r in trial_results]
            
            success_rate = success_count / n_trials
            mean_rank = mean(ranks)
            mean_posterior = mean(posteriors)
            
            results["success_rate"][i, j] = success_rate
            results["mean_rank"][i, j] = mean_rank
            results["mean_posterior"][i, j] = mean_posterior
            results["trial_details"][i, j] = trial_results
            
            println("\n  ������ Results:")
            println("     ⏱️  Time: $(round(condition_elapsed/60, digits=1)) min")
            println("     ✅ Success: $(round(success_rate*100, digits=1))% ($success_count/$n_trials)")
            println("     ������ Mean Rank: $(round(mean_rank, digits=1))/$n_topologies")
            println("     ������ Best: $(minimum(ranks)) | ⚠️ Worst: $(maximum(ranks))")
        end
    end
    
    total_elapsed = time() - total_start_time
    results["timing"]["total_seconds"] = total_elapsed
    results["timing"]["total_hours"] = total_elapsed / 3600
    
    println("\n" * "="^80)
    println("������ FINAL SUMMARY")
    println("="^80)
    println("⏱️  Total Time: $(round(total_elapsed/3600, digits=2)) hours")
    
    println("\n������ Success Rate Matrix:")
    header = "Missing% \\ Noise%  |  " * join([lpad("$(Int(nl*100))%", 8) for nl in noise_levels], " | ")
    println(header)
    println("-"^80)
    for (j, mp) in enumerate(missing_percentages)
        row_str = lpad("$(Int(mp*100))%", 17) * " | "
        row_str *= join([lpad("$(round(results["success_rate"][i,j]*100, digits=1))", 8) 
                        for i in 1:n_noise], " | ")
        println(row_str)
    end
    
    return results
end

#############################
# 保存函数
#############################

function save_monte_carlo_results(results::Dict, filename::String)
    jld2_file = replace(filename, r".\w+$" => ".jld2")
    @save jld2_file results
    println("✅ Results saved to: $jld2_file")
    
    csv_file = replace(filename, r"\.\w+$" => "_summary.csv")
    df = DataFrame(
        Noise = Float64[],
        Missing = Float64[],
        SuccessRate = Float64[],
        MeanRank = Float64[],
        MeanPosterior = Float64[]
    )
    
    noise_levels = [0.01, 0.05]
    missing_percentages = [0.1, 0.2, 0.4, 0.6]
    
    for (i, noise) in enumerate(noise_levels)
        for (j, missing) in enumerate(missing_percentages)
            push!(df, (
                noise,
                missing,
                results["success_rate"][i, j],
                results["mean_rank"][i, j],
                results["mean_posterior"][i, j]
            ))
        end
    end
    
    CSV.write(csv_file, df)
    println("✅ Summary saved to: $csv_file")
end

#############################
# 主执行代码
#############################

println("\n" * "="^80)
println("������ 128-Core Optimized Monte Carlo Analysis")
println("="^80)

println("\n[1] ������ Loading data...")
pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases, pmu_sensors, scada_sensors, ami_sensors = FAD10_config()

(batch_data_1, batch_data_2, batch_data_3, batch_data_4,
 batch_data_5, batch_data_6, batch_data_7, batch_data_8,
 batch_data_9, batch_data_10, batch_data_11, batch_data_12,
 batch_data_13, batch_data_14, batch_data_15, batch_data_16,
 batch_data_17, batch_data_18) = read_mat()

ds = extract_requested_dataset_multibatch(
    (batch_data_1, batch_data_2, batch_data_3, batch_data_4,
     batch_data_5, batch_data_6, batch_data_7, batch_data_8,
     batch_data_9, batch_data_10, batch_data_11, batch_data_12,
     batch_data_13, batch_data_14, batch_data_15, batch_data_16,
     batch_data_17, batch_data_18);
    pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases
)

println("\n[2] ������️ Building dataset...")
data = build_complete_multisensor_data(
    ds;
    max_points_per_sensor = 300,
    pmu_sensors, scada_sensors, ami_sensors
)

println("\n[3] ������ Generating topologies...")
branch = read_topology_mat("/home/user/Downloads/Distribution_System_State_Estimation-main/pf_out/topology.mat")
branch_list, prob_list = generate_branch_list_with_prior(
    branch;
    param_sets = nothing,
    param_source_rows = (35, 5, 1),
    per_line_cartesian = true
)

println("\n[4] ������ Running 128-Core Parallel Monte Carlo...")
mc_results = evaluate_topology_identification_monte_carlo_parallel(
    data;
    noise_levels = [0.01, 0.05],
    missing_percentages = [0.1, 0.2, 0.4, 0.6],
    mtgp_epochs = 200,
    branch_list = branch_list,
    prob_list = prob_list,
    correct_topology_idx = 24,
    n_trials = 20,
    top_k_percent = 0.1,
    verbose = true
)

println("\n[5] ������ Saving results...")
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
save_monte_carlo_results(
    mc_results,
    "/home/user/topology_mc_128core_$(timestamp).jld2"
)

println("\n" * "="^80)
println("✅ 128-Core Parallel Monte Carlo Complete!")
println("⏱️  Total time: $(round(mc_results["timing"]["total_hours"], digits=2)) hours")
println("="^80)
