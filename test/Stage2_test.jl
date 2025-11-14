using LinearAlgebra
using SparseArrays
using Statistics

include("../src/build_observed_matrix_z.jl")
include("../src/build_noise_precision_beta.jl")
include("../src/implement_data.jl")
include("../src/matrix_completion.jl")
include("../ios/read_mat.jl")
include("../src/power_flow_optimal.jl")
include("../src/likelihood_gaussian.jl")
include("../src/ac_dc_power_flow.jl")

function run_stage2_test()
    branchAC, branchDC = read_topology_mat("C:/Users/PC/Desktop/paper_case/topology_results.mat")
    daily_predictions = generate_daily_predictions(result, 1, 1)
    observed_matrix_Z, observed_pairs, monitored_obs = build_observed_matrix_Z(daily_predictions,monitor_buses = Set([8,12]))
    noise_precision_β = build_noise_precision_beta(daily_predictions)

    tolerance = 1e-6
    c = 1e-7
    d = 1e-7
    max_iter = 400

    root_bus = 1
    Vref = 1.0
    inv_bus = 18
    rec_bus = 1
    eta = 0.9
    nac = 33
    ndc = 4

    observed_matrix_Z = Array{Float64}(observed_matrix_Z)
    noise_precision_β = Array{Float64}(noise_precision_β)

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

    history = Dict{Symbol, Vector{Float64}}(
        :rel_change => Float64[]
    )

    Pij_sol = Float64[]
    Qij_sol = Float64[]
    V_sol   = Float64[]
    Pinj_sol = Float64[]
    Qinj_sol = Float64[]

    for it in 1:max_iter
        for i in 1:size(A_mean, 1)
            βBtB = cal_beta_BTB_i(i, B_mean, Σb_list, observed_pairs, noise_precision_β, latent_dim)
            Σa_list[i] = cal_sigma_a_i(βBtB, γ)
            A_mean[i, :] = cal_a_mean_i(i, B_mean, Σa_list[i], observed_pairs, noise_precision_β, observed_matrix_Z)
        end
        for j in 1:size(B_mean, 1)
            βAtA = cal_beta_ATA_j(j, A_mean, Σa_list, observed_pairs, noise_precision_β, latent_dim)
            Σb_list[j] = cal_sigma_b_j(βAtA, γ)
            B_mean[j, :] = cal_b_mean_j(j, A_mean, Σb_list[j], observed_pairs, noise_precision_β, observed_matrix_Z)
        end
        for k in 1:length(γ)
            aTa = cal_aTa_i(k, A_mean, Σa_list)
            bTb = cal_bTb_j(k, B_mean, Σb_list)
            γ[k] = clamp((2c + size(A_mean,1) + size(B_mean,1)) / (aTa + bTb + 2d), 1e-6, 1e6)
        end

        X_new = Array{Float64}(A_mean * B_mean')
        P_inj = X_new[:, 1]./100; Q_inj = X_new[:, 2]./100; Vb = X_new[:, 5]; 
        Vbr = X_new[:, 3]; Vbi = X_new[:, 4];

         Vr_ac_sol, Vi_ac_sol, V_ac_sol, Pinj_ac_sol, Qinj_ac_sol, V_dc_sol, Pinj_dc_sol = ac_dc_power_flow(branchAC,branchDC, nac, ndc, P_inj, Q_inj, Vb, Vbr, Vbi, root_bus, 1, 1.00833, inv_bus, rec_bus, eta, Vref, observed_pairs, true)


        X_new[:, 5] .= vcat(V_ac_sol, V_dc_sol)
        X_new[:, 1] .= vcat(Pinj_ac_sol.*100, Pinj_dc_sol.*100)
        X_new[:, 2] .= vcat(Qinj_ac_sol.*100, zeros(length(Pinj_dc_sol)).*100)
        X_new[:, 3] .= vcat(Vr_ac_sol, V_dc_sol)
        X_new[:, 4] .= vcat(Vi_ac_sol, zeros(length(V_dc_sol)))

        numerator = norm(X_new - X_old)
        denominator = max(norm(X_old), 1e-12)
        rel = numerator / denominator

        println("Iter $it: rel_change = $rel")
        push!(history[:rel_change], rel)
        X_old = X_new

        if rel < tolerance
            println("Converged at iter=$it, rel=$(rel)")
            break
        end
    end

    if isempty(history[:rel_change]) || history[:rel_change][end] ≥ tolerance
        @warn "Not below tolerance yet. tail(rel)=$(history[:rel_change][max(end-4,1):end])"
    end
    println("Loss =$( abs(monitored_obs[12][5] - X_old[12,5])/monitored_obs[12][5] * 100  + abs(monitored_obs[8][5] - X_old[8,5])/monitored_obs[8][5] * 100) %)")
    

    return (X = X_old, history = history,
            flows = (P = Pij_sol, Q = Qij_sol, V = V_sol),
            injections = (P = Pinj_sol, Q = Qinj_sol))
end

X, history, flows, injections = run_stage2_test()
# println("P(Z | X, β) = $(likelihood)")
# 结构后验分布
