using LinearAlgebra
using SparseArrays
using Statistics

include("../src/build_observed_matrix_z.jl")
include("../src/build_noise_precision_beta.jl")
include("../src/implement_data.jl")
include("../src/matrix_completion.jl")
include("../ios/read_mat.jl")
include("../src/lindistflow.jl")
include("../src/power_flow_optimal.jl")
include("../src/likelihood_gaussian.jl")
include("../src/physic_constraint_opf.jl")

function run_stage2_test()
    branch = read_topology_mat("D:/luosipeng/matpower8.1/pf_parallel_out/topology.mat")
    daily_predictions = generate_daily_predictions(result, 1, 1)
    observed_matrix_Z, observed_pairs, monitored_obs = build_observed_matrix_Z(daily_predictions; monitor_buses=Set([8, 12]))
    noise_precision_β = build_noise_precision_beta(daily_predictions)

    tolerance = 1e-6
    c = 1e-7
    d = 1e-7
    max_iter = 400
    δ = 1e-4
    β_phys = 1e-4
    η= 0.2

    root_bus = 1
    Vr_ref = 1.0
    Vi_ref = 0.0

    observed_matrix_Z = Array{Float64}(observed_matrix_Z)
    noise_precision_β = Array{Float64}(noise_precision_β)

    X0 = copy(observed_matrix_Z)
    svd_res = svd(observed_matrix_Z)
    r = min(5, minimum(size(observed_matrix_Z)))
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
        # P_inj = X_new[:, 1]./10; Q_inj = X_new[:, 2]./10; Vb = X_new[:, 5]; 
        # Vr = X_new[:, 3]; Vi = X_new[:, 4];

        # V_sol, θ_sol, Pinj_sol, Qinj_sol, Vr_sol, Vi_sol =
        #     ac_nodal_injection(P_inj, Q_inj, Vb, branch, root_bus, Vref, 0.0, observed_pairs)
        X_refined = physic_constraint(X_new, branch, root_bus, Vr_ref, Vi_ref, observed_pairs, observed_matrix_Z, δ, β_phys, X0; verbose=false)
        X_new = η*X_refined + (1-η)*X_new

        numerator = norm(X_new - X_old)
        denominator = max(norm(X_old), 1e-12)
        rel = numerator / denominator
        push!(history[:rel_change], rel)
        X_old = X_new

        η = min(η * 1.5, 1)
        if rel < tolerance
            println("Converged at iter=$it, rel=$(rel)")
            break
        end
    end

    if isempty(history[:rel_change]) || history[:rel_change][end] ≥ tolerance
        @warn "Not below tolerance yet. tail(rel)=$(history[:rel_change][max(end-4,1):end])"
    end
    println("Loss =$( abs(monitored_obs[12][5] - X_old[12,5])/monitored_obs[12][5] * 100  + abs(monitored_obs[8][5] - X_old[8,5])/monitored_obs[8][5] * 100) %)")

    elbo_result = compute_elbo_with_physics(X_old, A_mean, B_mean, Σa_list, Σb_list, γ,
                                 observed_matrix_Z, noise_precision_β, observed_pairs)
    

    return (X = X_old, history = history,
            flows = (P = Pij_sol, Q = Qij_sol, V = V_sol),
            injections = (P = Pinj_sol, Q = Qinj_sol), elbo = elbo_result)
end

X, history, flows, injections, elbo = run_stage2_test()
# println("P(Z | X, β) = $(likelihood)")