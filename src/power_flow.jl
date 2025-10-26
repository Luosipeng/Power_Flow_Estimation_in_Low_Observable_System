# using JuMP
# using Ipopt
# using LinearAlgebra
# using Gurobi

# function build_ybus(branch, n)
#     in_service = branch[:, 11] .== 1
#     Y = zeros(ComplexF64, n, n)
#     for row in eachrow(branch[in_service, :])
#         f = Int(row[1]); t = Int(row[2])
#         r = row[3]; x = row[4]
#         y = inv(r + im * x)
#         Y[f, f] += y
#         Y[t, t] += y
#         Y[f, t] -= y
#         Y[t, f] -= y
#     end
#     return Y
# end

# function ac_nodal_injection(P_inj, Q_inj, Vb, branch, root_bus, Vref, θref, observed_pairs; verbose=true)
#     n = length(P_inj)
#     Y = build_ybus(branch, n)
#     G = real.(Y); B = imag.(Y)

#     model = Model(Gurobi.Optimizer)
#     set_silent(model)

#     ΩP = Set{Int}(); ΩQ = Set{Int}(); ΩV = Set{Int}()
#     for (i, j) in observed_pairs
#         j == 1 && push!(ΩP, i)
#         j == 2 && push!(ΩQ, i)
#         j == 5 && push!(ΩV, i)
#     end

#     @variable(model, 0.85 <= V[1:n] <= 1.05)
#     @variable(model, θ[1:n])
#     @variable(model, Pinj[1:n] <= 0)
#     @variable(model, Qinj[1:n] <= 0)
#     @variable(model, Pgen[1:n] >= 0)
#     @variable(model, Qgen[1:n] >= 0)
#     @variable(model, mismatch_P[1:n])
#     @variable(model, mismatch_Q[1:n])

#     @constraint(model, V[root_bus] == Vref)
#     @constraint(model, θ[root_bus] == θref)
#     @constraint(model, Pinj[root_bus] == 0)
#     @constraint(model, Qinj[root_bus] == 0)

#     Cg = zeros(Float64, n); Cg[root_bus] = 1.0
#     for i in ΩV
#         @constraint(model, V[i] == Vb[i])
#     end
#     for i in ΩP
#         @constraint(model, Pinj[i] == P_inj[i])
#     end
#     for i in ΩQ
#         @constraint(model, Qinj[i] == Q_inj[i])
#     end

#     for i in 1:n
#         @constraint(model,
#             sum(V[i] * V[j] * (G[i, j] * cos(θ[i] - θ[j]) + B[i, j] * sin(θ[i] - θ[j])) for j in 1:n)
#             - Pinj[i] - Cg[i] * Pgen[i] == mismatch_P[i])
#         @constraint(model,
#             sum(V[i] * V[j] * (G[i, j] * sin(θ[i] - θ[j]) - B[i, j] * cos(θ[i] - θ[j])) for j in 1:n)
#             - Qinj[i] - Cg[i] * Qgen[i] == mismatch_Q[i])
#     end

#     @objective(model, Min,
#         sum(mismatch_P[i]^2 for i in 1:n) +
#         sum(mismatch_Q[i]^2 for i in 1:n))

#     optimize!(model)

#     if termination_status(model) == MOI.OPTIMAL
#         if verbose
#             println("Optimization converged successfully.")
#         end
#     else
#         if verbose
#             println("Warning: Optimization did not converge to optimality.")
#         end
#     end

#     V_sol = value.(V)
#     θ_sol = value.(θ)

#     Pinj_sol = zeros(Float64, n)
#     Qinj_sol = zeros(Float64, n)
#     for i in 1:n
#         Pinj_sol[i] = sum(V_sol[i] * V_sol[j] * (G[i, j] * cos(θ_sol[i] - θ_sol[j]) + B[i, j] * sin(θ_sol[i] - θ_sol[j])) for j in 1:n)
#         Qinj_sol[i] = sum(V_sol[i] * V_sol[j] * (G[i, j] * sin(θ_sol[i] - θ_sol[j]) - B[i, j] * cos(θ_sol[i] - θ_sol[j])) for j in 1:n)
#     end

#     if verbose
#         println("Voltage deviation max = ", maximum(abs.(V_sol .- Vref)))
#     end

#     return V_sol, θ_sol, Pinj_sol, Qinj_sol
# end

using JuMP
using Ipopt
using LinearAlgebra

function build_ybus(branch, n)
    in_service = branch[:, 11] .== 1
    Y = zeros(ComplexF64, n, n)
    for row in eachrow(branch[in_service, :])
        f = Int(row[1]); t = Int(row[2])
        r = row[3]; x = row[4]
        y = inv(r + im * x)
        Y[f, f] += y
        Y[t, t] += y
        Y[f, t] -= y
        Y[t, f] -= y
    end
    return Y
end

function ac_nodal_injection(P_inj, Q_inj, Vb, branch, root_bus, Vref, θref, observed_pairs; verbose=true)
    n = length(P_inj)
    Y = build_ybus(branch, n)
    G = real.(Y); B = imag.(Y)

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "sb", "yes")
    set_optimizer_attribute(model, "print_level", 0)

    ΩP = Set{Int}(); ΩQ = Set{Int}(); ΩV = Set{Int}()
    for (i, j) in observed_pairs
        j == 1 && push!(ΩP, i)
        j == 2 && push!(ΩQ, i)
        j == 5 && push!(ΩV, i)
    end
    unobsP = setdiff(1:n, collect(ΩP))
    unobsQ = setdiff(1:n, collect(ΩQ))
    unobsV = setdiff(1:n, collect(ΩV))

    @variable(model, Vr[1:n])
    @variable(model, Vi[1:n])
    @variable(model, Pinj_var[1:n] )
    @variable(model, Qinj_var[1:n] )
    @variable(model, Pg >= 0)
    @variable(model, Qg)

    Vmin2 = 0.85^2
    Vmax2 = 1.05^2
    for i in 1:n
        @NLconstraint(model, Vr[i]^2 + Vi[i]^2 >= Vmin2)
        @NLconstraint(model, Vr[i]^2 + Vi[i]^2 <= Vmax2)
    end

    @constraint(model, Vr[root_bus] == Vref)
    @constraint(model, Vi[root_bus] == 0.0)
    @constraint(model, Pinj_var[root_bus] == 0)
    @constraint(model, Qinj_var[root_bus] == 0)

    for i in ΩV
        @NLconstraint(model, Vr[i]^2 + Vi[i]^2 == Vb[i]^2)
    end

    for i in ΩP
        @constraint(model, Pinj_var[i] == P_inj[i])
    end
    for i in ΩQ
        @constraint(model, Qinj_var[i] == Q_inj[i])
    end

    Cg = zeros(Float64, n); Cg[root_bus] = 1.0

    @NLexpression(model, real_curr[i=1:n], sum(G[i, j] * Vr[j] - B[i, j] * Vi[j] for j in 1:n))
    @NLexpression(model, imag_curr[i=1:n], sum(B[i, j] * Vr[j] + G[i, j] * Vi[j] for j in 1:n))
    @NLexpression(model, Pcalc[i=1:n], Vr[i] * real_curr[i] + Vi[i] * imag_curr[i])
    @NLexpression(model, Qcalc[i=1:n], Vi[i] * real_curr[i] - Vr[i] * imag_curr[i])

    for i in 1:n
        @NLconstraint(model, Pcalc[i] - Pinj_var[i] - Cg[i] * Pg == 0)
        @NLconstraint(model, Qcalc[i] - Qinj_var[i] - Cg[i] * Qg == 0)
    end

    λ = 1e-1
    @objective(model, Min,
        λ * (sum(Pinj_var[i]^2 for i in unobsP) +
             sum(Qinj_var[i]^2 for i in unobsQ) 
             
             ))

    set_optimizer_attribute(model, "tol", 1e-8)
    set_optimizer_attribute(model, "acceptable_tol", 1e-6)

    for i in 1:n
        set_start_value(Vr[i], Vref * cos(θref))
        set_start_value(Vi[i], Vref * sin(θref))
        set_start_value(Pinj_var[i], P_inj[i])
        set_start_value(Qinj_var[i], Q_inj[i])
    end
    set_start_value(Pg, sum(P_inj))
    set_start_value(Qg, sum(Q_inj))

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL || termination_status(model) == MOI.LOCALLY_SOLVED
        if verbose
            println("Optimization converged successfully.")
        end
    else
        if verbose
            println("Warning: Optimization did not converge to optimality.")
        end
    end

    Vr_sol = value.(Vr)
    Vi_sol = value.(Vi)
    V_sol = sqrt.(Vr_sol .^ 2 .+ Vi_sol .^ 2)
    θ_sol = atan.(Vi_sol, Vr_sol)
    Pinj_sol = value.(Pinj_var)
    Qinj_sol = value.(Qinj_var)

    if verbose
        println("Voltage deviation max = ", maximum(abs.(V_sol .- Vref)))
    end

    return V_sol, θ_sol, Pinj_sol, Qinj_sol, Vr_sol, Vi_sol
end