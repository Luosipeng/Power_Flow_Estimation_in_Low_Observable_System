
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

function physic_constraint(X_new, branch, root_bus, Vr_ref, Vi_ref, observed_pairs, observed_matrix_Z, δ, β_phys, X0; verbose=true)
    P_inj = X_new[:, 1] ./ 10
    Q_inj = X_new[:, 2] ./ 10
    n = size(X_new, 1)

    Y = build_ybus(branch, n)
    G = real.(Y); B = imag.(Y)
    

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "sb", "yes")
    set_optimizer_attribute(model, "print_level", 0)
    set_optimizer_attribute(model, "tol", 1e-8)
    set_optimizer_attribute(model, "acceptable_tol", 1e-6)

    # observed sets
    ΩP = Set{Int}(); ΩQ = Set{Int}(); ΩV = Set{Int}(); ΩVr = Set{Int}(); ΩVi = Set{Int}()
    for (i, j) in observed_pairs
        j == 1 && push!(ΩP, i)
        j == 2 && push!(ΩQ, i)
        j == 3 && push!(ΩVr, i)
        j == 4 && push!(ΩVi, i)
        j == 5 && push!(ΩV, i)
    end

    @variable(model, Vr[1:n])
    @variable(model, Vi[1:n])
    @variable(model, V[1:n])
    @variable(model, Pinj_var[1:n] <= 0)
    @variable(model, Qinj_var[1:n] <= 0)
    @variable(model, Pg >= 0)
    @variable(model, Qg)

    # voltage magnitude bounds and definition
    Vmin2 = 0.85^2
    Vmax2 = 1.05^2
    for i in 1:n
        @NLconstraint(model, Vr[i]^2 + Vi[i]^2 >= Vmin2)
        @NLconstraint(model, Vr[i]^2 + Vi[i]^2 <= Vmax2)
        @NLconstraint(model, V[i] == sqrt(Vr[i]^2 + Vi[i]^2))
    end

    # reference bus and injections at root
    @constraint(model, Vr[root_bus] == Vr_ref)
    @constraint(model, Vi[root_bus] == Vi_ref)
    @constraint(model, Pinj_var[root_bus] == 0)
    @constraint(model, Qinj_var[root_bus] == 0)

    # measurement constraints (linear)
    for i in ΩV
        @constraint(model, V[i] - observed_matrix_Z[i, 5] <= δ)
        @constraint(model, V[i] - observed_matrix_Z[i, 5] >= -δ)
    end
    for i in ΩVr
        @constraint(model, Vr[i] - observed_matrix_Z[i, 3] <= δ)
        @constraint(model, Vr[i] - observed_matrix_Z[i, 3] >= -δ)
    end
    for i in ΩVi
        @constraint(model, Vi[i] - observed_matrix_Z[i, 4] <= δ)
        @constraint(model, Vi[i] - observed_matrix_Z[i, 4] >= -δ)
    end
    for i in ΩP
        @constraint(model, Pinj_var[i] - observed_matrix_Z[i, 1]/10 <= δ)
        @constraint(model, Pinj_var[i] - observed_matrix_Z[i, 1]/10 >= -δ)
    end
    for i in ΩQ
        @constraint(model, Qinj_var[i] - observed_matrix_Z[i, 2]/10 <= δ)
        @constraint(model, Qinj_var[i] - observed_matrix_Z[i, 2]/10 >= -δ)
    end

    # network equations (nonlinear, bilinear terms)
    Cg = zeros(Float64, n); Cg[root_bus] = 1.0
    @NLexpression(model, real_curr[i=1:n], sum(G[i,j]*Vr[j] - B[i,j]*Vi[j] for j=1:n))
    @NLexpression(model, imag_curr[i=1:n], sum(B[i,j]*Vr[j] + G[i,j]*Vi[j] for j=1:n))
    @NLexpression(model, Pcalc[i=1:n], Vr[i]*real_curr[i] + Vi[i]*imag_curr[i])
    @NLexpression(model, Qcalc[i=1:n], Vi[i]*real_curr[i] - Vr[i]*imag_curr[i])

    for i in 1:n
        @NLconstraint(model, Pcalc[i] - Pinj_var[i] - Cg[i]*Pg <= β_phys)
        @NLconstraint(model, Pcalc[i] - Pinj_var[i] - Cg[i]*Pg >= -β_phys)
        @NLconstraint(model, Qcalc[i] - Qinj_var[i] - Cg[i]*Qg <= β_phys)
        @NLconstraint(model, Qcalc[i] - Qinj_var[i] - Cg[i]*Qg >= -β_phys)
    end

    # Nonlinear objective (unify with NL)
    @NLobjective(model, Min, sum(
        (Pinj_var[i]*10 - X0[i,1])^2 +
        (Qinj_var[i]*10 - X0[i,2])^2 +
        (Vr[i] - X0[i,3])^2 +
        (Vi[i] - X0[i,4])^2 +
        (sqrt(Vr[i]^2 + Vi[i]^2) - X0[i,5])^2
    for i=1:n))

    # optional warm starts (help Ipopt)
    for i in 1:n
        set_start_value(Vr[i], 1.0)
        set_start_value(Vi[i], 0.0)
        set_start_value(Pinj_var[i], P_inj[i])
        set_start_value(Qinj_var[i], Q_inj[i])
    end
    set_start_value(Vr[root_bus], Vr_ref)
    set_start_value(Vi[root_bus], Vi_ref)
    set_start_value(Pg, max(0.0, -sum(P_inj)))  # 根据符号约定调整
    set_start_value(Qg, -sum(Q_inj))

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
    V_sol = value.(V)
    Pinj_sol = value.(Pinj_var)
    Qinj_sol = value.(Qinj_var)

    X_refined = [Pinj_sol.*10 Qinj_sol.*10 Vr_sol Vi_sol V_sol]

    if verbose
        println("Voltage deviation max = ", maximum(abs.(V_sol .- Vref)))
    end

    return X_refined
end