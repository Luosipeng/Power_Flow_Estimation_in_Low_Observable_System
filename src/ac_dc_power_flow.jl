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

function ac_dc_power_flow(branchAC,branchDC, nac, ndc, P_inj, Q_inj, Vb, root_bus, inv_bus, rec_bus, eta, Vref, observed_pairs, verbose)
    Yac = build_ybus(branchAC, nac)
    Ydc = build_ybus(branchDC, ndc)
    Gac = real.(Yac); Bac = imag.(Yac)
    Gdc = real.(Ydc); Bdc = imag.(Ydc)

    Vb_ac = Vb[1:nac]
    Vb_dc = Vb[nac+1:end]

    P_inj_ac = P_inj[1:nac]
    Q_inj_ac = Q_inj[1:nac]
    P_inj_dc = P_inj[nac+1:end]

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "sb", "yes")
    set_optimizer_attribute(model, "print_level", 0)

    ΩP_ac = Set{Int}(); ΩQ_ac = Set{Int}(); ΩV_ac = Set{Int}(); ΩVr_ac = Set{Int}(); ΩVi_ac = Set{Int}()
    ΩP_dc = Set{Int}(); ΩV_dc = Set{Int}(); ΩVr_dc = Set{Int}();
    for (i, j) in observed_pairs
        if i <= nac
            j == 1 && push!(ΩP_ac, i)
            j == 2 && push!(ΩQ_ac, i)
            j == 3 && push!(ΩVr_ac, i)
            j == 4 && push!(ΩVi_ac, i)
            j == 5 && push!(ΩV_ac, i)
        else
            idx_dc = i - nac
            j == 1 && push!(ΩP_dc, idx_dc)
            j == 3 && push!(ΩVr_dc, idx_dc)
            j == 5 && push!(ΩV_dc, idx_dc)
        end
    end

    unobsP = setdiff(1:nac, collect(ΩP_ac))
    unobsQ = setdiff(1:nac, collect(ΩQ_ac))
    unobsVr = setdiff(1:nac, collect(ΩVr_ac))
    unobsVi = setdiff(1:nac, collect(ΩVi_ac))
    unobsV = setdiff(1:nac, collect(ΩV_ac))
    unobsV_dc = setdiff(1:ndc, collect(ΩV_dc))
    unobsP_dc = setdiff(1:ndc, collect(ΩP_dc))

    @variable(model, Vr_ac[1:nac])
    @variable(model, Vi_ac[1:nac])
    @variable(model, Vr_dc[1:ndc])
    @variable(model, Pinj_var_ac[1:nac] <= 0 )
    @variable(model, Qinj_var_ac[1:nac] <= 0 )
    @variable(model, Pinj_var_dc[1:ndc])
    @variable(model, P_inv >= 0)
    @variable(model, Q_inv)
    @variable(model, P_rec >= 0)
    @variable(model, Pg >= 0)
    @variable(model, Qg)

    @constraint(model, Pinj_var_dc[1:2] .<= 0)
    # ===== patch===========
    @constraint(model, Vr_ac[18]^2 + Vi_ac[18]^2 .<= 1)
    # ======================

    Vmin2 = 0.85^2
    Vmax2 = 1.05^2
    for i in 1:nac
        @NLconstraint(model, Vr_ac[i]^2 + Vi_ac[i]^2 >= Vmin2)
        @NLconstraint(model, Vr_ac[i]^2 + Vi_ac[i]^2 <= Vmax2)
    end

    for i in 1:ndc
        @NLconstraint(model, Vr_dc[i]^2 >= Vmin2)
        @NLconstraint(model, Vr_dc[i]^2 <= Vmax2)
    end

    @constraint(model, Vr_ac[root_bus] == Vref)
    @constraint(model, Vi_ac[root_bus] == 0.0)
    @constraint(model, Pinj_var_ac[root_bus] == 0)
    @constraint(model, Qinj_var_ac[root_bus] == 0)

    # for i in ΩV_ac
    #     @NLconstraint(model, Vr_ac[i]^2 + Vi_ac[i]^2 == Vb_ac[i]^2)
    # end
    # for i in ΩV_dc
    #     @NLconstraint(model, Vr_dc[i] == Vb_dc[i])
    # end

    # for i in ΩP_ac
    #     @constraint(model, Pinj_var_ac[i] == P_inj_ac[i])
    # end
    # for i in ΩQ_ac
    #     @constraint(model, Qinj_var_ac[i] == Q_inj_ac[i])
    # end

    # for i in ΩP_dc
    #     @constraint(model, Pinj_var_dc[i] == P_inj_dc[i])
    # end

    Cg = zeros(Float64, nac); Cg[root_bus] = 1.0
    Cinv = zeros(Float64, nac); Cinv[inv_bus] = 1.0
    Crec = zeros(Float64, ndc); Crec[rec_bus] = 1.0

    @NLexpression(model, real_curr_ac[i=1:nac], sum(Gac[i, j] * Vr_ac[j] - Bac[i, j] * Vi_ac[j] for j in 1:nac))
    @NLexpression(model, imag_curr_ac[i=1:nac], sum(Bac[i, j] * Vr_ac[j] + Gac[i, j] * Vi_ac[j] for j in 1:nac))
    @NLexpression(model, Pcalc_ac[i=1:nac], Vr_ac[i] * real_curr_ac[i] + Vi_ac[i] * imag_curr_ac[i])
    @NLexpression(model, Qcalc_ac[i=1:nac], Vi_ac[i] * real_curr_ac[i] - Vr_ac[i] * imag_curr_ac[i])

    for i in 1:nac
        @NLconstraint(model, Pcalc_ac[i] - Pinj_var_ac[i] - Cg[i] * Pg + Cinv[i] * (P_inv - eta * P_rec) == 0)
        @NLconstraint(model, Qcalc_ac[i] - Qinj_var_ac[i] - Cg[i] * Qg + Cinv[i] * Q_inv == 0)
    end

    
    for i in 1:ndc
        @NLconstraint(model, Vr_dc[i] * sum(Gdc[i, j] * Vr_dc[j] for j in 1:ndc) - Pinj_var_dc[i] + Crec[i] * (P_rec - eta * P_inv) == 0)
    end

    @constraint(model, P_inv * P_rec == 0)

    @NLexpression(model, term_Vac, sum((Vr_ac[i]^2 + Vi_ac[i]^2 - Vb_ac[i]^2)^2 for i in ΩV_ac))
    @NLexpression(model, term_Pdc_meas, sum((Pinj_var_dc[i] - P_inj_dc[i])^2 for i in ΩP_dc))
    @NLexpression(model, term_Vdc, sum((Vr_dc[i] - Vb_dc[i])^2 for i in ΩV_dc))
    @NLexpression(model, term_Pac_meas, sum((Pinj_var_ac[i] - P_inj_ac[i])^2 for i in ΩP_ac))
    @NLexpression(model, term_Qac_meas, sum((Qinj_var_ac[i] - Q_inj_ac[i])^2 for i in ΩQ_ac))


    @NLobjective(model, Min,
        term_Vac + term_Pdc_meas + term_Vdc +
        term_Pac_meas + term_Qac_meas
    )

    set_optimizer_attribute(model, "tol", 1e-8)
    set_optimizer_attribute(model, "acceptable_tol", 1e-6)

    for i in 1:nac
        set_start_value(Vr_ac[i], 1)
        set_start_value(Vi_ac[i], 0)
        set_start_value(Pinj_var_ac[i], 0)
        set_start_value(Qinj_var_ac[i], 0)
    end

    for i in 1:ndc
        set_start_value(Vr_dc[i], 1)
        set_start_value(Pinj_var_dc[i], 0)
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

    Vr_ac_sol = value.(Vr_ac)
    Vi_ac_sol = value.(Vi_ac)
    V_ac_sol = sqrt.(Vr_ac_sol .^ 2 .+ Vi_ac_sol .^ 2)
    Pinj_ac_sol = value.(Pinj_var_ac)
    Qinj_ac_sol = value.(Qinj_var_ac)
    Vr_dc_sol = value.(Vr_dc)
    V_dc_sol = Vr_dc_sol
    Pinj_dc_sol = value.(Pinj_var_dc)


    return Vr_ac_sol, Vi_ac_sol, V_ac_sol, Pinj_ac_sol, Qinj_ac_sol, V_dc_sol, Pinj_dc_sol

end