using JuMP
using Gurobi
using Ipopt

function lindistflow(P_inj, Q_inj, Vb, branch, root_bus, Vref, observed_pairs; verbose=true)
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    # model = Model(Ipopt.Optimizer)
    # set_optimizer_attribute(model, "sb", "yes")
    # set_optimizer_attribute(model, "print_level", 0)  # Output verbosity
    # set_optimizer_attribute(model, "max_iter", 3000)  # Maximum iterations
    # set_optimizer_attribute(model, "tol", 1e-6)       # Convergence tolerance

    n = length(Vb)
    in_service = branch[:, 11] .== 1
    R = branch[in_service, 3]
    X = branch[in_service, 4]
    m = length(R)

    A, _ = build_incidence_matrix_td(n, branch)
    parents = [findfirst(>(0.5), A[e, :])::Union{Int,Nothing} for e in 1:m]
    childs  = [findfirst(<(-0.5), A[e, :])::Union{Int,Nothing} for e in 1:m]

    # 观测集合
    ΩP = Set{Int}(); ΩQ = Set{Int}(); ΩV = Set{Int}()
    for (i, j) in observed_pairs
        j == 1 && push!(ΩP, i)
        j == 2 && push!(ΩQ, i)
        j == 5 && push!(ΩV, i)
    end
    unobsP = setdiff(1:n, collect(ΩP))
    unobsQ = setdiff(1:n, collect(ΩQ))

    Cg = zeros(Float64, n); Cg[root_bus] = 1.0

    # 关键：注入允许正负；去掉方向盒约束
    @variable(model, Pinj[1:n])
    @variable(model, Qinj[1:n])
    @variable(model, V[1:n])
    @variable(model, Pij[1:m])
    @variable(model, Qij[1:m])
    @variable(model, Pg)
    @variable(model, Qg)

    # 电压盒与根锚定
    @constraint(model, 0.85 .<= V)
    @constraint(model, V .<= 1.05)
    @constraint(model, V[root_bus] == Vref)

    # 仅绑定观测
    for i in ΩP; @constraint(model, Pinj[i] == P_inj[i]); end
    for i in ΩQ; @constraint(model, Qinj[i] == Q_inj[i]); end
    for i in ΩV; @constraint(model, V[i]    == Vb[i]);    end

    # 统一号的 KCL（注入为正）
    @constraint(model, A' * Pij + Pinj .- Cg .* Pg .== 0)
    @constraint(model, A' * Qij + Qinj .- Cg .* Qg .== 0)

    # 逐边线性电压递推（硬等式）
    for e in 1:m
        u = parents[e]; v = childs[e]
        (u === nothing || v === nothing) && continue
        uu = u::Int; vv = v::Int
        @constraint(model, V[vv] - V[uu] == -2 * (R[e] * Pij[e] + X[e] * Qij[e]))
    end

    # 极小正则，稳定未观测注入
    @objective(model, Min, 1e-6 * (sum(Pinj[i]^2 for i in unobsP) + sum(Qinj[i]^2 for i in unobsQ)))

    optimize!(model)

    Pij_sol = value.(Pij)
    Qij_sol = value.(Qij)
    V_sol   = value.(V)
    Proot_sol = value(Pg)
    Qroot_sol = value(Qg)
    Pinj_sol  = value.(Pinj)
    Qinj_sol  = value.(Qinj)

    if verbose
        # 简要诊断：电压上升计数、相关性
        dV = Float64[]; drop = Float64[]; rises = 0
        for e in 1:m
            u = parents[e]; v = childs[e]
            (u === nothing || v === nothing) && continue
            uu = u::Int; vv = v::Int
            push!(dV, V_sol[uu] - V_sol[vv])
            push!(drop, 2 * (R[e]*Pij_sol[e] + X[e]*Qij_sol[e]))
            if V_sol[vv] > V_sol[uu] + 1e-6; rises += 1; end
        end
        c = cor(dV, drop)
        println("diag: corr(dV,2(rP+xQ))=$(round(c, digits=4)), V rises=$rises")
    end

    return Pij_sol, Qij_sol, V_sol, Proot_sol, Qroot_sol, Pinj_sol, Qinj_sol
end