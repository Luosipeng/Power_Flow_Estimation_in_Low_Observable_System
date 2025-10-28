using LinearAlgebra

"""
在当前 X0 基础上，解小型凸问题：
  minimize  ||X - X0||_F^2
  s.t.      ||(X - Z)_Ω||_2 ≤ δ
            ||A_phys * vec(X) - b_phys||_2 ≤ β
            可选：Vmin ≤ X[:, idxV] ≤ Vmax

入参：
  X0::Matrix{Float64}
  Z::Matrix{Float64}
  Ω::Vector{Tuple{Int,Int}}
  A_phys::Matrix{Float64}
  b_phys::Vector{Float64}
  δ::Float64, β::Float64
  voltage_bounds::Union{Nothing,Tuple{Float64,Float64,Int}}

出参：
  X_refined::Matrix{Float64}
"""
function refine_with_embedded_constraints(X0::Matrix{Float64},
                                          Z::Matrix{Float64},
                                          Ω::Vector{Tuple{Int,Int}},
                                          A_phys::Matrix{Float64},
                                          b_phys::Vector{Float64};
                                          δ::Float64=0.05,
                                          β::Float64=0.5,
                                          voltage_bounds::Union{Nothing,Tuple{Float64,Float64,Int}}=nothing)

    n, m = size(X0)
    X = Variable(n, m)

    # 观测残差符号向量：用 vcat 逐项拼接
    function obs_vec_symbolic(Xvar)
        expr = nothing
        for (i,j) in Ω
            term = Xvar[i,j] - Z[i,j]   # Convex 标量表达式
            expr = expr === nothing ? term : vcat(expr, term)
        end
        # 若 Ω 为空，返回零长度常量向量，避免 norm 出错
        expr === nothing && return Constant(zeros(Float64, 0))
        return expr
    end

    r_obs  = obs_vec_symbolic(X)
    r_phys = A_phys * vec(X) - b_phys  # 仿射符号向量

    cons = Constraint[]
    push!(cons, norm(r_obs, 2) <= δ)
    push!(cons, norm(r_phys, 2) <= β)

    if voltage_bounds !== nothing
        Vmin, Vmax, idxV = voltage_bounds
        for i in 1:n
            push!(cons, X[i, idxV] >= Vmin)
            push!(cons, X[i, idxV] <= Vmax)
        end
    end

    # 目标：||X - X0||_F^2 使用 sumsquares
    obj = sumsquares(X - X0)
    problem = minimize(obj, cons)
    solve!(problem, SCS.Optimizer; silent_solver=true)

    return X.value
end

