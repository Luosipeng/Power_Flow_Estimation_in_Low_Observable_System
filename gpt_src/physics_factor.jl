using LinearAlgebra
using SparseArrays

# -----------------------
# Prior variance helpers from factorization
# -----------------------
# Var(x_ij) ≈ aᵀ Σb_j a + bᵀ Σa_i b + tr(Σa_i Σb_j)
@views function approx_var_xij(i::Int, j::Int, A_mean, Σa_list, B_mean, Σb_list)
    a = A_mean[i, :]
    b = B_mean[j, :]
    Σa = Σa_list[i]
    Σb = Σb_list[j]
    return dot(a, Σb * a) + dot(b, Σa * b) + tr(Σa * Σb)
end

# y_diag_prior for y=[P;Q;Vr;Vi] or [P;Q;V] (choose cols via args)
@views function build_prior_diag_vars(A_mean, Σa_list, B_mean, Σb_list, X_old;
                               cols::NTuple{4,Int}=(1,2,3,4))
    n = size(X_old, 1)
    cP, cQ, cVr, cVi = cols
    μ = vcat(X_old[:, cP], X_old[:, cQ], X_old[:, cVr], X_old[:, cVi])
    vars = zeros(Float64, 4n)
    @inbounds for i in 1:n
        vars[i]        = max(approx_var_xij(i, cP,  A_mean, Σa_list, B_mean, Σb_list), 1e-8)
        vars[n + i]    = max(approx_var_xij(i, cQ,  A_mean, Σa_list, B_mean, Σb_list), 1e-8)
        vars[2n + i]   = max(approx_var_xij(i, cVr, A_mean, Σa_list, B_mean, Σb_list), 1e-8)
        vars[3n + i]   = max(approx_var_xij(i, cVi, A_mean, Σa_list, B_mean, Σb_list), 1e-8)
    end
    return μ, vars
end

@views function build_prior_diag_vars_lindistflow(A_mean, Σa_list, B_mean, Σb_list, X_old;
                                           cols::NTuple{3,Int}=(1,2,5))
    n = size(X_old, 1)
    cP, cQ, cV = cols
    μ = vcat(X_old[:, cP], X_old[:, cQ], X_old[:, cV])
    vars = zeros(Float64, 3n)
    @inbounds for i in 1:n
        vars[i]        = max(approx_var_xij(i, cP, A_mean, Σa_list, B_mean, Σb_list), 1e-8)
        vars[n + i]    = max(approx_var_xij(i, cQ, A_mean, Σa_list, B_mean, Σb_list), 1e-8)
        vars[2n + i]   = max(approx_var_xij(i, cV, A_mean, Σa_list, B_mean, Σb_list), 1e-8)
    end
    return μ, vars
end

# -----------------------
# Conditional Gaussian update (covariance form, stable, keeps UQ)
# -----------------------
# ...existing code...

function conditional_gaussian_update_cov(
    μ::Vector{Float64}, vars::Vector{Float64},
    C::AbstractMatrix{Float64}, d::Vector{Float64};
    λ_pf::Float64 = 1e6, jitter::Float64 = 1e-10, max_retry::Int = 8
)
    @assert size(C,2) == length(μ)
    m = size(C, 1)

    Σ_diag = max.(vars, 1e-10)
    Σ = Diagonal(Σ_diag)

    S = if isfinite(λ_pf) && λ_pf > 0
        Symmetric(Matrix(C * Σ * C') + (1/λ_pf) * I(m))
    else
        Symmetric(Matrix(C * Σ * C'))
    end

    I_m = Matrix{Float64}(I, m, m)
    jitter_eff = jitter
    F = nothing
    for attempt = 1:max_retry
        try
            S_plus = Symmetric(S + jitter_eff * I_m)
            F = cholesky(S_plus; check=true)
            break
        catch err
            jitter_eff *= 10
            if attempt == max_retry
                rethrow(err)
            end
        end
    end
    @assert F !== nothing

    r = d - C * μ
    z = F \ r
    z = F' \ z
    J = Matrix(Σ * C')
    μ_post = μ + J * z

    T = F \ (C * Matrix(Σ))
    T = F' \ T
    Σ_post = Symmetric(Matrix(Σ) - J * T)
    vars_post = clamp.(diag(Σ_post), 1e-12, Inf)
    return μ_post, vars_post
end

# ...existing code...
# -----------------------
# LinDistFlow sensitivities (radial)
# -----------------------
struct TreeTopo
    parent::Vector{Int}
    edge_id::Vector{Int}
    children::Vector{Vector{Int}}
end

function build_tree(n::Int, edges_from::Vector{Int}, edges_to::Vector{Int}, root::Int)
    adj = [Int[] for _ in 1:n]
    eid = Dict{Tuple{Int,Int},Int}()
    for (e,(u,v)) in enumerate(zip(edges_from, edges_to))
        push!(adj[u], v); push!(adj[v], u)
        eid[(u,v)] = e; eid[(v,u)] = e
    end
    parent = fill(0, n); edge_id = fill(0, n); children = [Int[] for _ in 1:n]
    q = [root]; parent[root] = root
    while !isempty(q)
        u = pop!(q)
        for v in adj[u]
            if parent[v] == 0
                parent[v] = u
                edge_id[v] = eid[(u,v)]
                push!(children[u], v)
                push!(q, v)
            end
        end
    end
    return TreeTopo(parent, edge_id, children)
end

function path_and_subtree_mats(n::Int, m::Int, topo::TreeTopo, root::Int)
    P = spzeros(Float64, n, m)
    S = spzeros(Float64, m, n)
    # build P
    for i in 1:n
        u = i
        while u != topo.parent[u]
            e = topo.edge_id[u]
            P[i, e] = 1.0
            u = topo.parent[u]
        end
    end
    # subtree sets (post-order from root)
    order = Int[]
    stack = [root]
    while !isempty(stack)
        u = pop!(stack)
        push!(order, u)
        append!(stack, topo.children[u])
    end
    in_subtree = [Set{Int}() for _ in 1:n]
    for u in reverse(order)
        if u != topo.parent[u] && topo.edge_id[u] != 0
            push!(in_subtree[u], u)
        end
        for v in topo.children[u]
            union!(in_subtree[u], in_subtree[v])
        end
    end
    for v in 1:n
        e = topo.edge_id[v]
        if e != 0
            for k in in_subtree[v]
                S[e, k] = 1.0
            end
        end
    end
    return P, S
end

# MATPOWER matrix extract [f,t,r,x,status]
function extract_edges_matpower(branch::AbstractMatrix; col_f::Int=1, col_t::Int=2, col_r::Int=3, col_x::Int=4, col_status::Int=11)
    mask = trues(size(branch,1))
    if 1 ≤ col_status ≤ size(branch,2)
        mask .= branch[:, col_status] .== 1
    end
    froms = Int.(round.(branch[mask, col_f]))
    tos   = Int.(round.(branch[mask, col_t]))
    rr    = Float64.(branch[mask, col_r])
    xx    = Float64.(branch[mask, col_x])
    return froms, tos, rr, xx
end

function build_lindistflow_H(n::Int, branch::AbstractMatrix, root::Int;
                             col_f::Int=1, col_t::Int=2, col_r::Int=3, col_x::Int=4, col_status::Int=11)
    froms, tos, rr, xx = extract_edges_matpower(branch; col_f=col_f, col_t=col_t, col_r=col_r, col_x=col_x, col_status=col_status)
    m = length(froms)
    topo = build_tree(n, froms, tos, root)
    P, S = path_and_subtree_mats(n, m, topo, root)
    Hr = P * spdiagm(0 => rr) * S
    Hx = P * spdiagm(0 => xx) * S
    return Matrix(Hr), Matrix(Hx)
end

# 高斯物理更新：LinDistFlow，y=[P;Q;V]
function physics_update_lindistflow(X_old, A_mean, Σa_list, B_mean, Σb_list,
                                    branch::AbstractMatrix, root::Int; Vref::Float64=1.0,
                                    λ_pf::Float64=1e8,
                                    colmap::NamedTuple = (P=1, Q=2, V=5),
                                    cols_branch=(1,2,3,4,11))
    n = size(X_old, 1)
    μ, vars = build_prior_diag_vars_lindistflow(A_mean, Σa_list, B_mean, Σb_list, X_old;
                                                cols=(colmap.P, colmap.Q, colmap.V))
    col_f, col_t, col_r, col_x, col_status = cols_branch
    Hr, Hx = build_lindistflow_H(n, branch, root; col_f=col_f, col_t=col_t, col_r=col_r, col_x=col_x, col_status=col_status)
    C = hcat(2.0 .* Hr, 2.0 .* Hx, Matrix{Float64}(I, n, n))
    d = fill(Vref, n)
    μ_post, vars_post = conditional_gaussian_update_cov(μ, vars, C, d; λ_pf=λ_pf)
    # 解包并回写
    X_new = copy(X_old)
    X_new[:, colmap.P] .= μ_post[1:n]
    X_new[:, colmap.Q] .= μ_post[n+1:2n]
    X_new[:, colmap.V] .= μ_post[2n+1:3n]
    return X_new, μ_post, vars_post
end

# -----------------------
# AC power balance physics (rectangular, linearize + Gaussian update)
# Variables y = [P; Q; Vr; Vi] (length 4n)
# Constraints: for i=1..n: Pcalc(V) - P = 0, Qcalc(V) - Q = 0
#              Vr[root] = Vref, Vi[root] = 0
# -----------------------
function build_ybus(branch::AbstractMatrix, n::Int; col_f::Int=1, col_t::Int=2, col_r::Int=3, col_x::Int=4, col_status::Int=11)
    in_service = trues(size(branch,1))
    if 1 ≤ col_status ≤ size(branch,2)
        in_service .= branch[:, col_status] .== 1
    end
    Y = zeros(ComplexF64, n, n)
    for row in eachrow(branch[in_service, :])
        f = Int(round(row[col_f])); t = Int(round(row[col_t]))
        r = row[col_r]; x = row[col_x]
        y = inv(r + im * x)
        Y[f,f] += y; Y[t,t] += y
        Y[f,t] -= y; Y[t,f] -= y
    end
    return Y
end

# g(y): stacked residuals [Pcalc - P; Qcalc - Q; Vr[root]-Vref; Vi[root]-0] (length 2n+2)
@views function ac_residual_and_jacobian_numeric(y::Vector{Float64}, G::Matrix{Float64}, B::Matrix{Float64},
                                          n::Int, root::Int, Vref::Float64; ϵ::Float64=1e-5)
    # unpack
    P = y[1:n]; Q = y[n+1:2n]
    Vr = y[2n+1:3n]; Vi = y[3n+1:4n]
    # compute currents and powers
    real_curr = similar(Vr)
    imag_curr = similar(Vr)
    @inbounds for i in 1:n
        s1 = 0.0; s2 = 0.0
        for j in 1:n
            s1 += G[i,j]*Vr[j] - B[i,j]*Vi[j]
            s2 += B[i,j]*Vr[j] + G[i,j]*Vi[j]
        end
        real_curr[i] = s1
        imag_curr[i] = s2
    end
    Pcalc = Vr .* real_curr .+ Vi .* imag_curr
    Qcalc = Vi .* real_curr .- Vr .* imag_curr
    # residual
    g = Vector{Float64}(undef, 2n + 2)
    g[1:n] .= Pcalc .- P
    g[n+1:2n] .= Qcalc .- Q
    g[2n+1] = Vr[root] - Vref
    g[2n+2] = Vi[root] - 0.0
    # numeric Jacobian wrt y (4n vars) by finite-diff
    m = length(g)
    J = zeros(Float64, m, 4n)
    # helper closure to recompute g given y
    function eval_g!(buf::Vector{Float64}, yy::Vector{Float64})
        P2 = yy[1:n]; Q2 = yy[n+1:2n]
        Vr2 = yy[2n+1:3n]; Vi2 = yy[3n+1:4n]
        @inbounds for i in 1:n
            s1 = 0.0; s2 = 0.0
            for j in 1:n
                s1 += G[i,j]*Vr2[j] - B[i,j]*Vi2[j]
                s2 += B[i,j]*Vr2[j] + G[i,j]*Vi2[j]
            end
            buf[i] = Vr2[i]*s1 + Vi2[i]*s2 - P2[i]
            buf[n+i] = Vi2[i]*s1 - Vr2[i]*s2 - Q2[i]
        end
        buf[2n+1] = Vr2[root] - Vref
        buf[2n+2] = Vi2[root]
        return nothing
    end
    # FD columns
    base = copy(y)
    g_base = copy(g)
    col = similar(g)
    for k in 1:4n
        yk = base[k]
        δ = ϵ * max(1.0, abs(yk))
        base[k] = yk + δ
        eval_g!(col, base)
        @inbounds @simd for i in 1:m
            J[i,k] = (col[i] - g_base[i]) / δ
        end
        base[k] = yk
    end
    return g, J
end

# 高斯物理更新：AC 线性化迭代，y=[P;Q;Vr;Vi]
function physics_update_ac_linearized(X_old, A_mean, Σa_list, B_mean, Σb_list,
                                      branch::AbstractMatrix, root::Int;
                                      Vref::Float64 = 1.0, λ_pf::Float64 = 1e6,
                                      iters::Int = 3,
                                      colmap::NamedTuple = (P=1, Q=2, Vr=3, Vi=4, V=5),
                                      cols_branch=(1,2,3,4,11))
    n = size(X_old, 1)
    # prior
    μ, vars = build_prior_diag_vars(A_mean, Σa_list, B_mean, Σb_list, X_old;
                                    cols=(colmap.P, colmap.Q, colmap.Vr, colmap.Vi))
    # bus admittance
    col_f, col_t, col_r, col_x, col_status = cols_branch
    Y = build_ybus(branch, n; col_f=col_f, col_t=col_t, col_r=col_r, col_x=col_x, col_status=col_status)
    G = real.(Y); B = imag.(Y)

    y = copy(μ); v = copy(vars)
    for _ in 1:iters
        g, J = ac_residual_and_jacobian_numeric(y, G, B, n, root, Vref)
        # Linearization: g(μ) + J (y - μ) ≈ 0 => J y = -g(μ) + J μ
        d = -g .+ J * y
        y, v = conditional_gaussian_update_cov(y, v, J, d; λ_pf=λ_pf)
    end

    # 回写到 X，更新 V = sqrt(Vr^2 + Vi^2)
    X_new = copy(X_old)
    X_new[:, colmap.P] .= y[1:n]
    X_new[:, colmap.Q] .= y[n+1:2n]
    Vr = y[2n+1:3n]; Vi = y[3n+1:4n]
    X_new[:, colmap.Vr] .= Vr
    X_new[:, colmap.Vi] .= Vi
    X_new[:, colmap.V]  .= sqrt.(max.(0.0, Vr.^2 .+ Vi.^2))
    return X_new, y, v
end