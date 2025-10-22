using SparseArrays

function fraction_available_data(Z::AbstractMatrix)
    total = length(Z)
    total == 0 && return 0.0f0
    available = count(!iszero, Z)
    return available / total
end

function cal_beta_BTB_i(i, B, Σb_list, observed_pairs, noise_precision_β, d)
    βBtB_i = zeros(Float64, d, d)
    
    # 只对第 i 行的观测位置求和
    for (row, col) in observed_pairs
        if row == i
            bj = B[col, :]  # 第 j 列的均值向量
            β_ij = noise_precision_β[i, col]  # 精度
            
            # 累加 β_ij * (bⱼbⱼᵀ + Σbⱼ)
            βBtB_i += β_ij * (bj * bj' + Σb_list[col])
        end
    end
    
    return βBtB_i
end

function cal_sigma_a_i(βBtB_i, γ)
    Σa_i_inv = Diagonal(γ) + βBtB_i
    return inv(Σa_i_inv)
end

function cal_a_mean_i(i, B, Σa_i, observed_pairs, noise_precision_β, observed_matrix_Z)
    d = size(B, 2)
    weighted_sum = zeros(Float64, d)
    
    # 对第 i 行的所有观测位置求和
    for (row, col) in observed_pairs
        if row == i
            bj = B[col, :]  # 第 j 列的均值向量
            β_ij = noise_precision_β[i, col]  # 精度
            z_ij = observed_matrix_Z[i, col]  # 观测值
            
            # 累加 β_ij * bⱼ * z_ij
            weighted_sum += β_ij * bj * z_ij
        end
    end
    
    # <aᵢ> = Σaᵢ * weighted_sum
    return Σa_i * weighted_sum
end

function cal_b_mean_j(j, A_mean, Σb_j, observed_pairs, noise_precision_β, observed_matrix_Z)
    d = size(A_mean, 2)
    weighted_sum = zeros(Float64, d)
    
    # 对第 j 列的所有观测位置求和
    for (row, col) in observed_pairs
        if col == j
            ai = A_mean[row, :]  # 第 i 行的均值向量
            β_ij = noise_precision_β[row, j]  # 精度
            z_ij = observed_matrix_Z[row, j]  # 观测值
            
            # 累加 β_ij * aᵢ * z_ij
            weighted_sum += β_ij * ai * z_ij
        end
    end
    
    # <bⱼ> = Σbⱼ * weighted_sum
    return Σb_j * weighted_sum
    
end

function cal_beta_ATA_j(j, A_mean, Σa_list, observed_pairs, noise_precision_β, d)
    βAtA_j = zeros(Float64, d, d)
    
    # 只对第 j 列的观测位置求和
    for (row, col) in observed_pairs
        if col == j
            ai = A_mean[row, :]  # 第 i 行的均值向量
            β_ij = noise_precision_β[row, j]  # 精度
            
            # 累加 β_ij * (aᵢaᵢᵀ + Σaᵢ)
            βAtA_j += β_ij * (ai * ai' + Σa_list[row])
        end
    end
    
    return βAtA_j
end

function cal_sigma_b_j(βAtA_j, γ)
    Σb_j_inv = Diagonal(γ) + βAtA_j
    return inv(Σb_j_inv)
    
end

function cal_aTa_i(i, A_mean, Σa_list)
    m = size(A_mean, 1)  # 行数
    
    # 提取第 i 维的所有均值
    a_i_means = A_mean[:, i]  # m×1 向量
    
    # 均值的平方和
    mean_squared_sum = dot(a_i_means, a_i_means)  # = aᵢᵀaᵢ
    
    # 方差的和
    variance_sum = sum(Σa_list[k][i, i] for k in 1:m)
    
    return mean_squared_sum + variance_sum
end

function cal_bTb_j(i, B_mean, Σb_list)
    n = size(B_mean, 1)  # 行数
    
    # 提取第 i 维的所有均值
    b_i_means = B_mean[:, i]  # n×1 向量
    
    # 均值的平方和
    mean_squared_sum = dot(b_i_means, b_i_means)  # = bᵢᵀbᵢ
    
    # 方差的和
    variance_sum = sum(Σb_list[k][i, i] for k in 1:n)
    
    return mean_squared_sum + variance_sum
end


function parse_branch(branch::Matrix{Float64})
    M = size(branch, 1)
    f = Vector{Int}(branch[:, 1])
    t = Vector{Int}(branch[:, 2])
    R = branch[:, 3]
    X = branch[:, 4]
    status = size(branch, 2) >= 11 ? Vector{Int}(branch[:, 11]) : ones(Int, M)
    return f, t, R, X, status
end

function build_spanning_tree_from_branch(branch::Matrix{Float64}; root::Int=1)
    f, t, R_all, X_all, status = parse_branch(branch)
    idx_on = findall(status .> 0)
    f = f[idx_on]; t = t[idx_on]; R_all = R_all[idx_on]; X_all = X_all[idx_on]
    M = length(f)
    n = max(maximum(f), maximum(t))
    adj = [Int[] for _ in 1:n]
    for i in 1:M
        push!(adj[f[i]], t[i])
        push!(adj[t[i]], f[i])
    end
    visited = falses(n)
    q = [root]; visited[root] = true
    edges_tree = Tuple{Int,Int}[]
    while !isempty(q)
        u = popfirst!(q)
        for v in adj[u]
            if !visited[v]
                visited[v] = true
                push!(edges_tree, (u, v))
                push!(q, v)
            end
        end
    end
    if !all(visited)
        @warn "图不连通：仅对可达分量构建树。"
    end
    edge_idx_all = Dict{Tuple{Int,Int}, Int}()
    for i in 1:M
        edge_idx_all[(f[i], t[i])] = i
        edge_idx_all[(t[i], f[i])] = i
    end
    return edges_tree, n, edge_idx_all, R_all, X_all
end

function build_Ain_T_Abr_from_tree(edges_tree::Vector{Tuple{Int,Int}}, n::Int)
    m = length(edges_tree)
    AI = Int[]; AJ = Int[]; AV = Float64[]
    for (j, (u, v)) in enumerate(edges_tree)
        push!(AI, u); push!(AJ, j); push!(AV,  1.0)
        push!(AI, v); push!(AJ, j); push!(AV, -1.0)
    end
    Ain = sparse(AI, AJ, AV, n, m)
    children = [Int[] for _ in 1:n]
    for (u, v) in edges_tree
        push!(children[u], v)
    end
    TI = Int[]; TJ = Int[]; TV = Float64[]
    function collect_subtree(start_v)
        stack = [start_v]; nodes = Int[]
        while !isempty(stack)
            x = pop!(stack)
            push!(nodes, x)
            append!(stack, children[x])
        end
        return nodes
    end
    for (ℓ, (_u, v)) in enumerate(edges_tree)
        sub = collect_subtree(v)
        for b in sub
            push!(TI, ℓ); push!(TJ, b); push!(TV, 1.0)
        end
    end
    T = sparse(TI, TJ, TV, m, n)
    A_br = -sparse(transpose(Ain))
    return Ain, T, A_br
end

function map_Rx_to_tree(edges_tree::Vector{Tuple{Int,Int}},
                        edge_idx_all::Dict{Tuple{Int,Int},Int},
                        R_all::Vector{Float64},
                        X_all::Vector{Float64})
    m = length(edges_tree)
    R_line = zeros(Float64, m)
    X_line = zeros(Float64, m)
    for (ℓ, e) in enumerate(edges_tree)
        k = edge_idx_all[e]
        R_line[ℓ] = R_all[k]
        X_line[ℓ] = X_all[k]
    end
    return spdiagm(0 => R_line), spdiagm(0 => X_line)
end

function build_M_from_branch(branch::Matrix{Float64}; root::Int=1)
    edges_tree, n, edge_idx_all, R_all, X_all = build_spanning_tree_from_branch(branch; root=root)
    Ain, T, A_br = build_Ain_T_Abr_from_tree(edges_tree, n)
    R_line, X_line = map_Rx_to_tree(edges_tree, edge_idx_all, R_all, X_all)
    Mv = A_br
    Mp = 1.0 .* (R_line * T)
    Mq = 1.0 .* (X_line * T)
    return (Ain=Ain, T=T, A_br=A_br, edges=edges_tree, R_line=R_line, X_line=X_line, Mv=Mv, Mp=Mp, Mq=Mq)
end

function cal_beta_BTB_i(i, B, Σb_list, observed_pairs, noise_precision_β, d)
    βBtB_i = zeros(Float64, d, d)
    for (row, col) in observed_pairs
        if row == i
            bj = B[col, :]
            β_ij = noise_precision_β[i, col]
            βBtB_i += β_ij * (bj * bj' + Σb_list[col])
        end
    end
    return βBtB_i
end

function cal_sigma_a_i(βBtB_i, γ)
    Σa_i_inv = Diagonal(γ) + βBtB_i
    return inv(Σa_i_inv)
end

function cal_a_mean_i(i, B, Σa_i, observed_pairs, noise_precision_β, observed_matrix_Z)
    d = size(B, 2)
    weighted_sum = zeros(Float64, d)
    for (row, col) in observed_pairs
        if row == i
            bj = B[col, :]
            β_ij = noise_precision_β[i, col]
            z_ij = observed_matrix_Z[i, col]
            weighted_sum += β_ij * bj * z_ij
        end
    end
    return Σa_i * weighted_sum
end

function cal_beta_ATA_j(j, A_mean, Σa_list, observed_pairs, noise_precision_β, d)
    βAtA_j = zeros(Float64, d, d)
    for (row, col) in observed_pairs
        if col == j
            ai = A_mean[row, :]
            β_ij = noise_precision_β[row, j]
            βAtA_j += β_ij * (ai * ai' + Σa_list[row])
        end
    end
    return βAtA_j
end

function cal_sigma_b_j(βAtA_j, γ)
    Σb_j_inv = Diagonal(γ) + βAtA_j
    return inv(Σb_j_inv)
end

function cal_b_mean_j(j, A_mean, Σb_j, observed_pairs, noise_precision_β, observed_matrix_Z)
    d = size(A_mean, 2)
    weighted_sum = zeros(Float64, d)
    for (row, col) in observed_pairs
        if col == j
            ai = A_mean[row, :]
            β_ij = noise_precision_β[row, j]
            z_ij = observed_matrix_Z[row, j]
            weighted_sum += β_ij * ai * z_ij
        end
    end
    return Σb_j * weighted_sum
end

function cal_aTa_i(i, A_mean, Σa_list)
    m = size(A_mean, 1)
    a_i_means = A_mean[:, i]
    mean_squared_sum = dot(a_i_means, a_i_means)
    variance_sum = sum(Σa_list[k][i, i] for k in 1:m)
    return mean_squared_sum + variance_sum
end

function cal_bTb_j(i, B_mean, Σb_list)
    n = size(B_mean, 1)
    b_i_means = B_mean[:, i]
    mean_squared_sum = dot(b_i_means, b_i_means)
    variance_sum = sum(Σb_list[k][i, i] for k in 1:n)
    return mean_squared_sum + variance_sum
end

# -----------------------
# Physics projection (only V) with blending
# -----------------------
function project_X_with_linphys(
    X::AbstractMatrix{<:Real},
    idx_P::Int, idx_Q::Int, idx_V::Int,
    Mv::AbstractMatrix{<:Real},
    Mp::AbstractMatrix{<:Real},
    Mq::AbstractMatrix{<:Real},
    v0::Real;
    λ_reg::Real=1e-3,
    root::Int=1,
    Vref::Real=1.0,
    anchor::Bool=true,
    w_anchor::Real=120.0,
    η::Float64=0.35
)
    n = size(X, 1)
    m = size(Mv, 1)
    @assert size(Mv,2) == n
    @assert size(Mp) == (m,n)
    @assert size(Mq) == (m,n)

    PD = Array{Float64}(X[:, idx_P])
    QD = Array{Float64}(X[:, idx_Q])
    V  = Array{Float64}(X[:, idx_V])

    Mv_f = Array{Float64}(Mv)
    Mp_f = Array{Float64}(Mp)
    Mq_f = Array{Float64}(Mq)

    rhs = float(v0) .* ones(Float64, m) .- Mp_f*PD .- Mq_f*QD

    H = Mv_f' * Mv_f .+ float(λ_reg) .* I(n)
    b = Mv_f' * rhs
    if anchor
        H[root, root] += float(w_anchor)
        b[root] += float(w_anchor) * float(Vref)
    end
    V_new = H \ b
    # blended update
    X[:, idx_V] .= V .+ η .* (V_new .- V)
     println("root=$root, V_before=$(round(V[root],sigdigits=5)), V_new=$(round(V_new[root],sigdigits=5))")
    return X
end
