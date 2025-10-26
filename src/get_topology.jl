# 列索引定义（MATPOWER 风格）
const COL_F = 1
const COL_T = 2
const COL_R = 3
const COL_X = 4
const COL_STATUS = 11

# 无向边判等
same_edge(u::Int, v::Int, a::Int, b::Int) = (u==a && v==b) || (u==b && v==a)

# 在矩阵中查找无向边(u,v)对应的所有行索引（并行线返回多个）
function find_branch_indices(mat::AbstractMatrix, u::Int, v::Int)
    idx = Vector{Int}()
    for i in 1:size(mat,1)
        f = Int(mat[i, COL_F])
        t = Int(mat[i, COL_T])
        if same_edge(u,v,f,t)
            push!(idx, i)
        end
    end
    return idx
end

# 默认的 8 种拓扑（每个元素为“开断边”的集合）
function default_topologies() :: Vector{Set{Tuple{Int,Int}}}
    return [
        Set([(8,21), (9,15), (12,22), (18,33), (25,29)]),
        Set([(9,10), (8,21), (9,15), (18,33), (25,29)]),
        Set([(7,8), (9,10), (12,22), (18,33), (25,29)]),
        Set([(7,8), (9,10), (8,21), (18,33), (25,29)]),
        Set([(9,10), (14,15), (8,21), (9,15), (25,29)]),
        Set([(7,8), (14,15), (8,21), (12,22), (25,29)]),
        Set([(7,8), (9,10), (32,33), (8,21), (25,29)]),
        Set([(9,10), (14,15), (32,33), (8,21), (25,29)])
    ]
end

"""
    generate_branch_list_with_prior(
        branch_based::AbstractMatrix;
        unknown_edges::Vector{Tuple{Int,Int}} = [(8,21),(9,15),(12,22),(18,33),(25,29),(29,30),(30,31),(31,32)],
        topologies_open::Vector{Set{Tuple{Int,Int}}} = default_topologies(),
        param_sets::Union{Nothing, Vector{Tuple{Float64,Float64}}} = nothing,
        param_source_rows::Union{Nothing,NTuple{3,Int}} = (35,5,1),
        per_line_cartesian::Bool = true
    ) -> Tuple{Vector{Matrix{Float64}}, Vector{Float64}}

生成所有“拓扑 × 参数组合”的分支矩阵列表，并返回对应先验概率。

先验设定
- 拓扑均匀先验：1/8
- 线路型号均匀先验：每条闭合未知线为三选一，且独立，先验为 (1/3)^M
  - 当 per_line_cartesian = true：M 为该拓扑下闭合且被遍历的未知线路条数
  - 当 per_line_cartesian = false：同一拓扑所有闭合未知线共享同一型号选择，仅 3 种选择，先验为 1/3

返回
- branch_list: Vector{Matrix{Float64}}
- prob_list: Vector{Float64}，与 branch_list 一一对应
"""
function generate_branch_list_with_prior(
    branch_based::AbstractMatrix;
    unknown_edges::Vector{Tuple{Int,Int}} = [(8,21),(9,15),(12,22),(18,33),(25,29),(29,30),(30,31),(31,32)],
    topologies_open::Vector{Set{Tuple{Int,Int}}} = default_topologies(),
    param_sets::Union{Nothing, Vector{Tuple{Float64,Float64}}} = nothing,
    param_source_rows::Union{Nothing,NTuple{3,Int}} = (35,5,1),
    per_line_cartesian::Bool = true
) :: Tuple{Vector{Matrix{Float64}}, Vector{Float64}}

    # 基本检查
    @assert size(branch_based, 2) >= COL_STATUS "branch_based 列数不足，至少需要包含第 $(COL_STATUS) 列"
    @assert length(topologies_open) > 0 "topologies_open 不能为空"

    # 参数来源：优先使用 param_sets，否则用 param_source_rows 从 branch_based 取值
    ps::Vector{Tuple{Float64,Float64}} = if param_sets === nothing
        @assert param_source_rows !== nothing "param_sets 和 param_source_rows 不能同时为 nothing"
        (i1,i2,i3) = param_source_rows
        @assert all(i -> 1 <= i <= size(branch_based,1), (i1,i2,i3)) "param_source_rows 中存在超出行范围的索引"
        [
            (float(branch_based[i1, COL_R]), float(branch_based[i1, COL_X])),
            (float(branch_based[i2, COL_R]), float(branch_based[i2, COL_X])),
            (float(branch_based[i3, COL_R]), float(branch_based[i3, COL_X])),
        ]
    else
        @assert length(param_sets) == 3 "param_sets 必须包含三组 (R,X)"
        [(float(r), float(x)) for (r,x) in param_sets]
    end

    # 建立未知边 -> 行索引列表映射（并行线返回多个索引）
    unknown_idx_list = Dict{Tuple{Int,Int}, Vector{Int}}()
    for (u,v) in unknown_edges
        unknown_idx_list[(u,v)] = find_branch_indices(branch_based, u, v)
    end

    # 生成笛卡尔积索引的工具
    function cartesian_indices(K::Int, choices::Vector{Int})
        if K == 0
            return [Int[]]
        else
            prev = cartesian_indices(K-1, choices)
            acc = Vector{Vector{Int}}()
            for p in prev, c in choices
                push!(acc, vcat(p, [c]))
            end
            return acc
        end
    end

    branch_list = Vector{Matrix{Float64}}()
    prob_list   = Vector{Float64}()

    # 先验：拓扑均匀
    topo_prior = 1.0 / length(topologies_open)
    model_choice_prior = 1.0 / 3.0

    # 遍历每个拓扑
    for open_set in topologies_open
        base_mat = copy(branch_based)

        # 设置开断/闭合状态：open_set -> 0，其余 -> 1
        for i in 1:size(base_mat,1)
            f = Int(base_mat[i, COL_F]); t = Int(base_mat[i, COL_T])
            if (f,t) in open_set || (t,f) in open_set
                base_mat[i, COL_STATUS] = 0.0
            else
                base_mat[i, COL_STATUS] = 1.0
            end
        end

        # 收集该拓扑下处于闭合状态的未知线路对应的行索引
        closed_unknown_rows = Int[]
        for (u,v) in unknown_edges
            idxs = get(unknown_idx_list, (u,v), Int[])
            if !isempty(idxs)
                append!(closed_unknown_rows, [i for i in idxs if base_mat[i, COL_STATUS] == 1.0])
            end
        end

        if isempty(closed_unknown_rows)
            # 没有需要遍历的未知线，概率就是拓扑先验
            push!(branch_list, base_mat)
            push!(prob_list, topo_prior)
            continue
        end

        if per_line_cartesian
            # 对每条闭合未知线独立选 3 种参数（笛卡尔积）
            M = length(closed_unknown_rows)
            all_pick_lists = cartesian_indices(M, [1,2,3])
            # 每个组合的先验：P = topo_prior * (1/3)^M
            p_each = topo_prior * (model_choice_prior ^ M)
            for picks in all_pick_lists
                mat = copy(base_mat)
                for (pos, irow) in enumerate(closed_unknown_rows)
                    (R, X) = ps[picks[pos]]
                    mat[irow, COL_R] = R
                    mat[irow, COL_X] = X
                end
                push!(branch_list, mat)
                push!(prob_list, p_each)
            end
        else
            # 所有闭合未知线使用同一组参数（3 种选择）
            # 每个组合的先验：P = topo_prior * (1/3)
            p_each = topo_prior * model_choice_prior
            for (R, X) in ps
                mat = copy(base_mat)
                for irow in closed_unknown_rows
                    mat[irow, COL_R] = R
                    mat[irow, COL_X] = X
                end
                push!(branch_list, mat)
                push!(prob_list, p_each)
            end
        end
    end

    return branch_list, prob_list
end