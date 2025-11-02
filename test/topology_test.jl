include("../ios/read_mat.jl")
include("../src/get_topology.jl")
using MAT

branchAC, branchDC = read_topology_mat("C:/Users/PC/Desktop/paper_case/topology_results.mat")

# 示例：从指定行抽取三组参数，逐线笛卡尔积遍历，并输出概率
branch_list, prob_list = generate_branch_list_with_prior(
    branchAC;
    param_sets = nothing,                # 允许显式传 nothing
    param_source_rows = (35,5,1),
    per_line_cartesian = true
)

@info "Generated instances" length(branch_list)
@info "First 5 probs" prob_list[1:min(end,5)]


