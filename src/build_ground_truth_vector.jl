function build_ground_truth_vector(power_data_a, power_data_b, power_data_c,
                                   lack_a::Vector{Int}, lack_b::Vector{Int}, lack_c::Vector{Int},
                                   time_step::Int=432000, n_total_nodes::Int=132)
    """
    按照与观测矩阵相同的顺序构造真实值向量
    
    数据假设：
    - power_data_a[i, :] 对应 Node i （132行，包含Node 1）
    """
    
    ground_truth = Float64[]
    
    for bus in 2:n_total_nodes
        # 直接用 bus 作为索引（因为数据矩阵有 132 行）
        data_idx = bus  # ← 改这里！
        
        # A 相
        if !(bus in lack_a)
            val = power_data_a[data_idx, time_step]
            # if abs(val) < 1e-6
            #     @warn "Node $bus Phase A 的值为 0: power_data_a[$data_idx, $time_step] = $val"
            # end
            push!(ground_truth, val)
        end
        
        # B 相
        if !(bus in lack_b)
            val = power_data_b[data_idx, time_step]
            # if abs(val) < 1e-6
            #     @warn "Node $bus Phase B 的值为 0: power_data_b[$data_idx, $time_step] = $val"
            # end
            push!(ground_truth, val)
        end
        
        # C 相
        if !(bus in lack_c)
            val = power_data_c[data_idx, time_step]
            # if abs(val) < 1e-6
            #     @warn "Node $bus Phase C 的值为 0: power_data_c[$data_idx, $time_step] = $val"
            # end
            push!(ground_truth, val)
        end
    end
    
    return ground_truth
end
