function lack_phase_analysis(voltage_mag_a, voltage_mag_b, voltage_mag_c)
    """
    分析配电网的缺相情况（基于真实拓扑数据）
    
    参数：
    - voltage_mag_a, voltage_mag_b, voltage_mag_c: 三相电压幅值
    
    返回：
    - lack_a: 缺失 A 相的节点索引列表
    - lack_b: 缺失 B 相的节点索引列表
    - lack_c: 缺失 C 相的节点索引列表
    """
    vm_a = voltage_mag_a[:, 1]
    vm_b = voltage_mag_b[:, 1]
    vm_c = voltage_mag_c[:, 1]

    lack_a = findall(vm_a .== 0.0)
    lack_b = findall(vm_b .== 0.0)
    lack_c = findall(vm_c .== 0.0)
    
    # 打印统计信息
    n_nodes = length(vm_a)
    println("\n������ 配电网缺相分析（基于真实拓扑）:")
    println("="^80)
    println("  总节点数: $n_nodes")
    println("  缺失 A 相的节点数: $(length(lack_a))")
    println("  缺失 B 相的节点数: $(length(lack_b))")
    println("  缺失 C 相的节点数: $(length(lack_c))")
    
    if !isempty(lack_a)
        println("\n  缺失 A 相的节点: $lack_a")
    end
    if !isempty(lack_b)
        println("  缺失 B 相的节点: $lack_b")
    end
    if !isempty(lack_c)
        println("  缺失 C 相的节点: $lack_c")
    end
    
    # 统计每个节点的相位配置
    phase_config = Dict{String, Int}()
    for i in 1:n_nodes
        phases = String[]
        if vm_a[i] > 0.0 push!(phases, "a") end
        if vm_b[i] > 0.0 push!(phases, "b") end
        if vm_c[i] > 0.0 push!(phases, "c") end
        
        config = join(phases, "")
        if config == ""
            config = "none"
        end
        phase_config[config] = get(phase_config, config, 0) + 1
    end
    
    println("\n  节点相位配置统计:")
    for (config, count) in sort(collect(phase_config), by=x->x[2], rev=true)
        println("    $config: $count 个节点")
    end
    println("="^80)
    
    return lack_a, lack_b, lack_c
end
# lack_a, lack_b, lack_c = lack_phase_analysis(voltage_mag_a, voltage_mag_b, voltage_mag_c)