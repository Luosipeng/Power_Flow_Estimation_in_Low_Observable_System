function build_observed_matrix_Z(daily_predictions::Dict{String, Any}, 
                                  lack_a::Vector{Int}, 
                                  lack_b::Vector{Int}, 
                                  lack_c::Vector{Int},
                                  baseMVA::Float64=10.0,
                                  n_total_nodes::Int=132)  # ✅ 新增参数：总节点数
    """
    构建观测矩阵 Z，只包含存在的相位（排除缺相和平衡节点）
    
    参数：
    - daily_predictions: MTGP 预测结果
    - lack_a, lack_b, lack_c: 缺相节点列表
    - n_total_nodes: 配电网总节点数（默认 132）
    
    返回：
    - observed_matrix_Z: 观测矩阵 (n_valid_phases × 5)
    - observed_pairs: 有观测的 (row, col) 对
    - monitored_obs: 观测的标准差
    - phase_info: 缺相信息和映射表
    """
    
    sensors = daily_predictions["sensors"]
    
    println("\n������ 构建观测矩阵:")
    println("="^80)
    println("  总节点数: $n_total_nodes")
    println("  缺失 A 相节点数: $(length(lack_a))")
    println("  缺失 B 相节点数: $(length(lack_b))")
    println("  缺失 C 相节点数: $(length(lack_c))")
    
    # 1. 构建 (bus, phase) → row_idx 的映射
    # 只为存在的相位分配行索引
    bus_phase_to_row = Dict{Tuple{Int, Int}, Int}()
    row_to_bus_phase = Dict{Int, Tuple{Int, Int}}()
    current_row = 1
    
    for bus in 1:n_total_nodes
        # 跳过平衡节点（节点 1）
        if bus == 1
            continue
        end
        
        # 检查每个相位是否存在
        for phase in 1:3
            is_missing = false
            
            if phase == 1 && bus in lack_a
                is_missing = true
            elseif phase == 2 && bus in lack_b
                is_missing = true
            elseif phase == 3 && bus in lack_c
                is_missing = true
            end
            
            # 只为存在的相位分配行索引
            if !is_missing
                bus_phase_to_row[(bus, phase)] = current_row
                row_to_bus_phase[current_row] = (bus, phase)
                current_row += 1
            end
        end
    end
    
    n_valid_phases = current_row - 1
    n_cols = 5  # [P, Q, Vr, Vi, Vmag]
    
    # 验证计算
    expected_phases = (n_total_nodes - 1) * 3 - length(lack_a) - length(lack_b) - length(lack_c)
    println("  有效相位数: $n_valid_phases")
    println("  预期: ($n_total_nodes - 1) × 3 - $(length(lack_a)) - $(length(lack_b)) - $(length(lack_c)) = $expected_phases")
    
    if n_valid_phases != expected_phases
        @warn "有效相位数不匹配！实际: $n_valid_phases, 预期: $expected_phases"
    else
        println("  ✅ 维度验证通过")
    end
    
    # 2. 初始化矩阵
    observed_matrix_Z = zeros(Float64, n_valid_phases, n_cols)
    observed_pairs = Tuple{Int, Int}[]
    monitored_obs = Dict{Tuple{Int,Int}, Float64}()
    
    # 3. 解析传感器
    sensor_parsed = Dict{String, Any}()
    sensors_skipped_missing_phase = 0
    sensors_skipped_slack_bus = 0
    
    for (sensor_name, sensor_data) in sensors
        parts = split(sensor_name, "-")
        if length(parts) < 4
            @warn "无法解析传感器名称: $sensor_name"
            continue
        end
        
        sensor_type = parts[1]
        bus_num = parse(Int, parts[2])
        phase_char = lowercase(parts[3][1])
        meas_type = join(parts[4:end], "-")
        
        # 跳过平衡节点
        if bus_num == 1
            sensors_skipped_slack_bus += 1
            continue
        end
        
        # 检查节点编号是否有效
        if bus_num < 1 || bus_num > n_total_nodes
            @warn "传感器 $sensor_name 的节点编号 $bus_num 超出范围 [1, $n_total_nodes]"
            continue
        end
        
        # 确定相位索引
        phase_idx = if phase_char == 'a'
            1
        elseif phase_char == 'b'
            2
        elseif phase_char == 'c'
            3
        else
            @warn "未知相位: $phase_char in $sensor_name"
            continue
        end
        
        # 检查该相位是否存在
        if !haskey(bus_phase_to_row, (bus_num, phase_idx))
            # 这是一个缺相位置，跳过
            sensors_skipped_missing_phase += 1
            continue
        end
        
        # 确定测量类型列索引
        col_idx = if meas_type == "P"
            1
        elseif meas_type == "Q"
            2
        elseif meas_type == "V_real"
            3
        elseif meas_type == "V_imag"
            4
        elseif meas_type == "Vmag"
            5
        else
            @warn "未知测量类型: $meas_type in $sensor_name"
            continue
        end
        
        sensor_parsed[sensor_name] = Dict(
            "bus" => bus_num,
            "phase" => phase_idx,
            "phase_char" => phase_char,
            "col" => col_idx,
            "data" => sensor_data
        )
    end
    
    println("\n  传感器解析统计:")
    println("    总传感器数: $(length(sensors))")
    println("    有效传感器: $(length(sensor_parsed))")
    println("    跳过（平衡节点）: $sensors_skipped_slack_bus")
    println("    跳过（缺相位置）: $sensors_skipped_missing_phase")
    
    # 4. 填充观测数据
    sensors_with_data = 0
    sensors_zero_load = 0
    
    for (sensor_name, info) in sensor_parsed
        bus_num = info["bus"]
        phase_idx = info["phase"]
        col_idx = info["col"]
        sensor_data = info["data"]
        
        # 获取行索引
        row_idx = bus_phase_to_row[(bus_num, phase_idx)]
        
        # 检查是否有预测数据
        is_zero = get(sensor_data, "is_zero_load", false)
        pred_mean = sensor_data["prediction_mean"]
        pred_std = sensor_data["prediction_std"]
        
        if !is_zero && !isempty(pred_mean)
            # 有真实观测数据
            observed_matrix_Z[row_idx, col_idx] = pred_mean[1]
            push!(observed_pairs, (row_idx, col_idx))
            monitored_obs[(row_idx, col_idx)] = pred_std[1]
            sensors_with_data += 1
            
        else
                
            if col_idx in [1, 2]  # P, Q
                observed_matrix_Z[row_idx, col_idx] = 0.0
                push!(observed_pairs, (row_idx, col_idx))
                monitored_obs[(row_idx, col_idx)] = 0.001
            end
            
            sensors_zero_load += 1
        end
    end
    
    # 5. 构建缺相信息字典
    phase_info = Dict(
        "lack_a" => lack_a,
        "lack_b" => lack_b,
        "lack_c" => lack_c,
        "n_total_nodes" => n_total_nodes,
        "n_valid_phases" => n_valid_phases,
        "bus_phase_to_row" => bus_phase_to_row,
        "row_to_bus_phase" => row_to_bus_phase
    )
    
    println("\n������ 观测矩阵构建完成:")
    println("  有数据的传感器: $sensors_with_data")
    println("  零负荷传感器: $sensors_zero_load")
    println("  矩阵维度: $(size(observed_matrix_Z))")
    println("  观测点数: $(length(observed_pairs))")
    println("  观测率: $(round(length(observed_pairs) / (n_valid_phases * n_cols) * 100, digits=2))%")
    println("="^80)
    
    # normalize observed_matrix_Z
    # observed_matrix_Z[:,1] .= observed_matrix_Z[:,1] ./ (1000.0 * baseMVA)
    # observed_matrix_Z[:,2] .= observed_matrix_Z[:,2] ./ (1000.0 * baseMVA)

    return observed_matrix_Z, observed_pairs, monitored_obs, phase_info
end
