function build_noise_precision_beta(daily_predictions::Dict{String, Any}, phase_info::Dict{String, Any})
    sensors = get(daily_predictions, "sensors", Dict{String, Any}())
    
    # ==========================================
    # 1. 维度与映射设置 (必须与观测矩阵一致)
    # ==========================================
    # 从 phase_info 获取真实的行数和行索引映射
    n_valid_phases = phase_info["n_valid_phases"]
    bus_phase_to_row = phase_info["bus_phase_to_row"] # (bus, phase) -> row_idx
    
    num_rows = n_valid_phases
    num_cols = 5
    
    column_map = Dict(
        "p" => 1, 
        "q" => 2, 
        "vreal" => 3, "v_real" => 3, # 增加兼容性防止格式微差
        "vimag" => 4, "v_imag" => 4,
        "vmag" => 5, "v_mag" => 5
    )
    
    phase_map = Dict("a" => 1, "b" => 2, "c" => 3)

    # 初始化 β 矩阵 (保持你的逻辑，默认为 0)
    # 建议使用 Float64 以保持精度一致
    β = zeros(Float64, num_rows, num_cols)

    for (name, data) in sensors
        # 示例名称: "SCADA-32-a-Vmag"
        parts = split(name, '-')
        length(parts) < 4 && continue 
        
        # 1. 解析节点索引
        node_idx = try
            parse(Int, parts[2])
        catch
            continue
        end
        
        # 排除平衡节点(Bus 1)
        if node_idx == 1 
            continue
        end
        
        # 2. 解析相位
        phase_str = lowercase(parts[3])
        haskey(phase_map, phase_str) || continue
        phase_offset = phase_map[phase_str] # 1, 2, or 3

        # ==========================================
        # 3. 关键修改：计算行索引 (Row Index Calculation)
        # ==========================================
        # 这里的逻辑必须改变：不能用公式计算，必须查表
        # 如果该节点的该相位是缺相的，表中就没有这个键，直接跳过
        if !haskey(bus_phase_to_row, (node_idx, phase_offset))
            continue 
        end
        
        # 获取该 (节点, 相位) 在压缩矩阵中对应的行号
        row_idx = bus_phase_to_row[(node_idx, phase_offset)]

        # 4. 解析物理量列
        # 处理可能的名称格式差异
        token_raw = join(parts[4:end], "")
        token = lowercase(replace(token_raw, r"[\s_()]" => ""))
        
        haskey(column_map, token) || continue
        col = column_map[token]
        
        # 5. 获取标准差 (sigma)
        stds = get(data, "prediction_std", Float64[])
        isempty(stds) && continue
        σ = abs(stds[1])
        
        # 防止除以0
        σ < 1e-9 && continue 
        
        # 6. 计算并填入精度
        β[row_idx, col] = 1.0 / (σ^2)
    end

    return β
end

# function build_noise_precision_beta(
#     daily_predictions::Dict{String, Any},
#     phase_info::Dict{String, Any};
#     # --- 稳健性超参数（建议先用这些，再按效果微调）---
#     β_min::Float64 = 1.0,          # 过小会“没约束”
#     β_max::Float64 = 1e6,          # 关键：压住 1e9~1e13 这种锁死级别
#     # 不同量测类型的 sigma 地板（按你的量纲微调）
#     σ_floor_PQ::Float64 = 0.5,     # P/Q 若在 kW 量级，这个可以是 0.5~5
#     σ_floor_VrVi::Float64 = 1e-3,  # Vr/Vi 若在 pu 量级，1e-3~1e-2 合理
#     σ_floor_Vmag::Float64 = 1e-3,  # Vmag 同上
#     # 额外“观测噪声”项（nugget），防止 GP 过度自信
#     σ_nugget_PQ::Float64 = 0.5,
#     σ_nugget_V::Float64 = 1e-3
# )
#     sensors = get(daily_predictions, "sensors", Dict{String, Any}())

#     n_valid_phases = phase_info["n_valid_phases"]
#     bus_phase_to_row = phase_info["bus_phase_to_row"]

#     num_rows = n_valid_phases
#     num_cols = 5

#     column_map = Dict(
#         "p" => 1,
#         "q" => 2,
#         "vreal" => 3, "v_real" => 3,
#         "vimag" => 4, "v_imag" => 4,
#         "vmag" => 5, "v_mag" => 5
#     )
#     phase_map = Dict("a" => 1, "b" => 2, "c" => 3)

#     β = zeros(Float64, num_rows, num_cols)

#     for (name, data) in sensors
#         parts = split(name, '-')
#         length(parts) < 4 && continue

#         node_idx = try
#             parse(Int, parts[2])
#         catch
#             continue
#         end
#         node_idx == 1 && continue

#         phase_str = lowercase(parts[3])
#         haskey(phase_map, phase_str) || continue
#         phase_offset = phase_map[phase_str]

#         haskey(bus_phase_to_row, (node_idx, phase_offset)) || continue
#         row_idx = bus_phase_to_row[(node_idx, phase_offset)]

#         token_raw = join(parts[4:end], "")
#         token = lowercase(replace(token_raw, r"[\s_()]" => ""))
#         haskey(column_map, token) || continue
#         col = column_map[token]

#         stds = get(data, "prediction_std", Float64[])
#         isempty(stds) && continue
#         σ_gp = abs(Float64(stds[1]))

#         # --- 关键：按列设置 sigma floor + nugget ---
#         σ_floor = (col == 1 || col == 2) ? σ_floor_PQ :
#                   (col == 3 || col == 4) ? σ_floor_VrVi :
#                                            σ_floor_Vmag

#         σ_nugget = (col == 1 || col == 2) ? σ_nugget_PQ : σ_nugget_V

#         # 观测总不确定性（避免 σ->0）
#         σ_eff = sqrt(max(σ_gp, σ_floor)^2 + σ_nugget^2)

#         β_ij = 1.0 / (σ_eff^2)
#         β[row_idx, col] = clamp(β_ij, β_min, β_max)
#     end

#     return β
# end

