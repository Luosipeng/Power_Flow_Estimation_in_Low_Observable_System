using MAT
using LinearAlgebra
using SparseArrays
using Statistics
using Printf

function get_aligned_physics_matrices(phase_info::Dict{String, Any}, baseMVA::Float64=10.0)
    """
    基于缺相信息构建物理矩阵（智能修复版）
    功能：自动识别高阻抗（悬空）节点，并将其物理参数归零，防止数值爆炸。
    """
    
    # ================= 1. 读取数据 =================
    topo_path = "/home/user/Codes/luosipeng/topology.mat"
    idx_path  = "/home/user/Codes/luosipeng/bus_index.mat"
    
    if !isfile(topo_path) || !isfile(idx_path)
        error("❌ 找不到拓扑文件")
    end

    topo = matread(topo_path)
    Y_full = topo["Ybus"]
    kVBase_full = topo["kVBase"]
    ybus_node_names = topo["nodes"]
    
    bus_idx_data = matread(idx_path)
    ordered_bus_names = bus_idx_data["buses"]
    
    println("\n⚡ 物理矩阵构建 (智能修复版):")
    println("="^60)
    
    # ================= 2. 建立映射与索引 =================
    node_lookup = Dict{Tuple{String, Int}, Int}()
    for (row, node_str) in enumerate(ybus_node_names)
        parts = split(node_str, '.')
        b_name = lowercase(strip(parts[1]))
        ph = length(parts) > 1 ? parse(Int, parts[2]) : 1
        node_lookup[(b_name, ph)] = row
    end
    
    slack_bus_name = lowercase(strip(string(ordered_bus_names[1])))
    slack_indices = Int[]
    load_indices = Int[]
    load_mapping_list = Tuple{Int, Int}[]
    
    # 找平衡节点
    for ph in 1:3
        key = (slack_bus_name, ph)
        if haskey(node_lookup, key)
            push!(slack_indices, node_lookup[key])
        end
    end
    
    # 找负载节点 (排除缺相)
    lack_a = phase_info["lack_a"]
    lack_b = phase_info["lack_b"]
    lack_c = phase_info["lack_c"]
    n_total_nodes = phase_info["n_total_nodes"]
    
    for bus_int in 2:n_total_nodes
        if bus_int > length(ordered_bus_names) continue end
        bus_name_str = lowercase(strip(string(ordered_bus_names[bus_int])))
        
        for phase in 1:3
            is_missing = (phase == 1 && bus_int in lack_a) ||
                         (phase == 2 && bus_int in lack_b) ||
                         (phase == 3 && bus_int in lack_c)
            
            key = (bus_name_str, phase)
            if !is_missing && haskey(node_lookup, key)
                push!(load_indices, node_lookup[key])
                push!(load_mapping_list, (bus_int, phase))
            end
        end
    end
    
    # ================= 3. 矩阵对齐 (BMC 顺序) =================
    Y_LL_raw = Y_full[load_indices, load_indices]
    Y_LS_raw = Y_full[load_indices, slack_indices]
    kVBase_LL_raw = kVBase_full[load_indices]
    
    bus_phase_to_row = phase_info["bus_phase_to_row"]
    n_bmc_rows = phase_info["n_valid_phases"]
    bmc_perm = zeros(Int, n_bmc_rows)
    
    load_list_lookup = Dict{Tuple{Int, Int}, Int}()
    for (i, val) in enumerate(load_mapping_list)
        load_list_lookup[val] = i
    end
    
    for ((bus, phase), bmc_row) in bus_phase_to_row
        if bmc_row <= n_bmc_rows && haskey(load_list_lookup, (bus, phase))
            bmc_perm[bmc_row] = load_list_lookup[(bus, phase)]
        end
    end
    
    # 重排
    Y_LL = Y_LL_raw[bmc_perm, bmc_perm]
    Y_LS = Y_LS_raw[bmc_perm, :]
    kVBase_aligned = kVBase_LL_raw[bmc_perm]
    
    # ================= 4. ������️‍♂️ 阻抗分析与坏节点隔离 =================
    println("\n[智能诊断] 计算阻抗矩阵...")
    
    # 添加微小正则化以保证可逆，用于检测
    Y_check = Matrix(Y_LL) + 1e-7 * I
    Z_check = inv(Y_check)
    z_diag = abs.(diag(Z_check))
    
    # 阈值：1000 欧姆 (远大于正常配电网阻抗)
    bad_node_indices = findall(x -> x > 1000.0, z_diag)
    
    if !isempty(bad_node_indices)
        println("  ⚠️ 警告: 检测到 $(length(bad_node_indices)) 个高阻抗(悬空)节点！")
        println("     这些节点会导致物理矩阵爆炸。正在进行物理隔离...")
        
        for idx in bad_node_indices
            val = z_diag[idx]
            # 尝试反向查找是哪个节点
            # (这里为了代码简洁跳过反查名字，直接处理)
            if idx <= 5 || idx >= length(z_diag)-5
                println("     - Row $idx: |Z| = $(Printf.@sprintf("%.2e", val)) Ω (Isolating...)")
            end
        end
        
        # === 核心修复策略: 物理隔离 (Clamping) ===
        # 我们不改变矩阵维度，而是将这些节点的物理方程“失效化”
        # 1. 在 Z_check 中，将对应行/列设为 0 (切断耦合)
        # 2. 将对角线设为 1.0 (防止奇异，虽然此时物理意义已改变)
        
        # 注意：我们直接修改 Z_LL_ohms，而不是 Y
        # 因为 Y 已经包含了导致高阻抗的结构
        
        # 将 Z 矩阵中的坏行坏列清零
        Z_check[bad_node_indices, :] .= 0.0
        Z_check[:, bad_node_indices] .= 0.0
        
        # 保持对角线非零 (避免后续计算除零错误)，设为标称值对应的基准阻抗
        # 或者简单地设为 0，然后在 M, K 构建时处理
        # 这里我们选择在 M, K 构建阶段强制置零
    else
        println("  ✓ 未检测到高阻抗节点。")
    end
    
    Z_LL_ohms = Z_check

    # ================= 5. 计算零载电压 w =================
    a = -0.5 + im * sqrt(3)/2
    v_slack = [1.0 + 0im, a^2, a]
    
    # w = -Z_LL * Y_LS * v_slack
    w = -Z_LL_ohms * Y_LS * v_slack
    
    # === 修复 w ===
    # 对于坏节点，w 可能会计算出异常值，或者如果是 0 (因为 Z 被置零了)
    # 我们将其重置为标称电压 (1.0 p.u.)
    for idx in bad_node_indices
        # 找到该行对应的相位
        # 我们需要反查 phase。这有点麻烦，但我们可以根据 w 的当前角度或者简单的 1.0
        # 简单策略：设为 1.0，相位由数据驱动部分决定
        w[idx] = 1.0 + 0.0im 
    end
    
    # 额外的安全检查
    if any(abs.(w) .> 2.0)
        println("  ⚠️ 检测到 w 电压异常 (> 2.0 p.u.)，强制截断...")
        w .= clamp.(abs.(w), 0.0, 2.0) .* (w ./ abs.(w))
    end

    # ================= 6. 构建 M, K =================
    Z_base_vec = (kVBase_aligned .^ 2) ./ baseMVA
    replace!(Z_base_vec, 0.0 => 1.0)
    inv_Z_base_diag = Diagonal(1.0 ./ Z_base_vec)
    
    Z_LL_pu = inv_Z_base_diag * Z_LL_ohms
    
    # === 再次确保坏节点在标幺值矩阵中也是 0 ===
    if !isempty(bad_node_indices)
        Z_LL_pu[bad_node_indices, :] .= 0.0
        Z_LL_pu[:, bad_node_indices] .= 0.0
    end
    
    # M = [Z_nn * diag(conj(w))^{-1} | -j * Z_nn * diag(conj(w))^{-1}]
    # The diag(conj(w))^{-1} factor is required by the linearised power-flow model
    w_conj_safe = conj.(w)
    w_conj_safe[abs.(w_conj_safe) .< 0.1] .= 1.0  # protect against near-zero w
    inv_W_conj = Diagonal(1.0 ./ w_conj_safe)
    
    M_P = Z_LL_pu * inv_W_conj
    M_Q = -im .* M_P
    M = hcat(M_P, M_Q)
    
    abs_w = max.(abs.(w), 0.1) # 防止除零
    inv_abs_w = 1.0 ./ abs_w
    w_conj_diag = Diagonal(conj.(w))
    
    K_P = real.(inv_abs_w .* (w_conj_diag * M_P))
    K_Q = real.(inv_abs_w .* (w_conj_diag * M_Q))
    K = hcat(K_P, K_Q)
    
    println("\n  最终矩阵统计:")
    println("    M 最大值: $(round(maximum(abs.(M)), digits=4))")
    println("    K 最大值: $(round(maximum(abs.(K)), digits=4))")
    
    if maximum(abs.(M)) > 20.0
        @warn "M 矩阵仍包含较大数值，请检查输入数据。"
    end
    
    println("✅ 物理矩阵构建完成 (已隔离 $(length(bad_node_indices)) 个坏节点)")
    println("="^60)
    
    return w, M, K
end
