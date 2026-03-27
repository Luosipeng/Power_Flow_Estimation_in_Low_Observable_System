function calculate_matrix_uncertainty(A_mean, B_mean, Σa_list, Σb_list)
    rows = size(A_mean, 1)
    cols = size(B_mean, 1)
    
    var_matrix = zeros(Float64, rows, cols)
    
    # 遍历矩阵的每一个元素
    for i in 1:rows
        μa = A_mean[i, :]
        Σa = Σa_list[i]
        
        for j in 1:cols
            μb = B_mean[j, :]
            Σb = Σb_list[j]
            
            # 1. 迹项 (交互噪声)
            # Tr(Σa * Σb)
            term1 = tr(Σa * Σb)
            
            # 2. A 的均值 投影到 B 的不确定性
            # μa' * Σb * μa
            term2 = dot(μa, Σb * μa)
            
            # 3. B 的均值 投影到 A 的不确定性
            # μb' * Σa * μb
            term3 = dot(μb, Σa * μb)
            
            var_matrix[i, j] = term1 + term2 + term3
        end
    end
    
    # 标准差 (Standard Deviation) = sqrt(Variance)
    std_matrix = sqrt.(var_matrix)
    
    return var_matrix, std_matrix
end