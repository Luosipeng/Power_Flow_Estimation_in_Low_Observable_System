struct StandardScaler
    mean::Vector{Float64}
    std::Vector{Float64}
end

function fit_transform!(matrix::Matrix{Float64})
    n_cols = size(matrix, 2)
    means = zeros(n_cols)
    stds = zeros(n_cols)
    
    for c in 1:n_cols
        # 仅利用非零元素计算统计量 (假设0是缺失值)
        # 或者如果你已经填补了初始值，直接算整体
        vals = matrix[:, c]
        means[c] = mean(vals)
        stds[c] = std(vals)
        if stds[c] < 1e-6 stds[c] = 1.0 end # 防止除以0
        
        # 执行归一化
        matrix[:, c] .= (matrix[:, c] .- means[c]) ./ stds[c]
    end
    return StandardScaler(means, stds)
end

function inverse_transform(matrix::Matrix{Float64}, scaler::StandardScaler)
    denorm_matrix = copy(matrix)
    for c in 1:size(matrix, 2)
        denorm_matrix[:, c] .= (matrix[:, c] .* scaler.std[c]) .+ scaler.mean[c]
    end
    return denorm_matrix
end

# 专门用于反归一化方差/标准差
function inverse_transform_std(std_matrix::Matrix{Float64}, scaler::StandardScaler)
    denorm_std = copy(std_matrix)
    for c in 1:size(std_matrix, 2)
        denorm_std[:, c] .= std_matrix[:, c] .* scaler.std[c]
    end
    return denorm_std
end
