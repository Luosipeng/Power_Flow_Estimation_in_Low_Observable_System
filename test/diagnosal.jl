# 检查原始数据范围
println("=== observed_matrix_Z 范围 ===")
for col in 1:5
    col_data = [observed_matrix_Z[r, col] for (r, c) in observed_pairs if c == col]
    if !isempty(col_data)
        println("Col $col: min=$(minimum(col_data)), max=$(maximum(col_data)), mean=$(mean(col_data))")
    end
end

println("\n=== init_matrix 范围 ===")
for col in 1:5
    println("Col $col: min=$(minimum(init_matrix[:, col])), max=$(maximum(init_matrix[:, col])), mean=$(mean(init_matrix[:, col]))")
end
