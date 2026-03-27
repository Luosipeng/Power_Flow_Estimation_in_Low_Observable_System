function build_w_matrix(Y_bus, v_slack_ph=[1.0; exp(-im*2pi/3); exp(im*2pi/3)])
    Yn0 = Y_bus[4:end, 1:3]
    Ynn = Y_bus[4:end, 4:end]
    # 注意：这里 Yn0 是稀疏的，v_slack_ph 是稠密的，乘积是稠密向量
    current_injection = Yn0 * v_slack_ph
    w = -(Ynn \ current_injection)
    return w
end

function build_matrix_M(Y_bus, w)
    # 提取 Ynn
    Ynn = Y_bus[4:end, 4:end]
    
    # 1. 计算 Ynn 的逆
    Znn = inv(Matrix(Ynn)) 
    
    # 2. 构建 diag(w_conj)^-1
    inv_W_conj = Diagonal(1 ./ conj(w))
    
    # 3. 计算 M_left (对应 P)
    # M_left = Znn * inv_W_conj
    M_left = Znn * inv_W_conj
    
    # 4. 计算 M_right (对应 Q)
    # M_right = -j * M_left
    M_right = -im .* M_left
    
    # 5. 拼接
    M = hcat(M_left, M_right)
    
    return M
end

function build_matrix_K(M, w)
    # 修正点：将 conj(w) 放入 Diagonal() 中
    # 修正点：abs(w) 等于 abs(conj(w))，直接用 abs.(w) 即可
    
    inv_abs_W = Diagonal(1.0 ./ abs.(w)) # 对应 1/|w|
    W_conj_diag = Diagonal(conj(w))      # 对应 diag(w*)
    
    # K = inv(|w|) * Real( diag(w*) * M )
    K = inv_abs_W * real(W_conj_diag * M)
    
    return K
end