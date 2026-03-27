function cal_beta_BTB_i(i, B, Σb_list, observed_pairs, noise_precision_β, d)
    βBtB_i = zeros(Float64, d, d)
    for (row, col) in observed_pairs
        if row == i
            bj = B[col, :]
            β_ij = noise_precision_β[i, col]
            βBtB_i += β_ij * (bj * bj' + Σb_list[col])
        end
    end
    return βBtB_i
end

function cal_sigma_a_i(βBtB_i, γ)
    # 1. 增大 Jitter 到 1e-5，抑制弱特征维度的数值爆炸
    # 2. 使用 pinv (伪逆) 代替 inv，这是处理奇异矩阵的"核武器"，永不崩溃
    Σa_i_inv = Diagonal(γ) + βBtB_i + 1e-5 * I
    return pinv(Σa_i_inv)
end

function cal_a_mean_i(i, B, Σa_i, observed_pairs, noise_precision_β, observed_matrix_Z)
    d = size(B, 2)
    weighted_sum = zeros(Float64, d)
    for (row, col) in observed_pairs
        if row == i
            bj = B[col, :]
            β_ij = noise_precision_β[i, col]
            z_ij = observed_matrix_Z[i, col]
            weighted_sum += β_ij * bj * z_ij
        end
    end
    return Σa_i * weighted_sum
end

function cal_beta_ATA_j(j, A_mean, Σa_list, observed_pairs, noise_precision_β, d)
    βAtA_j = zeros(Float64, d, d)
    for (row, col) in observed_pairs
        if col == j
            ai = A_mean[row, :]
            β_ij = noise_precision_β[row, j]
            βAtA_j += β_ij * (ai * ai' + Σa_list[row])
        end
    end
    return βAtA_j
end

function cal_sigma_b_j(βAtA_j, γ)
    Σb_j_inv = Diagonal(γ) + βAtA_j + 1e-5 * I
    return pinv(Σb_j_inv)
end

function cal_b_mean_j(j, A_mean, Σb_j, observed_pairs, noise_precision_β, observed_matrix_Z)
    d = size(A_mean, 2)
    weighted_sum = zeros(Float64, d)
    for (row, col) in observed_pairs
        if col == j
            ai = A_mean[row, :]
            β_ij = noise_precision_β[row, j]
            z_ij = observed_matrix_Z[row, j]
            weighted_sum += β_ij * ai * z_ij
        end
    end
    return Σb_j * weighted_sum
end

function cal_aTa_i(i, A_mean, Σa_list)
    m = size(A_mean, 1)
    a_i_means = A_mean[:, i]
    mean_squared_sum = dot(a_i_means, a_i_means)
    variance_sum = sum(Σa_list[k][i, i] for k in 1:m)
    return mean_squared_sum + variance_sum
end

function cal_bTb_j(i, B_mean, Σb_list)
    n = size(B_mean, 1)
    b_i_means = B_mean[:, i]
    mean_squared_sum = dot(b_i_means, b_i_means)
    variance_sum = sum(Σb_list[k][i, i] for k in 1:n)
    return mean_squared_sum + variance_sum
end
