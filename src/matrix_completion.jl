# -----------------------
# Topology helpers
# -----------------------

function fraction_available_data(Z::AbstractMatrix)
    total = length(Z); total == 0 && return 0.0f0
    available = count(!iszero, Z)
    return available / total
end

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
    Σa_i_inv = Diagonal(γ) + βBtB_i
    return inv(Σa_i_inv)
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
    Σb_j_inv = Diagonal(γ) + βAtA_j
    return inv(Σb_j_inv)
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

function build_incidence_matrix_td(n_nodes, branch)

    # Calculate total number of branches
    branch_in_service = branch[:, 11] .== 1
    n_branches = size(branch[branch_in_service,:], 1)
    
    # Extract start and end nodes for all branches, and mark branch type
    # Type markers: 1=AC, 2=DC, 3=Converter
    branch_data = Vector{Tuple{Int, Int, Int}}(undef, n_branches)
    
    # Add AC branches
    for i in 1:size(branch[branch_in_service,:], 1)
        # Ensure smaller node number as starting node
        node1 = Int(branch[branch_in_service,:][i, 1])
        node2 = Int(branch[branch_in_service,:][i, 2])
        from = min(node1, node2)
        to = max(node1, node2)
        branch_data[i] = (from, to, i)  # (start node, end node, type, original index)
    end
    
    # Sort by starting node
    sort!(branch_data, by = x -> (x[1], x[2]))
    
    # Create incidence matrix
    A = zeros(n_branches, n_nodes)
    
    # Build incidence matrix based on sorted branches
    for (idx, (from, to, _)) in enumerate(branch_data)
        A[idx, from] = 1   # Outflow from node is positive
        A[idx, to] = -1    # Inflow to node is negative
    end
    
    return A, branch_data
end