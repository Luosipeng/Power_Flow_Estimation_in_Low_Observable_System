using LinearAlgebra

# Quick standalone copy of compute_element_variance (same as updated in test_gpt.jl)
function compute_element_variance(A_mean::AbstractMatrix{<:Real},
                                  B_mean::AbstractMatrix{<:Real},
                                  Σa_list::Vector{<:AbstractMatrix{<:Real}},
                                  Σb_list::Vector{<:AbstractMatrix{<:Real}})
    m, rA = size(A_mean)
    n, rB = size(B_mean)
    @assert rA == rB "Latent dimension mismatch: size(A_mean,2)=$(rA) vs size(B_mean,2)=$(rB)"
    r = rA

    @assert all(size(Σ) == (r, r) for Σ in Σa_list) "Each Σa must be r×r"
    @assert all(size(Σ) == (r, r) for Σ in Σb_list) "Each Σb must be r×r"
    @assert length(Σa_list) == m "Σa_list length must be m"
    @assert length(Σb_list) == n "Σb_list length must be n"

    VarX = zeros(Float64, m, n)

    for i in 1:m
        ΣAi = Σa_list[i]::AbstractMatrix
        ai_vec = vec(@view A_mean[i, :])
        for j in 1:n
            ΣBj = Σb_list[j]::AbstractMatrix
            bj_vec = vec(@view B_mean[j, :])
            t1 = dot(bj_vec, ΣAi * bj_vec)
            t2 = dot(ai_vec, ΣBj * ai_vec)
            t3 = tr(ΣAi * ΣBj)
            VarX[i, j] = t1 + t2 + t3
        end
    end

    return VarX
end

# Synthetic test
m = 5; n = 4; r = 3
A_mean = randn(m, r)
B_mean = randn(n, r)
Σa_list = [Symmetric(0.1I + 0.01*randn(r,r)) for _ in 1:m]
Σb_list = [Symmetric(0.1I + 0.01*randn(r,r)) for _ in 1:n]

VarX = compute_element_variance(A_mean, B_mean, Σa_list, Σb_list)
println("VarX size: ", size(VarX))
println(VarX)
