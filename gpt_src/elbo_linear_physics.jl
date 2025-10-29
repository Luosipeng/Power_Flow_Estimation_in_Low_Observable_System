"""
  compute_elbo_linear_phys(Z, X, β, observed_pairs,
                           A_mean, B_mean, Σa_list, Σb_list, γ; 
                           J_phys, c_phys, x0, λP=1.0, λQ=1.0, λV=1.0,
                           c=1e-6, d=1e-6)

计算相对ELBO（去常数），包含：
  - 统计似然: -0.5 ∑ β (Z - X)^2（仅观测位置）
  - 先验AB(ARD): -0.5 ∑ tr(Γ (Σ + μμᵀ))
  - γ先验: ∑ [(c-1)logγ - dγ]
  - 线性化物理项: -0.5 ( r_linᵀ Λ r_lin )，其中 r_lin = J (x - x0) + c

物理项的精度矩阵 Λ 为对角块：diag([λP*1_m; λQ*1_m; λV*1_m])
"""
function compute_elbo_linear_phys(Z::AbstractMatrix, X::AbstractMatrix, β,
                                  observed_pairs::Vector{Tuple{Int,Int}},
                                  A_mean::AbstractMatrix, B_mean::AbstractMatrix,
                                  Σa_list::Vector{<:AbstractMatrix}, Σb_list::Vector{<:AbstractMatrix}, γ::AbstractVector;
                                  J_phys::AbstractMatrix, c_phys::AbstractVector, x0::AbstractVector,
                                  λP::Real=1.0, λQ::Real=1.0, λV::Real=1.0,
                                  c::Real=1e-6, d::Real=1e-6)

    m, n = size(X)
    @assert n == 5
    Γ = Diagonal(γ)

    # 统计似然（支持 β 为矩阵或列向量）
    like = 0.0
    if isa(β, AbstractVector) && length(β)==n
        βcols = β
        for (i,j) in observed_pairs
            e = Z[i,j] - X[i,j]
            like += -0.5 * βcols[j] * (e*e)
        end
    else
        βmat = β
        for (i,j) in observed_pairs
            e = Z[i,j] - X[i,j]
            like += -0.5 * βmat[i,j] * (e*e)
        end
    end

    # 先验AB（期望形式）
    prior_ab = 0.0
    for i in 1:size(A_mean,1)
        prior_ab += -0.5 * tr(Γ * (Σa_list[i] + A_mean[i,:]*A_mean[i,:]'))
    end
    for j in 1:size(B_mean,1)
        prior_ab += -0.5 * tr(Γ * (Σb_list[j] + B_mean[j,:]*B_mean[j,:]'))
    end

    # γ先验
    prior_gamma = 0.0
    for k in 1:length(γ)
        prior_gamma += (float(c)-1.0)*log(γ[k]) - float(d)*γ[k]
    end

    # 线性化物理项
    # 变量拼接
    P = X[:,1]; Q = X[:,2]; Vr = X[:,3]; Vi = X[:,4]; Vmag = X[:,5]
    x = vcat(P,Q,Vr,Vi,Vmag)
    r_lin = J_phys * (x - x0) .+ c_phys

    # 精度权重
    Λ = vcat(fill(float(λP), m), fill(float(λQ), m), fill(float(λV), m))
    phys = 0.5 * sum(Λ .* (r_lin .^ 2))

    elbo = like + prior_ab + prior_gamma - phys
    return (elbo=elbo, like=like, prior_ab=prior_ab, prior_gamma=prior_gamma, phys=phys)
end