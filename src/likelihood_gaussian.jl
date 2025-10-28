using SpecialFunctions: loggamma
# 计算在观测位置上的多元高斯独立似然的对数值
# Z, X, beta 为同形 Array{Float64,2}；observed_pairs 为 Vector{Tuple{Int,Int}} 或等效的观测索引集合
function log_likelihood_gaussian(Z, X, beta, observed_pairs)::Float64
    @assert size(Z) == size(X) == size(beta) "Z, X, beta must have same size"
    # 数值安全：避免 log(0) 或 beta<=0
    epsβ = 1e-12
    log2π = log(2π)

    s = 0.0
    for (i, j) in observed_pairs
        β = max(float(beta[i, j]), epsβ)
        Δ = float(Z[i, j]) - float(X[i, j])
        # 对应单项对数似然：-1/2 [ log(2π) - log β + β * Δ^2 ]
        if(-0.5 * (log2π - log(β) + β * (Δ * Δ))<0)
            @warn "Numerical issue in log likelihood computation at index=($i,$j), β=$β, Δ=$Δ"
        end
        s += -0.5 * (log2π - log(β) + β * (Δ * Δ))
    end
    return s
end

# 直接返回似然值（注意可能数值下溢）
function likelihood_gaussian(Z::AbstractMatrix{<:Real},
                             X::AbstractMatrix{<:Real},
                             beta::AbstractMatrix{<:Real},
                             observed_pairs)::Float64
    return exp(log_likelihood_gaussian(Z, X, beta, observed_pairs))
end

function compute_loglik_from_state(observed_matrix_Z, noise_precision_β, observed_pairs, X_hat)::Float64
    return log_likelihood_gaussian(observed_matrix_Z, X_hat, noise_precision_β, observed_pairs)
end

"""
Compute log p(A | gamma) for Bayesian matrix completion.
A :: AbstractMatrix{<:Real}  (m × k), columns a_.i
gamma :: AbstractVector{<:Real} length k, precision per component
Returns :: Float64 (log-probability)
"""
function log_prior_A(A::AbstractMatrix{<:Real}, gamma::AbstractVector{<:Real})::Float64
    m, k = size(A)
    @assert length(gamma) == k "gamma length must equal number of columns in A"
    log2π = log(2π)
    s = 0.0
    @inbounds for i in 1:k
        γ = max(float(gamma[i]), 1e-12)  # numerical safety
        a_i = @view A[:, i]
        norm2_ai = dot(a_i, a_i)
        s += -(m/2)*log2π + (m/2)*log(γ) - (γ/2)*norm2_ai
    end
    return s
end

"""
Compute p(A | gamma) (may underflow for large dimensions).
Prefer using log_prior_A.
"""
function prior_A(A::AbstractMatrix{<:Real}, gamma::AbstractVector{<:Real})::Float64
    return exp(log_prior_A(A, gamma))
end

"""
Compute log p(B | gamma) for Bayesian matrix completion.
B :: AbstractMatrix{<:Real}  (n × k), columns b_.i
gamma :: AbstractVector{<:Real} length k
Returns :: Float64 (log-probability)
"""
function log_prior_B(B::AbstractMatrix{<:Real}, gamma::AbstractVector{<:Real})::Float64
    n, k = size(B)
    @assert length(gamma) == k "gamma length must equal number of columns in B"
    log2π = log(2π)
    s = 0.0
    @inbounds for i in 1:k
        γ = max(float(gamma[i]), 1e-12)
        b_i = @view B[:, i]
        norm2_bi = dot(b_i, b_i)
        s += -(n/2)*log2π + (n/2)*log(γ) - (γ/2)*norm2_bi
    end
    return s
end

"""
Compute p(B | gamma) (may underflow).
Prefer using log_prior_B.
"""
function prior_B(B::AbstractMatrix{<:Real}, gamma::AbstractVector{<:Real})::Float64
    return exp(log_prior_B(B, gamma))
end


"""
Log prior for gamma under Gamma(shape=c, rate=1/d).
p(γ_i) = Gamma(c, 1/d)  with density:
p(γ_i) = (1/d)^c / Γ(c) * γ_i^{c-1} * exp(-γ_i / d),  γ_i > 0
Inputs:
  gamma :: AbstractVector{<:Real}  length k, γ_i > 0
  c     :: Real  (shape) must be > 0
  d     :: Real  (scale parameter's denominator; rate = 1/d) must be > 0
Returns:
  Float64: sum_i log p(γ_i)
"""
function log_prior_gamma(gamma::AbstractVector{<:Real}, c::Real, d::Real)::Float64
    @assert c > 0 "shape c must be > 0"
    @assert d > 0 "d must be > 0 (rate = 1/d)"
    s = 0.0
    c = float(c); d = float(d)
    epsγ = 1e-12
    term_const = -c*log(d) - loggamma(c)  # c*log(1/d) - logΓ(c)
    @inbounds for i in eachindex(gamma)
        γ = max(float(gamma[i]), epsγ)  # numerical safety; also enforces positivity
        s += (c - 1)*log(γ) - γ/d + term_const
    end
    return s
end

"""
Prior for gamma (may underflow). Prefer using log_prior_gamma.
"""
function prior_gamma(gamma::AbstractVector{<:Real}, c::Real, d::Real)::Float64
    return exp(log_prior_gamma(gamma, c, d))
end

using LinearAlgebra
using Statistics

# 计算考虑物理约束的ELBO
function compute_elbo_with_physics(X, A_mean, B_mean, Σa_list, Σb_list, γ, 
                                  observed_matrix_Z, noise_precision_β, observed_pairs,
                                  c=1e-7, d=1e-7)
    
    n, m = size(X)
    r = size(A_mean, 2)
    
    # 1. 数据似然项 - 只计算观测位置
    log_likelihood = 0.0
    for (i, j) in observed_pairs
        β_ij = noise_precision_β[i, j]
        z_ij = observed_matrix_Z[i, j]
        x_ij = X[i, j]
        
        # 高斯似然: -0.5 * β * (z - x)^2 + 0.5 * log(β) - 0.5 * log(2π)
        log_likelihood += -0.5 * β_ij * (z_ij - x_ij)^2 + 0.5 * log(β_ij) - 0.5 * log(2π)
    end
    
    # 2. KL散度项 - A的先验
    kl_A = 0.0
    for i in 1:n
        a_i = A_mean[i, :]
        Σa_i = Σa_list[i]
        
        # KL(q(a_i) || p(a_i|γ))
        # = 0.5 * [tr(Γ * Σa) + a' * Γ * a - log|Σa| - r + log|Γ^(-1)|]
        Γ = Diagonal(γ)
        kl_A += 0.5 * (tr(Γ * Σa_i) + dot(a_i, Γ * a_i) - logdet(Σa_i) - r + logdet(inv(Γ)))
    end
    
    # 3. KL散度项 - B的先验
    kl_B = 0.0
    for j in 1:m
        b_j = B_mean[j, :]
        Σb_j = Σb_list[j]
        
        # KL(q(b_j) || p(b_j|γ))
        Γ = Diagonal(γ)
        kl_B += 0.5 * (tr(Γ * Σb_j) + dot(b_j, Γ * b_j) - logdet(Σb_j) - r + logdet(inv(Γ)))
    end
    
    # 4. γ的先验项
    log_p_gamma = 0.0
    for k in 1:r
        # Gamma先验: p(γ_k) ∝ γ_k^(c-1) * exp(-d * γ_k)
        log_p_gamma += (c - 1) * log(γ[k]) - d * γ[k]
    end
    
    # 5. 物理约束的"似然"项（可选）
    # 这是您的方法独特之处 - 可以添加一个衡量物理一致性的项
    physics_penalty = 0.0
    if size(X, 2) >= 5
        # 检查功率平衡等物理约束的满足程度
        # 这里可以根据需要添加
    end
    
    # ELBO = log_likelihood - KL_A - KL_B + log_p_gamma - physics_penalty
    elbo = log_likelihood - kl_A - kl_B + log_p_gamma - physics_penalty
    
    # 返回详细信息
    return Dict(
        :elbo => elbo,
        :log_likelihood => log_likelihood,
        :kl_A => kl_A,
        :kl_B => kl_B,
        :log_p_gamma => log_p_gamma,
        :physics_penalty => physics_penalty
    )
end

# 近似ELBO计算 - 考虑物理约束作为确定性变换
function compute_approximate_elbo(X, A_mean, B_mean, Σa_list, Σb_list, γ,
                                 observed_matrix_Z, noise_precision_β, observed_pairs,
                                 X_before_physics=nothing, c=1e-7, d=1e-7)
    
    # 基础ELBO（使用物理修正后的X）
    elbo_dict = compute_elbo_with_physics(X, A_mean, B_mean, Σa_list, Σb_list, γ,
                                         observed_matrix_Z, noise_precision_β, observed_pairs, c, d)
    
    # 如果提供了物理修正前的X，计算修正的影响
    if X_before_physics !== nothing
        # 计算雅可比行列式的对数（近似）
        # 物理约束可以看作是一个确定性变换 g: X_latent -> X_physics
        # ELBO需要加上 log|det(∂g/∂X)|
        
        # 简化处理：假设变换是局部线性的
        diff_norm = norm(X - X_before_physics)
        n_elements = length(X)
        
        # 近似雅可比贡献（这是一个粗略估计）
        log_jacobian_approx = -0.5 * n_elements * log(1 + diff_norm^2 / n_elements)
        
        elbo_dict[:log_jacobian] = log_jacobian_approx
        elbo_dict[:elbo_adjusted] = elbo_dict[:elbo] + log_jacobian_approx
    end
    
    return elbo_dict
end

# 另一种方法：将物理约束作为软约束纳入ELBO
function compute_elbo_soft_physics(X, A_mean, B_mean, Σa_list, Σb_list, γ,
                                   observed_matrix_Z, noise_precision_β, observed_pairs,
                                   P_inj_target, Q_inj_target, V_target,
                                   λ_physics=100.0, c=1e-7, d=1e-7)
    
    # 基础ELBO
    elbo_dict = compute_elbo_with_physics(X, A_mean, B_mean, Σa_list, Σb_list, γ,
                                         observed_matrix_Z, noise_precision_β, observed_pairs, c, d)
    
    # 添加物理约束作为软约束
    physics_likelihood = 0.0
    
    if size(X, 2) >= 5
        # 功率注入约束
        P_pred = X[:, 1] ./ 10
        Q_pred = X[:, 2] ./ 10
        V_pred = X[:, 5]
        
        # 假设物理约束的"观测"具有精度λ_physics
        physics_likelihood -= 0.5 * λ_physics * sum((P_pred - P_inj_target).^2)
        physics_likelihood -= 0.5 * λ_physics * sum((Q_pred - Q_inj_target).^2)
        physics_likelihood -= 0.5 * λ_physics * sum((V_pred - V_target).^2)
        
        # 加上归一化常数
        n_physics = 3 * length(P_pred)
        physics_likelihood += 0.5 * n_physics * log(λ_physics) - 0.5 * n_physics * log(2π)
    end
    
    elbo_dict[:physics_likelihood] = physics_likelihood
    elbo_dict[:elbo_with_physics] = elbo_dict[:elbo] + physics_likelihood
    
    return elbo_dict
end

# 保存到文件