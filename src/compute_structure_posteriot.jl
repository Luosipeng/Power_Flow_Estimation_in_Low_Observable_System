# 正确计算结构后验的完整流程
function compute_structure_posterior(
    elbo_results::Vector{NamedTuple},  # 每个结构的ELBO结果
    prob_list::Vector{Float64};         # 先验概率列表
    temperature::Float64 = 1.0          # 温度参数（默认1.0）
)
    n_structures = length(elbo_results)
    @assert length(prob_list) == n_structures "ELBO结果和先验概率数量不匹配"
    
    # 1. 提取所有ELBO值
    elbo_values = [res[:elbo] for res in elbo_results]
    
    # 2. 数值稳定的计算（使用log-sum-exp技巧）
    # 先在对数空间计算
    log_unnormalized_posteriors = zeros(n_structures)
    
    for i in 1:n_structures
        # log P(F_i, T_i | Z) = log P(F_i, T_i) + ELBO(F_i, T_i) / temperature
        log_unnormalized_posteriors[i] = log(prob_list[i]) + elbo_values[i] / temperature
    end
    
    # 3. 归一化（在对数空间）
    log_Z = logsumexp(log_unnormalized_posteriors)
    log_posteriors = log_unnormalized_posteriors .- log_Z
    
    # 4. 转换回概率空间
    posteriors = exp.(log_posteriors)
    
    # 5. 验证归一化
    @assert abs(sum(posteriors) - 1.0) < 1e-10 "后验概率和不为1"
    
    # 6. 返回结果
    return (
        posteriors = posteriors,
        log_posteriors = log_posteriors,
        log_evidence = log_Z,  # 模型证据的对数
        best_idx = argmax(posteriors),
        entropy = -sum(p * log(p + 1e-10) for p in posteriors)  # 后验熵
    )
end

# 辅助函数：log-sum-exp
function logsumexp(x::Vector{Float64})
    max_x = maximum(x)
    return max_x + log(sum(exp.(x .- max_x)))
end

# 对于您的具体情况
function compute_posterior_for_structure_24(
    elbo_results::Vector{NamedTuple},
    prob_list::Vector{Float64},
    idx::Int = 24
)
    # 计算所有结构的后验
    posterior_results = compute_structure_posterior(elbo_results, prob_list)
    
    # 提取第24个结构的后验
    P_24 = posterior_results.posteriors[idx]
    
    println("结构 $idx 的后验概率: $P_24")
    println("最优结构索引: $(posterior_results.best_idx)")
    println("后验熵: $(posterior_results.entropy)")
    
    return P_24
end

# 温度参数的影响分析
function analyze_temperature_effect(
    elbo_results::Vector{NamedTuple},
    prob_list::Vector{Float64};
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
)
    println("\n温度参数对后验的影响:")
    println("温度\t最优结构\t后验熵")
    
    for T in temperatures
        results = compute_structure_posterior(elbo_results, prob_list; temperature=T)
        println("$T\t$(results.best_idx)\t$(round(results.entropy, digits=3))")
    end
end

# 诊断函数：检查ELBO的合理性
function diagnose_elbo_values(elbo_results::Vector{NamedTuple})
    elbo_values = [res[:elbo] for res in elbo_results]
    
    println("\nELBO诊断:")
    println("最小ELBO: $(minimum(elbo_values))")
    println("最大ELBO: $(maximum(elbo_values))")
    println("ELBO范围: $(maximum(elbo_values) - minimum(elbo_values))")
    
    # 检查是否有异常值
    if any(isnan.(elbo_values)) || any(isinf.(elbo_values))
        @warn "ELBO中存在NaN或Inf值！"
    end
    
    # 检查ELBO的尺度
    if maximum(abs.(elbo_values)) > 1e6
        @warn "ELBO值过大，可能需要重新缩放"
    end
end