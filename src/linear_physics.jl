using LinearAlgebra

"""
基于当前参考点 (Vr_ref, Vi_ref) 和 Ybus，构造 AC 潮流的一阶泰勒线性化，使 P,Q 与 Vr,Vi 的关系写成：
  [P; Q] ≈ c_phys + J_phys * [ΔVr; ΔVi]
为了在结构化矩阵 X 上施加约束，我们把该线性化转化为：
  r_phys = A_phys * vec(X) - b_phys
并在优化中约束 ||r_phys||₂ ≤ β

输入：
  Y::Matrix{ComplexF64} - 导纳矩阵
  Vr_ref::Vector{Float64}, Vi_ref::Vector{Float64} - 当前参考点
  n::Int - 节点数
  nfeat::Int - X的列数
  idx_Vr::Int=3, idx_Vi::Int=4, idx_P::Int=1, idx_Q::Int=2 - X各列索引

输出：
  A_phys::Matrix{Float64}, b_phys::Vector{Float64}

说明：
  - 使用直角坐标雅可比，严格一阶泰勒（含常数项），比“冻结系数法”更接近真实潮流。
  - 推导：Ir = G Vr - B Vi, Ii = B Vr + G Vi
    P_i = Vr_i*Ir_i + Vi_i*Ii_i
    Q_i = Vi_i*Ir_i - Vr_i*Ii_i
    对 Vr,Vi 求偏导即得雅可比。
"""
function build_linearized_physics(Y::Matrix{ComplexF64},
                                  Vr_ref::Vector{Float64}, Vi_ref::Vector{Float64},
                                  n::Int, nfeat::Int;
                                  idx_Vr::Int=3, idx_Vi::Int=4, idx_P::Int=1, idx_Q::Int=2)

    @assert length(Vr_ref) == n && length(Vi_ref) == n
    G = real.(Y)
    B = imag.(Y)

    # 选择矩阵：把 vec(X) 中的某一列抽出为长度 n 的向量
    function S_col(k::Int)
        S = zeros(n, n*nfeat)
        for i in 1:n
            S[i, (k-1)*n + i] = 1.0
        end
        return S
    end
    SVr = S_col(idx_Vr)
    SVi = S_col(idx_Vi)
    SP  = S_col(idx_P)
    SQ  = S_col(idx_Q)

    # 计算参考点的电流与功率（常数项）
    Ir_ref = G*Vr_ref .- B*Vi_ref
    Ii_ref = B*Vr_ref .+ G*Vi_ref
    P_ref  = Vr_ref .* Ir_ref .+ Vi_ref .* Ii_ref
    Q_ref  = Vi_ref .* Ir_ref .- Vr_ref .* Ii_ref

    # 雅可比：对每个节点 i，计算 ∂P_i/∂Vr_j, ∂P_i/∂Vi_j, ∂Q_i/∂Vr_j, ∂Q_i/∂Vi_j
    # Ir_i = Σ_j (G_ij Vr_j - B_ij Vi_j), Ii_i = Σ_j (B_ij Vr_j + G_ij Vi_j)
    # P_i = Vr_i*Ir_i + Vi_i*Ii_i
    # ∂P_i/∂Vr_j = δ_{ij}*Ir_i + Vr_i*(∂Ir_i/∂Vr_j) + Vi_i*(∂Ii_i/∂Vr_j)
    #             = (j==i ? Ir_ref[i] : 0) + Vr_i*G_ij + Vi_i*B_ij
    # ∂P_i/∂Vi_j = δ_{ij}*Ii_i + Vr_i*(∂Ir_i/∂Vi_j) + Vi_i*(∂Ii_i/∂Vi_j)
    #             = (j==i ? Ii_ref[i] : 0) + Vr_i*(-B_ij) + Vi_i*G_ij
    # Q_i = Vi_i*Ir_i - Vr_i*Ii_i
    # ∂Q_i/∂Vr_j = δ_{ij}*(-Ii_i) + Vi_i*(∂Ir_i/∂Vr_j) - Vr_i*(∂Ii_i/∂Vr_j)
    #             = (j==i ? -Ii_ref[i] : 0) + Vi_i*G_ij - Vr_i*B_ij
    # ∂Q_i/∂Vi_j = δ_{ij}*(Ir_i) + Vi_i*(∂Ir_i/∂Vi_j) - Vr_i*(∂Ii_i/∂Vi_j)
    #             = (j==i ? Ir_ref[i] : 0) + Vi_i*(-B_ij) - Vr_i*G_ij

    J_P_Vr = zeros(Float64, n, n)
    J_P_Vi = zeros(Float64, n, n)
    J_Q_Vr = zeros(Float64, n, n)
    J_Q_Vi = zeros(Float64, n, n)

    for i in 1:n
        Vr_i = Vr_ref[i]
        Vi_i = Vi_ref[i]
        Ir_i = Ir_ref[i]
        Ii_i = Ii_ref[i]
        for j in 1:n
            Gij = G[i,j]
            Bij = B[i,j]
            J_P_Vr[i,j] = (j==i ? Ir_i : 0.0) + Vr_i*Gij + Vi_i*Bij
            J_P_Vi[i,j] = (j==i ? Ii_i : 0.0) + Vr_i*(-Bij) + Vi_i*Gij
            J_Q_Vr[i,j] = (j==i ? -Ii_i : 0.0) + Vi_i*Gij - Vr_i*Bij
            J_Q_Vi[i,j] = (j==i ? Ir_i : 0.0) + Vi_i*(-Bij) - Vr_i*Gij
        end
    end

    # 线性化：P ≈ P_ref + J_P_Vr*(Vr - Vr_ref) + J_P_Vi*(Vi - Vi_ref)
    #          Q ≈ Q_ref + J_Q_Vr*(Vr - Vr_ref) + J_Q_Vi*(Vi - Vi_ref)
    # 把 ΔVr = SVr*vec(X) - Vr_ref，ΔVi = SVi*vec(X) - Vi_ref
    # 链接约束：SP*vec(X) ≈ P_ref + J_P_Vr*(SVr*vec(X) - Vr_ref) + J_P_Vi*(SVi*vec(X) - Vi_ref)
    # 移项，得到 A_phys * vec(X) = b_phys（P块）
    A_P = SP - (J_P_Vr*SVr + J_P_Vi*SVi)
    b_P = P_ref - J_P_Vr*Vr_ref - J_P_Vi*Vi_ref

    A_Q = SQ - (J_Q_Vr*SVr + J_Q_Vi*SVi)
    b_Q = Q_ref - J_Q_Vr*Vr_ref - J_Q_Vi*Vi_ref

    # 合并
    A_phys = vcat(A_P, A_Q)
    b_phys = vcat(b_P, b_Q)

    return A_phys, b_phys
end

