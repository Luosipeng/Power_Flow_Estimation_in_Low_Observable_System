using LinearAlgebra

"""
  build_linearized_physics(Y::AbstractMatrix{<:Real}, x::AbstractMatrix{<:Real}, root_bus::Int)

基于当前估计 X（m×5，列为 [P,Q,Vr,Vi,|V|]）在AC功率方程处做一阶泰勒展开，
返回一个线性残差模型：
  r_lin(x) ≈ J * (x_vec - x0_vec) + c
其中我们将所有节点的三个残差块堆叠：
  - 功率平衡(P): fP(Vr,Vi) - P = 0
  - 功率平衡(Q): fQ(Vr,Vi) - Q = 0
  - 电压幅值(V): |V| - sqrt(Vr^2 + Vi^2) = 0

返回:
  J::Matrix: 残差对变量 x 的雅可比（维度 (3m) × (5m)）
  c::Vector: 常数项 (3m)
  x0::Vector: 当前展开点的变量向量 (5m)
注意:
  - 参考母线的 Vr,Vi 可选定固定，不在残差中约束（或以弱权重加入）
  - Y 为实部G与虚部B已拆开时传入 real(Y), imag(Y)
"""
function build_linearized_physics(G::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real},
                                  X::AbstractMatrix{<:Real}, root_bus::Int)

    m, n = size(X)
    @assert n == 5 "X must be m×5 with columns [P,Q,Vr,Vi,|V|]"
    P = X[:,1]; Q = X[:,2]; Vr = X[:,3]; Vi = X[:,4]; Vmag = X[:,5]

    # 计算AC功率：Pcalc = Vr .* (G*Vr - B*Vi) + Vi .* (B*Vr + G*Vi)
    #           Qcalc = Vi .* (G*Vr - B*Vi) - Vr .* (B*Vr + G*Vi)
    Gr = Matrix(G); Br = Matrix(B)
    m == size(Gr,1) || error("G size mismatch")
    # 预先计算电流成分
    t1 = Gr*Vr .- Br*Vi    # real_curr
    t2 = Br*Vr .+ Gr*Vi    # imag_curr
    Pcalc = Vr .* t1 .+ Vi .* t2
    Qcalc = Vi .* t1 .- Vr .* t2

    # 残差定义
    rP = Pcalc .- P
    rQ = Qcalc .- Q
    rV = Vmag .- sqrt.(Vr.^2 .+ Vi.^2)

    # 构造对变量的偏导：x = [P; Q; Vr; Vi; Vmag]（长度5m）
    # 对于每个节点 i，残差对本地变量与全局的导数:
    # d rP / dP = -I, d rP / dQ = 0
    # d rQ / dQ = -I, d rQ / dP = 0
    # d rV / dVmag = +I, d rV / dVr = -(Vr / sqrt(Vr^2+Vi^2)), d rV / dVi = -(Vi / sqrt(Vr^2+Vi^2))
    # d rP / dVr, d rP / dVi, d rQ / dVr, d rQ / dVi 需按AC潮流一阶导数推导：
    # 对任意 i,k:
    #   ∂t1_i/∂Vr_k = G[i,k], ∂t1_i/∂Vi_k = -B[i,k]
    #   ∂t2_i/∂Vr_k = B[i,k], ∂t2_i/∂Vi_k =  G[i,k]
    # rP_i = Vr_i * t1_i + Vi_i * t2_i - P_i
    # ∂rP_i/∂Vr_k = δ_{ik} * t1_i + Vr_i * ∂t1_i/∂Vr_k + δ_{ik} * t2_i + Vi_i * ∂t2_i/∂Vr_k
    #             = δ_{ik}*(t1_i + t2_i) + Vr_i*G[i,k] + Vi_i*B[i,k]
    # ∂rP_i/∂Vi_k = Vr_i*∂t1_i/∂Vi_k + δ_{ik}*t2_i + Vi_i*∂t2_i/∂Vi_k + δ_{ik}*t1_i
    #             = δ_{ik}*(t1_i + t2_i) + (-Vr_i)*B[i,k] + Vi_i*G[i,k]
    # rQ_i = Vi_i * t1_i - Vr_i * t2_i - Q_i
    # ∂rQ_i/∂Vr_k = Vi_i*G[i,k] - δ_{ik}*t2_i - Vr_i*B[i,k]
    # ∂rQ_i/∂Vi_k = δ_{ik}*t1_i + Vi_i*(-B[i,k]) - Vr_i*G[i,k]

    # 为了构造稀疏大雅可比，使用块结构拼接
    Zm = zeros(m,m)
    Im = I(m)

    # d rP / dP = -I, d rQ / dQ = -I, d rV / dV = +I
    d_rP_dP = -Matrix{Float64}(Im)
    d_rQ_dQ = -Matrix{Float64}(Im)
    d_rV_dV =  Matrix{Float64}(Im)

    # d rV / dVr, d rV / dVi
    eps = 1e-9
    denom = sqrt.(Vr.^2 .+ Vi.^2) .+ eps
    d_rV_dVr = -Diagonal(Vr ./ denom)
    d_rV_dVi = -Diagonal(Vi ./ denom)

    # d rP / dVr, d rP / dVi, d rQ / dVr, d rQ / dVi
    d_rP_dVr = zeros(m,m)
    d_rP_dVi = zeros(m,m)
    d_rQ_dVr = zeros(m,m)
    d_rQ_dVi = zeros(m,m)
    for i in 1:m
        for k in 1:m
            δik = (i==k) ? 1.0 : 0.0
            d_rP_dVr[i,k] = δik*(t1[i] + t2[i]) + Vr[i]*Gr[i,k] + Vi[i]*Br[i,k]
            d_rP_dVi[i,k] = δik*(t1[i] + t2[i]) + (-Vr[i])*Br[i,k] + Vi[i]*Gr[i,k]
            d_rQ_dVr[i,k] = Vi[i]*Gr[i,k] - δik*t2[i] - Vr[i]*Br[i,k]
            d_rQ_dVi[i,k] = δik*t1[i] + Vi[i]*(-Br[i,k]) - Vr[i]*Gr[i,k]
        end
    end

    # 组装J（3m × 5m），行顺序: [rP; rQ; rV]，列顺序: [P; Q; Vr; Vi; V]
    J = zeros(3m, 5m)
    # rP块
    J[1:m,                1:m]      .= d_rP_dP
    J[1:m,                m+1:2m]   .= Zm
    J[1:m,                2m+1:3m]  .= d_rP_dVr
    J[1:m,                3m+1:4m]  .= d_rP_dVi
    # rQ块
    J[m+1:2m,             1:m]      .= Zm
    J[m+1:2m,             m+1:2m]   .= d_rQ_dQ
    J[m+1:2m,             2m+1:3m]  .= d_rQ_dVr
    J[m+1:2m,             3m+1:4m]  .= d_rQ_dVi
    # rV块
    J[2m+1:3m,            1:m]      .= Zm
    J[2m+1:3m,            m+1:2m]   .= Zm
    J[2m+1:3m,            2m+1:3m]  .= d_rV_dVr
    J[2m+1:3m,            3m+1:4m]  .= d_rV_dVi
    J[2m+1:3m,            4m+1:5m]  .= d_rV_dV

    # 常数项 c：r ≈ J*(x - x0) + c 使得在 x0 时 r0 = c
    r0 = vcat(rP, rQ, rV)
    x0 = vcat(P, Q, Vr, Vi, Vmag)
    c = r0 .- J * zeros(5m) # 即 c = r0

    # 参考母线软处理：可通过稍后物理精度对root行加更小权重；这里不零化
    return J, c, x0
end