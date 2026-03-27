# GP Study 数学模型与算例设计文档

## 1. 概述

`gp_study.jl` 实现了**两阶段配电系统状态估计（DSSE）**框架中的**第一阶段**：基于多任务高斯过程（MTGP）与线性插值的传感器数据补全与对比评估。

**核心思路：** 在异构传感器（PMU / SCADA / AMI）的稀疏、有噪、不完整观测下，利用多任务学习跨传感器关联来恢复缺失数据，并量化预测不确定性。

---

## 2. 数学模型

### 2.1 多任务高斯过程（ICM 核）

对于 $S$ 个传感器任务，任务 $s$ 有 $n_s$ 个观测 $\{(x_i^{(s)}, y_i^{(s)})\}$，分别为时间戳和归一化量测值。

**ICM 核函数：**
$$
\text{Cov}[f_s(x), f_t(x')] = B_{st} \cdot k_{\text{time}}(x, x') + \delta_{st} \cdot k_{\text{local},s}(x, x')
$$

其中：
- $B = LL^\top \in \mathbb{R}^{S \times S}$ 为任务相关矩阵（Cholesky 参数化）
- $k_{\text{time}}(x, x') = \sigma_{\text{time}}^2 \exp\left(-\frac{(x-x')^2}{2\ell_{\text{time}}^2}\right)$ 为全局共享 RBF 核
- $k_{\text{local},s}(x, x') = \sigma_{\text{local},s}^2 \exp\left(-\frac{(x-x')^2}{2\ell_{\text{local},s}^2}\right)$ 为各传感器独立 RBF 核

**观测模型：**
$$
y_i^{(s)} = f_s(x_i^{(s)}) + \varepsilon_i^{(s)}, \quad \varepsilon_i^{(s)} \sim \mathcal{N}(0, \sigma_{\text{noise},s}^2)
$$

**共享均值函数：** 两层全连接网络 $m_\theta(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$

**训练目标：** 最小化联合负对数边际似然
$$
\mathcal{L}(\Theta) = \frac{1}{2} r^\top K^{-1} r + \frac{1}{2} \ln|K| + \frac{N}{2}\ln(2\pi)
$$
其中 $r = y - m_\theta(x)$，使用 Adam 优化器 + 早停。

### 2.2 预测分布

对于测试点 $x_*$（任务 $s$）：

$$
\mu_*^{(s)} = m_\theta(\tilde{x}_*) + k_*^\top \alpha, \quad \sigma_*^{(s)\,2} = k_{**} - v^\top v
$$

其中 $\alpha = L_K^{-\top}(L_K^{-1} r)$，$v = L_K^{-1} k_*$。反归一化后得到原始尺度的预测均值和方差。

### 2.3 线性插值（基线方法）

对每个传感器在归一化域内构建一维线性插值器（`Interpolations.jl` 的 `LinearInterpolation`），预测时执行 归一化→插值→反归一化 单遍融合运算，避免临时数组分配。

### 2.4 数据归一化

$$
\tilde{x} = \frac{x - \mu_x}{\sigma_x}, \quad \tilde{y}^{(s)} = \frac{y^{(s)} - \mu_y^{(s)}}{\sigma_y^{(s)}}
$$

---

## 3. 算例设计

### 3.1 测试系统

IEEE 123 节点三相不平衡配电系统。传感器配置通过 `FAD10_config_ieee123()` 定义：
- **PMU：** 高频同步相量测量
- **SCADA：** 中频监控数据
- **AMI：** 低频智能电表

### 3.2 数据来源

18 个批次的潮流仿真结果（`pf_out/batch_001.mat` ~ `batch_018.mat`），每个传感器最多截取 200 个时间点。

### 3.3 实验参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `noise_levels` | `[0.01, 0.05]` | 噪声水平：信号标准差的 1% 和 5% |
| `missing_percentages` | `[0.1, 0.2, 0.4, 0.6]` | 缺失比例：10% ~ 60% |
| `n_trials` | `5` | 每个配置独立重复 5 次取平均 |
| `mtgp_epochs` | `100` | MTGP 训练轮次（带早停） |
| `target_sensors` | `[104]` | 指定评估的传感器索引 |

### 3.4 实验流程

```
输入清洁数据
  │
  ├─ 对所有传感器添加高斯噪声（固定种子 seed=42）
  │
  └─ 对每个 (noise_level, missing_pct) 组合：
       │
       ├─ 预生成 n_trials 份缺失数据（顺序执行, 各自不同种子）
       │
       └─ 多线程并行执行 n_trials 次试验：
            │
            ├─ 训练 MTGP（verbose=false）
            ├─ 训练线性插值（verbose=false）
            ├─ 在删除点上评估 RMSE
            └─ 写入 trial_results[t]
       │
       └─ 单线程汇总所有 trial → 计算 mean/std
```

### 3.5 评估指标

- **RMSE**（均方根误差）：$\text{RMSE} = \sqrt{\frac{1}{n}\sum_i (y_i^{\text{true}} - \hat{y}_i)^2}$
- **Improvement**：$\text{Imp} = \frac{\text{RMSE}_{\text{linear}} - \text{RMSE}_{\text{MTGP}}}{\text{RMSE}_{\text{linear}}} \times 100\%$
- 按噪声水平 × 缺失比例 × 传感器类型 三维交叉统计

---

## 4. 多线程并行设计

### 4.1 线程安全策略

| 问题 | 解决方案 |
|------|----------|
| `Random.seed!` 修改全局 RNG，不可并行 | **预生成**：在并行前顺序生成所有 trial 的缺失数据 |
| `push!` 存在数据竞争 | **预分配索引写入**：`trial_results[t]`，每个线程写不同索引 |
| `sensor_detailed_results` 共享 Dict | **线程局部**：每个 trial 使用 `local_sensor_details`，并行结束后单线程聚合 |
| 训练函数内部 println 扰乱进度显示 | 所有 println 均由 `verbose` 参数控制，多线程时传 `verbose=false` |
| ProgressMeter 与多线程 IO 冲突 | 使用 `Threads.Atomic{Int}` 计数器 + `\r` 覆盖式进度显示 |

### 4.2 启动方式

```bash
julia -t 8 case_study/gp_study.jl     # 指定 8 线程
julia -t auto case_study/gp_study.jl   # 自动使用所有 CPU 核心
```

---

## 5. 文件依赖关系

```
gp_study.jl
  ├── src/implement_data.jl          # MultiSensorData 结构体定义
  ├── ios/read_mat.jl                # 读取 MAT 批次数据
  ├── src/extract_requested_dataset_multibatch.jl  # 提取数据集
  ├── src/build_complete_multisensor_data.jl       # 构建多传感器数据
  ├── src/data_processing.jl         # 数据归一化
  ├── src/multi_task_gaussian.jl     # ICM-MTGP 训练 (train_icm_mtgp)
  ├── src/gaussian_prediction.jl     # GP 后验预测 (icm_predict)
  ├── src/linear_imputation.jl       # 线性插值 (train_linear_interpolation, linear_predict)
  ├── src/missing_data_evaluation.jl # 缺失数据生成 (create_missing_data)
  └── data/sensor_location_123.jl    # IEEE 123 节点传感器配置
```

---

## 6. 关键函数说明

| 函数 | 文件 | 功能 |
|------|------|------|
| `evaluate_noise_and_missing_data_impact_corrected` | gp_study.jl | 主评估循环：噪声×缺失×多trial |
| `train_icm_mtgp` | multi_task_gaussian.jl | ICM 多任务 GP 训练（Adam+早停） |
| `icm_predict` | gaussian_prediction.jl | GP 后验均值和方差预测 |
| `train_linear_interpolation` | linear_imputation.jl | 线性插值模型构建 |
| `linear_predict` | linear_imputation.jl | 线性插值预测（融合单遍运算） |
| `create_missing_data` | missing_data_evaluation.jl | 随机移除指定比例数据点 |
| `add_gaussian_noise` | gp_study.jl | 添加与信号成比例的高斯噪声 |
| `generate_sensor_level_report_corrected` | gp_study.jl | 传感器级别详细报告 |
