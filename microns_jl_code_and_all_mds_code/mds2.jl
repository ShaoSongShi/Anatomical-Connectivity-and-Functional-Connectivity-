
include("/home/user/ShenRuihong/bishe/ERM-scale-main/src/util.jl")
using Distributions
using StatsPlots # 提供直方图和密度图的便捷接口
using MATLAB
using CSV, DataFrames
using Distances
using Parameters, SpecialFunctions, StatsBase
using LinearAlgebra
using Plots
using MultivariateStats  # 用于初始化
using ManifoldLearning   # 【核心】用于 Sammon 算法
using Clustering
using Colors 

# ==========================================
# 1. 数据读取与预处理
# ==========================================

# 1.1 读取相关性矩阵
# 假设文件结构：第一列是ID，第一行是ID，左上角是标签"pt_root_id"
corr_matrix_path = "/home/user/ShenRuihong/bishe/corr_principal_validation/coupling_correlation_matrix.csv"
df_raw = CSV.read(corr_matrix_path, DataFrame)

# 提取 ID 列表 (第一列的所有数据)
ids_matrix = df_raw[:, 1] 

# 提取数值矩阵 (去掉第一列，转为 Matrix{Float64})
Corr = Matrix{Float64}(df_raw[:, 2:end])

# 1. 获取非对角线元素的索引 (逻辑矩阵，对角线为 false，其余为 true)
non_diag_mask = .~I(size(Corr, 1)) 
# 2. 提取非对角线元素
off_diag_elements = Corr[non_diag_mask]
# 3. 统计负数个数
neg_count = count(<(0), off_diag_elements)
# 4. 计算总元素个数
total_count = length(off_diag_elements)
# 5. 计算占比
neg_ratio = neg_count / total_count
println("矩阵 Corr (非对角线) 负数统计:")
println("负数个数: $neg_count")
println("总元素个数: $total_count")
println("负数占比: $(neg_ratio * 100)%")

# 1.2 读取细胞类型数据
cell_info_path = "/home/user/ShenRuihong/bishe/connectomics_at_cosyne-main/docs/resources/data/v1718_cell_info.csv"
df_cell_info = CSV.read(cell_info_path, DataFrame)

# ==========================================
# 2. 数据清洗与对齐 (核心修改部分)
# ==========================================

# 2.1 统一 ID 格式为字符串，防止 Int vs String 匹配失败
ids_matrix_str = string.(ids_matrix)
# 确保细胞信息表里也有字符串格式的 pt_root_id
df_cell_info[!, :pt_root_id_str] = string.(df_cell_info.pt_root_id)

# 2.2 筛选：只保留在相关性矩阵中存在的细胞
# 使用 in 函数进行匹配
mask = in.(df_cell_info.pt_root_id_str, Ref(ids_matrix_str))
df_filtered = df_cell_info[mask, :]

# --- 记录清洗前的数量 ---
count_before_clean = nrow(df_filtered)
println("📊 数据清洗统计：")
println("   匹配到矩阵中的细胞总数: $count_before_clean")

# 2.3 删除缺失数据
# 去除 cell_type 为 missing, nothing 或空字符串的行
# 构建布尔索引
valid_mask = .!ismissing.(df_filtered.cell_type) .& 
            .!isnothing.(df_filtered.cell_type) .& 
            (df_filtered.cell_type .!= "")

df_clean = df_filtered[valid_mask, :]

# --- 输出删除数量 ---
count_removed = count_before_clean - nrow(df_clean)
println("   因缺失 cell_type 被剔除的细胞数: $count_removed")
println("   最终保留的有效细胞数: $(nrow(df_clean))")

# 2.3 删除缺失数据
# 去除 cell_type 为 missing, nothing 或空字符串的行
# df_clean = df_filtered[.!ismissing.(df_filtered.cell_type) .& 
#                     .!isnothing.(df_filtered.cell_type) .& 
#                     (df_filtered.cell_type .!= ""), :]

# 2.4 顺序对齐 (至关重要！)
# 我们需要让 df_clean 的行顺序 与 Corr 矩阵的行顺序 (ids_matrix) 完全一致
# 创建一个映射：ID -> 矩阵中的行索引
id_to_index = Dict(id => i for (i, id) in enumerate(ids_matrix_str))

# 1. 先检查 df_clean 是否为空，防止后续操作报错
if nrow(df_clean) == 0
    @error "错误：数据清洗后没有剩余细胞！请检查 ID 匹配情况。"
else
    # 2. 创建一个临时列 :sort_idx，存储每个细胞在矩阵中的行号
    # 使用 getindex.(Ref(id_to_index), df_clean.pt_root_id_str) 可以高效地向量化查找
    df_clean[!, :sort_idx] = getindex.(Ref(id_to_index), df_clean.pt_root_id_str)

    # 3. 按临时列排序
    sort!(df_clean, :sort_idx)

    # 4. 删除临时列 (可选，为了保持数据框整洁)
    select!(df_clean, Not(:sort_idx))
end

# 提取最终排序好的细胞类型标签
# 此时，cell_types[i] 对应的就是 Corr[i, :] 对应的细胞类型
cell_types = df_clean.cell_type

# 2.5 【关键修改】裁剪矩阵
# 获取最终保留下来的 ID 列表（这 903 个）
final_ids = df_clean.pt_root_id_str

# 找到这些 ID 在原始矩阵中的行号
keep_indices = [id_to_index[id] for id in final_ids]

# 裁剪矩阵 Corr 和 ids_matrix
Corr_raw = copy(Corr)
Corr = Corr[keep_indices, keep_indices] # 因此Corr行的顺序与cell type一致
ids_matrix = ids_matrix[keep_indices]

# 更新 N 的大小
N = size(Corr)[1]

# 检查对齐结果
println("矩阵维度: $(size(Corr))")
println("匹配到的细胞数量: $(length(cell_types))")
if size(Corr, 1) != length(cell_types)
    @error "错误：矩阵行数与匹配到的细胞数量不一致！"
else
    println("数据对齐成功！")
end

# ==========================================
# 3. 模型计算
# ==========================================
n = 2
d = n
# ρ, μ = Parameter_estimation(Corr, n = n)
ρ, μ = Parameter_estimation(Corr_raw, n = n)
# N = size(Corr)[1]
N = size(Corr_raw)[1]
L = (N/ρ)^(1/n)
p = ERMParameter(;N = N, L = L, ρ = ρ, n = n, ϵ = 0.03125, μ = μ, ξ = 10^18, β = 0, σ̄² = 1, σ̄⁴ = 1)
println("ERM参数：L=$(L)，p=$(p)，d=$(d),rho=$(ρ),mu=$(μ)")
ϵ = 0.03125
ξ = 10^18
β = 0
σ̄² = 1
σ̄⁴ = 1
# C = abs.(Corr)
C = abs.(Corr_raw)
D = copy(C)



# ==========================================
# 6. 分布拟合对比：数据 vs 模型
# ==========================================

# 6.1 提取非对角线元素 (数据源)
# 注意：Corr_raw 是原始的相关性矩阵（或经过 abs 处理后的矩阵）
# 我们需要将其展平并去除对角线元素
N = size(Corr_raw, 1)
# non_diag_mask = .~I(N) 
# corr_values = C[non_diag_mask] # 使用你代码中的 C (即 abs.(Corr_raw))
corr_values = C

# 过滤掉极罕见的异常值（如 NaN 或 Inf）以便绘图
corr_values = corr_values[.!isnan.(corr_values) .&& .!isinf.(corr_values)]

# 6.2 绘制数据的经验分布 (直方图 + 核密度估计)
p_dist = plot(title="Distribution: Data (Histogram) vs Model Fit (Line)", 
              xlabel="Coupling Alignment / Correlation", 
              ylabel="Density", 
              legend=:topright)

# 绘制数据的直方图 (归一化为概率密度)
# normalize=true 使得 y 轴为密度而非频数
histogram!(corr_values, 
           bins=100, 
           normalize=true, 
           alpha=0.6, 
           label="Data (Empirical)", 
           color=:blue)

# 叠加数据的核密度估计 (KDE)，平滑展示数据分布
density!(corr_values, 
         linewidth=2, 
         label="Data (KDE)", 
         color=:darkblue)

# 6.3 绘制模型拟合的理论分布 f(x)
# 根据论文公式 f(x) = ϵ^μ * (ϵ^2 + x^2)^(-μ/2)
# 注意：这里的 x 代表空间距离。但在我们的拟合中，C ≈ f(D)，即相关性对应于距离的函数。
# 为了在同一个 Correlation 轴上对比，我们需要反解或者直接绘制 f(x) 的形状。
# 但是，由于 f(x) 是单调递减函数，且 x >= 0，f(x) 的取值范围是 (0, ϵ^μ]。

# 策略：在合理的距离范围内生成 x，计算对应的 f(x) 值，然后统计 f(x) 的分布。
# 或者更直观地，我们绘制 f(x) 的概率密度函数（如果可能），但通常我们直接画 f(x) 曲线在数据散点图上。

# 这里采用更直观的方法：绘制理论曲线 f(x) 的形状，并与数据的直方图对比。
# 由于 f(x) 是确定性函数，它本身没有“概率密度”，但我们可以通过变换变量推导。
# 为了简化，我们绘制 f(x) 在其定义域上的“理论直方图”采样。

# 生成样本距离 x (假设 x 在 [0, L] 均匀分布，或者根据实际距离分布采样)
# 这里假设距离 x 服从某种分布，简单起见，我们生成 f(x) 的值域分布
num_samples = 10000

# 根据论文推导，距离 D 与相关性 C 的关系大致为：C = f(D)
# 我们已经计算出了参数 ϵ 和 μ
# 生成模拟的距离 (这里假设距离在有效范围内)
# 注意：f(x) 是单调递减的，x 越大，f(x) 越小

# 生成模拟的距离值 (简单的均匀采样或对数采样)
x_sim = range(1e-5, L*2, length=num_samples) 

# 计算理论函数值 C_model = f(x)
C_model = @. ϵ^μ * (ϵ^2 + x_sim^2)^(-μ/2)

# 过滤掉极小的值，只保留有意义的部分
C_model = C_model[C_model .> 1e-5]

# 绘制模型预测值的直方图 (理论分布)
# normalize=true 使其与数据密度在同一量级
histogram!(C_model, 
           bins=100, 
           normalize=true, 
           alpha=0.4, 
           label="Model (f(x) Sampling)", 
           color=:red,
           linestyle=:dash)

# 6.4 保存图像
savefig(p_dist, "/home/user/ShenRuihong/bishe/corr_principal_validation/dist_fit_comparison.png")
println("✅ 分布对比图已生成！")

# ==========================================
# 7. (可选) 打印拟合优度指标
# ==========================================
# 计算 KL 散度 (需要对齐 bin)
# 或者简单的相关系数
# 这里简单输出两个分布的均值和标准差对比
println("📊 分布统计对比：")
println("   数据 (Data) - 均值: $(mean(corr_values)), 标准差: $(std(corr_values))")
println("   模型 (Model) - 均值: $(mean(C_model)), 标准差: $(std(C_model))")

# 定义函数
function corr(D,p)  
    @unpack μ, ϵ, β = p 
    C_orr = ϵ^μ*(D.^2 .+ϵ.^2).^(-μ/2)
    return C_orr
end

function find_D(C,p)
    @unpack μ, ϵ, β, L = p 
    L = L
    D .= ϵ*sqrt.(abs.(C.^(-2/μ) .- 1))
    D[D.>L] .= L*log.((D[D.>L] ./L)) .+L
    println("截断数值为$(L)")
    return D
end


D = find_D(C, p)

println("="^50)
println("D 矩阵检查结果：")
println("矩阵大小：", size(D))
println("是否包含 NaN：", any(isnan, D))
println("是否包含 Inf：", any(isinf, D))
println("是否全为有限数值：", all(isfinite, D))
println("最小值：", minimum(D[isfinite.(D)]))
println("最大值：", maximum(D[isfinite.(D)]))
println("="^50)

D = D - Diagonal(D) # 计算不涉及重排列，依旧与celltype一致
D = (D+D')/2

# ==========================================
# 4. MDS 计算
# ==========================================
# W = ones(N,N)
# W[Corr .< 0] .= 0.01
# 原代码使用matlab函数: 
X, stress = mdscale(D, n = n, criterion = "sammon") # X的第i个结果就是从X的第i行计算出来的，celltype依旧对应
println("X矩阵维度: $(size(X))")
println("✅ Sammon 映射完成。stress=$stress")
CSV.write("/home/user/ShenRuihong/bishe/corr_principal_validation/distance_matrix_stress.csv", DataFrame(X, :auto))

# ==========================================
# 5. 绘图与输出
# ==========================================

# 5.1 聚类排序 (用于热图)
raw_dist = exp.(-100 * Corr)
sym_dist = (raw_dist + raw_dist') / 2.0
R = hclust(sym_dist, linkage = :average)

# 5.2 绘图：按细胞类型着色
# 注意：X 的行顺序与 cell_types 是一一对应的
p_scatter = scatter(X'[:, 1], X'[:, 2], 
            group = cell_types,       # 按细胞类型分组
            markersize = 4,           
            markerstrokewidth = 0,    # 去掉描边
            title = "Point Cloud (Colored by Cell Type)",
            alpha = 0.6,
            legend = :topleft,        
            framestyle = :box,
            palette = :tab10)         # 使用 tab10 调色板

savefig(p_scatter, "/home/user/ShenRuihong/bishe/corr_principal_validation/point_cloud_by_type.png")


# 5.3 保存热图 (按聚类顺序)
# heatmap(Corr[R.order, R.order], theme = :dark, clim = (0,1), title = "Correlation (Clustered)")
heatmap(Corr, theme = :dark, clim = (0,1), title = "Correlation (Clustered)")
savefig("/home/user/ShenRuihong/bishe/corr_principal_validation/corr.png")

# 5.4 重构相关性矩阵
if n == 1.0
    D2 = pairwise(Euclidean(), X')
else
    D2 = pairwise(Euclidean(), X', dims=1)
end
C2 = corr(D2, p)

# heatmap(C2[R.order, R.order], theme = :dark, clim = (0,1), title = "Refactored Correlation (Clustered)")
heatmap(C2, theme = :dark, clim = (0,1), title = "Refactored Correlation (Clustered)")
savefig("/home/user/ShenRuihong/bishe/corr_principal_validation/corr_refactoring.png")

# 5.5 特征值密度图
λ_sim, p_sim = eigendensity(C2, correction = false, λ_min = 0.5)
λ_id = findall(λ_sim .> 0.1)
plot(λ_sim[λ_id], p_sim[λ_id], label="Reconstructed", xlabel = L"\lambda", ylabel = "pdf", xaxis = :log, yaxis = :log)

λ_sim_orig, p_sim_orig = eigendensity(Corr, correction = false, λ_min = 0.5)
plot!(λ_sim_orig, p_sim_orig, label="Original")
plot!(title=L"L = %$(round(p.L,digits=3)), \mu = %$(round(p.μ ,digits=3))")

savefig("/home/user/ShenRuihong/bishe/corr_principal_validation/lambda_density.png")
# ==========================================
# 5. 绘图与输出 (Julia 原生版，可直接运行)
# ==========================================

# --- 5.1 聚类排序 ---
raw_dist = exp.(-100 * Corr)
sym_dist = (raw_dist + raw_dist') / 2.0
R = hclust(sym_dist, linkage=:average)
order = R.order  # 聚类排序结果

# 对相关性矩阵进行排序
Corr_ordered = Corr[order, order]
ct_ordered = string.(cell_types)[order]

# --- 5.2 细胞类型颜色映射 ---
unique_ct = unique(ct_ordered)
ct_colors = distinguishable_colors(length(unique_ct), [RGB(1,1,1), RGB(0,0,0)])
color_map = Dict(zip(unique_ct, ct_colors))
cell_colors = [color_map[ct] for ct in ct_ordered]

# --- 5.3 绘制热图 + 顶部颜色条 ---
p = plot(
    layout = @layout([a{0.05h}; b]),
    size = (1000, 800),
    margin = 5mm
)

# 顶部：细胞类型颜色条
heatmap!(
    p[1],
    reshape(1:length(ct_ordered), 1, :),
    color = reshape(cell_colors, 1, :),
    legend = false,
    axis = false
)

# 主热图
heatmap!(
    p[2],
    Corr_ordered,
    color = :RdBu,
    title = "Correlation Matrix",
    xlabel = "",
    ylabel = "",
    axis = false,
    colorbar_title = "Corr"
)

# 显示图片
display(p)

# 保存图片
savefig(p, "correlation_heatmap.png")