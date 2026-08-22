# include("/Users/ruiofpoems/Desktop/毕业设计/bishe/bishe/ERM-scale-main/src/util.jl")
# 修复段错误核心代码
# using RCall
using Distributions
using StatsPlots
# using MATLAB
using CSV, DataFrames
using Distances
using Parameters, SpecialFunctions, StatsBase
using LinearAlgebra
using Plots
using MultivariateStats
using ManifoldLearning
using Clustering
using Colors
using Distributions
using Statistics, LinearAlgebra, SpecialFunctions
using StatsBase
# ==========================================
# 1. 数据读取与预处理
# ==========================================

# ====================== 配置路径 ======================
corr_matrix_path = "/Users/ruiofpoems/Desktop/毕业设计/bishe/bishe/corr_principal_validation/results/syn_mat_ct/coupling_correlation_matrix_syn_mat_ct.csv"
cell_info_path = "/Users/ruiofpoems/Desktop/毕业设计/bishe/bishe/connectomics_at_cosyne-main/docs/resources/data/v1718_cell_info.csv"
output_csv_path = "/Users/ruiofpoems/Desktop/毕业设计/bishe/bishe/corr_principal_validation/results/syn_mat_ct/coupling_correlation_matrix_syn_mat_ct_clean.csv" # 新增输出路径

# ==========================================
# 1. 读取相关性矩阵
# ==========================================
println("正在读取矩阵...")
df_raw = CSV.read(corr_matrix_path, DataFrame)

ids_matrix = df_raw[:, 1] # 提取第一列作为 bodyId 列表
Corr = Matrix{Float64}(df_raw[:, 2:end]) # 提取数值矩阵
N = size(Corr, 1)

println("原始矩阵维度: $N x $N")

# ==========================================
# 1.2 读取细胞类型数据 (适配新格式)
# ==========================================
println("正在读取细胞类型...")
df_cell_info = CSV.read(cell_info_path, DataFrame)

# ==========================================
# 2. 数据清洗与对齐 (核心修改部分)
# ==========================================

# 2.1 统一 ID 格式为字符串
# 注意：这里假设矩阵里的ID和CSV里的bodyId可能是Int或String，统一转String防止匹配失败
ids_matrix_str = string.(ids_matrix)

# --- 修改点1：适配新列名 ---
# 原代码: df_cell_info.pt_root_id 
# 新代码: df_cell_info.bodyId
df_cell_info[!, :pt_root_id_str] = string.(df_cell_info.pt_root_id)

# 2.2 筛选：只保留在相关性矩阵中存在的细胞
mask = in.(df_cell_info.pt_root_id_str, Ref(ids_matrix_str))
df_filtered = df_cell_info[mask, :]

count_before_clean = nrow(df_filtered)
println("📊 数据清洗统计：")
println("   匹配到矩阵中的细胞总数: $count_before_clean")

# 2.3 删除缺失数据
# --- 修改点2：适配新列名 ---
# 原代码: df_filtered.cell_type
# 新代码: df_filtered.type
valid_mask = .!ismissing.(df_filtered.cell_type) .& 
            .!isnothing.(df_filtered.cell_type) .& 
            (df_filtered.cell_type .!= "")

df_clean = df_filtered[valid_mask, :]

count_removed = count_before_clean - nrow(df_clean)
println("   因缺失 type 被剔除的细胞数: $count_removed")
println("   最终保留的有效细胞数: $(nrow(df_clean))")

# 2.4 顺序对齐
id_to_index = Dict(id => i for (i, id) in enumerate(ids_matrix_str))

if nrow(df_clean) == 0
    @error "错误：数据清洗后没有剩余细胞！请检查 ID 匹配情况。"
else
    # --- 修改点3：适配新列名 ---
    # 原代码: df_clean.pt_root_id_str
    # 新代码: df_clean.bodyId_str
    df_clean[!, :sort_idx] = getindex.(Ref(id_to_index), df_clean.pt_root_id_str)
    sort!(df_clean, :sort_idx)
    select!(df_clean, Not(:sort_idx))
end

# --- 修改点4：适配新列名 ---
# 提取最终排序好的细胞类型标签
cell_types = df_clean.cell_type

# 2.5 裁剪矩阵
# --- 修改点5：适配新列名 ---
final_ids = df_clean.pt_root_id_str
keep_indices = [id_to_index[id] for id in final_ids]

# 裁剪矩阵
Corr = Corr[keep_indices, keep_indices]
ids_matrix = ids_matrix[keep_indices] # 这里的 ids_matrix 已经是裁剪后的了

# 更新 N 的大小
N = size(Corr)[1]

# # 检查对齐结果
# println("最终矩阵维度: $(size(Corr))")
# println("匹配到的细胞数量: $(length(cell_types))")
# if size(Corr, 1) != length(cell_types)
#     @error "错误：矩阵行数与匹配到的细胞数量不一致！"
# else
#     println("✅ 数据对齐成功！")
# end

# # ==========================================
# # 3. 按 Type 重新排序矩阵 (视觉上分块)
# # ==========================================
# println("正在按 type 对矩阵进行排序分块...")

# # 将细胞类型和ID绑定在一起进行排序
# sort_df = DataFrame(
#     id = ids_matrix, 
#     type = cell_types, 
#     idx = 1:N
# )

# # 先按 type 排序，type 内部保持原来的顺序（可选）
# sort!(sort_df, [:cell_type, :idx])

# # 获取新的顺序
# final_order = sort_df.idx
# final_sorted_ids = sort_df.id
# final_sorted_types = sort_df.cell_type

# # 对矩阵进行最终的重排
# Corr_final_sorted = Corr[final_order, final_order]
# Corr = copy(Corr_final_sorted)
# # ==========================================
# # 4. 保存排序后的 CSV (新增需求)
# # ==========================================
# println("正在保存 CSV...")

# # 创建要保存的 DataFrame
# # 第一列是 bodyId
# df_to_save = DataFrame()
# df_to_save[!, :bodyId] = final_sorted_ids # 第一列命名为 bodyId

# # 将矩阵的每一列作为 DataFrame 的一列，列名就是对应的 bodyId
# for (i, id) in enumerate(final_sorted_ids)
#     df_to_save[!, Symbol(id)] = Corr_final_sorted[:, i]
# end

# # 保存 CSV
# CSV.write(output_csv_path, df_to_save)

# println("🎉 全部完成！")
# println("   排序后的矩阵已保存至: $output_csv_path")

# ==========================================
# 3. 模型计算（全部使用 Corr_raw）
# ==========================================
n = 2
d = n

# ρ, μ = Parameter_estimation(Corr, n = n)
# # ρ, μ = Parameter_estimation(Corr_raw, n = n)
# L = (N/ρ)^(1/n)
# p = ERMParameter(;N = N, L = L, ρ = ρ, n = n, ϵ = 0.03125, μ = μ, ξ = 10^18, β = 0, σ̄² = 1, σ̄⁴ = 1)
# println("ERM参数：L=$(L)，p=$(p)，d=$(d),rho=$(ρ),mu=$(μ)")
ρ = 819.9725870054251
μ = 1.0387089539350047
L = (N/ρ)^(1/n)
ϵ = 0.03125
ξ = 10^18
β = 0
σ̄² = 1
σ̄⁴ = 1

C = abs.(Corr)
C = max.(C, 1e-6) # 防止 C=0 导致 Inf
# C = abs.(Corr_raw)
D = copy(C)

# ==========================================
# 6. 分布拟合验证
# ==========================================


R = copy(C)
triu_idx = triu(trues(N, N), 1)
R = Corr[triu_idx]
R = R[(R .> 0.1) .& (R .< 0.8)]
println("过滤后剩余相关对数: $(length(R))")

# 1. 提取上三角数据（保持代码二原本的筛选逻辑，也可以根据需求改为代码一的 max.(0, ...)）
# cd1 = max.(0, R) 

logR = log.(R)
nbins = 70
bins = range(minimum(logR), stop=maximum(logR), length=nbins)

# 用 StatsBase 拟合直方图（左闭右开，对应 closed=:left）
h = fit(Histogram, logR, bins, closed=:left)
counts = h.weights
edges = h.edges[1]

centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
bin_widths = diff(edges)
p = counts ./ (sum(counts) .* bin_widths)   # 概率密度

# 去除空 bin
mask = counts .> 0
x = centers[mask]          # ln(R)
y = log.(p[mask])          # ln(p_h(R))
# 2. 采用代码一的线性均匀分箱 (bin = 0:0.01:1)
# bin_edges = 0.1:0.01:0.8
# # 拟合直方图
# h = StatsBase.fit(Histogram, cd1, bin_edges)
# # 归一化为概率密度 (PDF)
# h = normalize(h, mode = :pdf)

# # 3. 提取非零的数据点
# # 1. 找出所有计数（weights）大于 0 的位置
# mask = h.weights .> 0  

# # 2. 提取这些位置对应的 x 轴数值（直方图的边沿中心点）和 y 轴概率密度
# # 计算每个箱子的中心点
# bin_centers = (h.edges[1][1:end-1] .+ h.edges[1][2:end]) ./ 2 
# # 提取非零的 x 和 y 值
# cor_vals = bin_centers[mask]  
# p_vals = h.weights[mask]
# valid_indices = 11:(length(cor_vals) - 18)
valid_indices = 20:(length(x) - 0)
# x = log.(cor_vals[valid_indices])  # 对应代码一的自变量 ln(r)
# y = log.(p_vals[valid_indices])    # 对应代码一的因变量 ln(p)
x = x[valid_indices]  # 对应代码一的自变量 ln(r)
y = y[valid_indices]    # 对应代码一的因变量 ln(p)

# ------------------------------
# 3. 最小二乘拟合：y = a + b * x (与代码一完全对齐)
# ------------------------------
X = hcat(ones(length(x)), x)
β = (X' * X) \ (X' * y)
b, k= β[1], β[2] # 这里的 a_fit 对应代码一的 a，b_fit 对应代码一的 b

# 根据公式 (3.6) 反解 μ
d = 2          # 功能空间维度，与你的 MDS 设定一致
μ = -d / (k + 1)

Γ_d2 = gamma(d/2)        # Gamma(d/2)
# ρ = N * (Γ_d2 * μ) / (2 * π^(d/2) * ϵ^d) * exp(b)
ρ = N*μ*exp(b)/(2*π^(d/2)*ϵ^d*gamma(d/2))

println("斜率 k = $k")
println("截距 b = $b")
println("拟合得到的 μ = $μ")
println("拟合得到的 ρ = $ρ")

# ==========================================
# 绘制 log-log 散点图和拟合直线
# ==========================================

# 生成拟合线
x_line = range(minimum(x), maximum(x), length=100)
y_line = b .+ k .* x_line

# 绘图
plot(x, y, seriestype=:scatter, label="Data (bin centers)", 
     xlabel="ln(R)", ylabel="ln(p_h(R))", 
     title="Log-Log Plot of Pairwise Correlation Distribution",
     legend=:topright, markersize=3, markeralpha=0.6)

plot!(x_line, y_line, label="Linear fit (slope = $(round(k, digits=4)))", 
      linewidth=2, color=:red)

# 保存图片（可选）
savefig("loglog_fit3mid200.png")

# # --------------------------
# # 5. 绘制拟合效果（论文风格 log-log 图）
# # --------------------------
# fig = Figure(resolution=(600, 400))
# ax = Axis(fig[1,1], xlabel="ln(R)", ylabel="ln(p_h(R))", title="OLS Fit of Eq. (3.6)")
# scatter!(ax, x, y, label="Data", markersize=8, color=:blue, alpha=0.6)
# lines!(ax, x, b .+ k.*x, label="OLS Fit", color=:red, linewidth=2)
# axislegend(ax, position=:rt)
# display(fig)
# # 提取非对角线相关性值（用于验证）
corr_values = C[.!I(size(C, 1))]  # 只取非对角线元素

p_dist = plot(title="ERM Parameter Fitting Validation\nμ = $(round(μ, digits=3)), ϵ = $(round(ϵ, digits=4)), ρ = $(round(ρ, digits=3))",
              xlabel="Pairwise Correlation",
              ylabel="Probability Density",
              legend=:topright,
              size=(800, 600),
              titlefontsize=11)

# 1. 绘制原始数据的归一化直方图（用于拟合的数据）
histogram!(p_dist, corr_values, 
           bins=0.1:0.01:0.8,
           normalize=true, 
           alpha=0.5, 
           label="Data (Used for Fitting)", 
           color=:blue)

# # 2. 绘制数据的KDE曲线
# density!(p_dist, corr_values, 
#          linewidth=2, 
#          label="Data KDE", 
#          color=:darkblue)

# ==================== 【正确】生成 ERM 理论相关性 R ∈ [0,1] ====================

function generate_ERM_samples_full(μ, d, ρ ,N = 1352,n=100000, ϵ=0.03125)
    # 使用完整的核函数，不只是近似
    # R = f(u) = ϵ^μ * (ϵ^2 + u^2)^(-μ/2)
    
    # 从距离分布采样 u: p(u) ∝ u^(d-1), u ∈ [0, L]
    L = (N/ρ)^(1/d)
    print(L)
    if d == 2
        u = sqrt.(rand(n)) .* L
    end
    # 完整核函数
    R = ϵ^μ .* (ϵ^2 .+ u.^2).^(-μ/2)
    
    R = clamp.(R, 1e-6, 1.0)
    R = R[R .> 1e-5]
    
    return R
end

d=2
C_simulated = generate_ERM_samples_full(μ,d,ρ)

println("生成样本统计:")
println("  最小值: $(minimum(C_simulated))")
println("  最大值: $(maximum(C_simulated))")
println("  均值: $(mean(C_simulated))")


# 绘制理论分布曲线
# density!(p_dist, C_simulated, 
#          linewidth=3, 
#          label="ERM Model (with fitted parameters)", 
#          color=:red, 
#          linestyle=:solid)
# 模拟数据的直方图也用同样的分箱
histogram!(p_dist, C_simulated,
           bins=0.1:0.01:0.8,
           normalize=true,
           alpha=0.3,
           label="ERM Simulated",
           color=:red)

savefig(p_dist, "/Users/ruiofpoems/Desktop/毕业设计/bishe/bishe/corr_principal_validation/param_fitting_validation223mid.png")
println("✅ 参数拟合验证图已生成！")
# ==================== 纵坐标对数刻度图 ====================
p_dist_log = plot(title="ERM Parameter Fitting Validation (Log Scale)\nμ = $(round(μ, digits=3)), ϵ = $(round(ϵ, digits=4)), ρ = $(round(ρ, digits=3))",
                  xlabel="Pairwise Correlation",
                  ylabel="Probability Density (log scale)",
                  legend=:topright,
                  size=(800, 600),
                  titlefontsize=11,
                  yscale=:log10)  # 核心：开启纵坐标log10对数刻度
# 在0.1到0.8区间加半透明红色高亮
# vspan!(p_dist_log, [0.1, 0.8], color=:red, alpha=0.2, label="0.1 ~ 0.8 Range")
# 对数刻度下绘制数据直方图
histogram!(p_dist_log, corr_values, 
           bins=0.1:0.01:0.8,
           alpha=0.5, 
           normalize=true,
           label="Data (Used for Fitting)", 
           color=:blue)

# # 数据KDE曲线
# density!(p_dist_log, corr_values, 
#          linewidth=2, 
#          label="Data KDE", 
#          color=:darkblue)

# # 理论分布曲线
# density!(p_dist_log, C_simulated, 
#          linewidth=3, 
#          label="ERM Model (with fitted parameters)", 
#          color=:red, 
#          linestyle=:solid)
# 模拟数据的直方图也用同样的分箱
histogram!(p_dist_log, C_simulated,
           bins=0.1:0.01:0.8,
           normalize=true,
           alpha=0.3,
           label="ERM Simulated",
           color=:red)

# 保存对数刻度图（新文件名）
savefig(p_dist_log, "/Users/ruiofpoems/Desktop/毕业设计/bishe/bishe/corr_principal_validation/param_fitting_validation_log223mid.png")
println("✅ 对数刻度参数拟合验证图已生成！")
# ==================== 纵坐标对数刻度 + 横坐标截取 0.1~0.8 ====================

corr_values_zoom = corr_values[(corr_values .>= 0.2) .& (corr_values .<= 0.8)]
C_simulated_zoom = C_simulated[(C_simulated .>= 0.2) .& (C_simulated .<= 0.8)]
p_dist_log_zoom = plot(title="ERM Parameter Fitting Validation (Log Scale, 0.2–0.8 Zoom)\nμ = $(round(μ, digits=3)), ϵ = $(round(ϵ, digits=4)), ρ = $(round(ρ, digits=3))",
                  xlabel="Pairwise Correlation (0.2 to 0.8)",
                  ylabel="Probability Density (log scale)",
                  legend=:topright,
                  size=(800, 600),
                  titlefontsize=11,
                  yscale=:log10,
                  ylims=(1e-3, 10))  # log坐标下y轴范围只能为正数
# 在0.1到0.8区间加半透明红色高亮
# vspan!(p_dist_log_zoom, [0.1, 0.8], color=:red, alpha=0.2, label="0.1 ~ 0.8 Range")

# 对数刻度下绘制数据直方图（只显示0.1-0.8区间）
histogram!(p_dist_log_zoom, corr_values, 
           bins=0.1:0.01:0.8,
           normalize=true, 
           alpha=0.5, 
           label="Data (Used for Fitting)", 
           color=:blue)

# # 数据KDE曲线
# density!(p_dist_log_zoom, corr_values, 
#          linewidth=2, 
#          label="Data KDE", 
#          color=:darkblue)

# # 理论分布曲线
# density!(p_dist_log_zoom, C_simulated, 
#          linewidth=3, 
#          label="ERM Model (with fitted parameters)", 
#          color=:red, 
#          linestyle=:solid)
histogram!(p_dist_log_zoom, C_simulated,
           bins=0.1:0.01:0.8,
           normalize=true,
           alpha=0.3,
           label="ERM Simulated",
           color=:red)

# 保存截取区间的图（新文件名，不会覆盖原图）
savefig(p_dist_log_zoom, "/Users/ruiofpoems/Desktop/毕业设计/bishe/bishe/corr_principal_validation/param_fitting_validation_zoom223mid.png")
println("✅ 0.1~0.8截取版对数刻度图已生成！")
# ==========================================
# 函数定义
# ==========================================
function corr(D)
    μ = 1.625983765749227
    ϵ = 0.03125
    β = 0.0
    C_orr = ϵ^μ*(D.^2 .+ϵ.^2).^(-μ/2)
    return C_orr
end

function find_D(C)
    ρ = 5472.960733724306
    μ = 1.625983765749227
    ϵ = 0.03125
    β = 0.0
    L = (N/ρ)^(1/n)
    D = ϵ*sqrt.(abs.(C.^(-2/μ) .- 1))
    D[D.>L] .= L*log.((D[D.>L] ./L)) .+ L
    println("截断数值为$(L)")
    return D
end

# ==========================================
# 计算 D 矩阵 + 自动修复 NaN/Inf
# ==========================================
D = find_D(C)

println("="^50)
println("D 矩阵检查结果：")
println("矩阵大小：", size(D))
println("D的最大值为$(maximum(D))")
println("D的最小值为$(minimum(D))")
println("是否包含 NaN：", any(isnan, D))
println("是否包含 Inf：", any(isinf, D))
println("是否全为有限数值：", all(isfinite, D))

# 自动修复非法值
# D[isnan.(D) .| isinf.(D)] .= 1e-6
D = D - Diagonal(D)
D = (D + D') / 2  # 强制对称
# D[D.>L] .= L*log.((D[D.>L] ./L)) .+ L

println("修复后 NaN：", any(isnan, D))
println("修复后 Inf：", any(isinf, D))
println("D的最大值为$(maximum(D))")
println("D的最小值为$(minimum(D))")
println("D的形状$(size(D))")
println("="^50)
p_D = plot(title="Distribution: D(Histogram)",
              xlabel="D",
              ylabel="Density",
              legend=:topright)
histogram!(D, bins=100, normalize=true, alpha=0.6, color=:blue)
savefig(p_D, "/Users/ruiofpoems/Desktop/毕业设计/bishe/bishe/corr_principal_validation/D_distribution_subset.png")
println("✅ D分布图已生成！")


# ==========================================
# 4. MDS 计算（使用完整矩阵）
# ==========================================

println("🧹 清理内存...")
GC.gc()
sleep(0.5) # 暂停 0.5 秒，让操作系统整理内存

X, stress = mdscale(D, n = n, criterion = "sammon")
X = X'
# R"""
# library(MASS)
# # 转换数据
# D_matrix <- as.matrix($D)
# # Sammon mapping
# result <- sammon(D_matrix, k = $n)
# X <- result$points
# stress <- result$stress
# """
# X = rcopy(R"X")
# stress = rcopy(R"stress")

println("X矩阵维度: $(size(X))")
println("✅ Sammon 映射完成。stress=$stress")

CSV.write("/home/user/ShenRuihong/bishe/corr_principal_validation/fly/fly_distance_matrix_stress_subset.csv", DataFrame(X, :auto))

# ==========================================
# 5. 散点图（全部细胞 + unknown）
# ==========================================
p_scatter = scatter(X[:, 1], X[:, 2],
            group = cell_types,
            markersize = 3,
            markerstrokewidth = 0,
            title = "Point Cloud (All Cells, missing = unknown)",
            alpha = 0.6,
            legend = :topleft,
            framestyle = :box,
            palette = :tab10)

savefig(p_scatter, "/home/user/ShenRuihong/bishe/corr_principal_validation/fly/fly_point_cloud_all_cells_subset.png")

# ==========================================
# 热图
# ==========================================

# ==========================================
# 重构相关性层次聚类验证
# ==========================================
D2 = pairwise(Euclidean(), X, dims=1)
C2 = corr(D2)

# 5.1 聚类排序 (用于热图)
raw_dist = exp.(-100 * Corr)
sym_dist = (raw_dist + raw_dist') / 2.0
R = hclust(sym_dist, linkage = :average)

heatmap(Corr[R.order, R.order], theme=:dark, clim=(0,1), title="Original Correlation")
savefig("/home/user/ShenRuihong/bishe/corr_principal_validation/fly/fly_corr_original_subset.png")

heatmap(C2[R.order, R.order], theme=:dark, clim=(0,1), title="Reconstructed Correlation")
savefig("/home/user/ShenRuihong/bishe/corr_principal_validation/fly/fly_corr_reconstructed_subset.png")

# ==========================================
# 验证细胞类型分割正确性
# ==========================================

# 获取唯一的细胞类型并按字母顺序排序
unique_cell_types = sort(unique(cell_types))
println("细胞类型数量: $(length(unique_cell_types))")
println("各类型细胞数量:")
for ct in unique_cell_types
    println("  $ct: $(count(==(ct), cell_types))")
end

# 创建排序映射
type_to_order = Dict(ct => i for (i, ct) in enumerate(unique_cell_types))

# 为每个细胞分配排序权重
sort_key = [(type_to_order[cell_types[i]], i) for i in 1:length(cell_types)]
sorted_indices = sortperm(sort_key)

# 获取排序后的数据和标签
Corr_sorted = Corr[sorted_indices, sorted_indices]
C2_sorted = C2[sorted_indices, sorted_indices]
cell_types_sorted = cell_types[sorted_indices]

# 计算分割线位置
split_positions = Float64[]
if length(cell_types_sorted) > 1
    change_indices = findall(i -> cell_types_sorted[i] != cell_types_sorted[i-1], 2:length(cell_types_sorted))
    split_positions = [idx - 0.5 for idx in change_indices]
end

# 计算每个细胞类型块的中心位置（用于添加注释）
type_centers = Float64[]
type_names_for_annotation = String[]
if !isempty(split_positions)
    for i in 1:length(split_positions)
        if i == 1
            center = split_positions[1] / 2
        else
            center = (split_positions[i-1] + split_positions[i]) / 2
        end
        push!(type_centers, center)
        idx = round(Int, center)
        if idx >= 1 && idx <= length(cell_types_sorted)
            push!(type_names_for_annotation, cell_types_sorted[idx])
        end
    end
    # 添加最后一个块
    last_center = (split_positions[end] + length(cell_types_sorted)) / 2
    push!(type_centers, last_center)
    idx = round(Int, last_center)
    if idx >= 1 && idx <= length(cell_types_sorted)
        push!(type_names_for_annotation, cell_types_sorted[idx])
    end
end

println("\n" * "="^60)
println("细胞类型分割验证")
println("="^60)

# 检查每个细胞类型的起始和结束位置
for ct in unique_cell_types
    type_indices = findall(x -> x == ct, cell_types_sorted)
    
    if !isempty(type_indices)
        first_idx = minimum(type_indices)
        last_idx = maximum(type_indices)
        count_ct = length(type_indices)
        is_contiguous = all(diff(type_indices) .== 1)
        
        println("细胞类型: $(rpad(ct, 20)) | 数量: $(lpad(count_ct, 4)) | 起始: $(lpad(first_idx, 4)) | 结束: $(lpad(last_idx, 4)) | 连续: $(is_contiguous ? "✓" : "✗")")
        
        if !is_contiguous
            @warn "警告：细胞类型 $ct 在排序后不连续！"
        end
    else
        println("细胞类型: $(rpad(ct, 20)) | 数量: 0 (未找到)")
    end
end

println("="^60)
println("✅ 细胞类型分割验证完成")
println("="^60 * "\n")

# ==========================================
# 按细胞类型排序的热图绘制
# ==========================================

# 创建细胞类型的颜色映射
type_colors = distinguishable_colors(length(unique_cell_types), [RGB(1,1,1), RGB(0,0,0)])
color_map = Dict(zip(unique_cell_types, type_colors))
cell_colors = [color_map[ct] for ct in cell_types_sorted]

# ==========================================
# 并排对比图：原始 vs 重构
# ==========================================
p_comparison = plot(layout = @layout([a{0.06h}; b c; d{0.06h}]), 
                    size = (1600, 800),
                    titlefontsize = 12)

# 顶部的细胞类型颜色条（共享）
heatmap!(p_comparison[1], 
         reshape(1:length(cell_types_sorted), 1, :), 
         color = reshape(cell_colors, 1, :),
         legend = false,
         axis = false,
         yflip = false)

# 原始相关性矩阵
heatmap!(p_comparison[2], 
         Corr_sorted,
         color = :RdBu,
         clim = (0, 1),
         title = "Original Correlation",
         xticks = false,
         yticks = false,
         colorbar_title = "Corr")

# 重构相关性矩阵
heatmap!(p_comparison[3], 
         C2_sorted,
         color = :RdBu,
         clim = (0, 1),
         title = "Reconstructed Correlation",
         xticks = false,
         yticks = false,
         colorbar_title = "Corr")

# 底部的细胞类型颜色条（共享）
heatmap!(p_comparison[4], 
         reshape(1:length(cell_types_sorted), 1, :), 
         color = reshape(cell_colors, 1, :),
         legend = false,
         axis = false,
         yflip = false)

# 为两个热图添加粗分割线
for pos in split_positions
    # 原始矩阵的分割线
    vline!(p_comparison[2], [pos], color=:black, linewidth=2.5, alpha=0.8)
    hline!(p_comparison[2], [pos], color=:black, linewidth=2.5, alpha=0.8)
    
    # 重构矩阵的分割线
    vline!(p_comparison[3], [pos], color=:black, linewidth=2.5, alpha=0.8)
    hline!(p_comparison[3], [pos], color=:black, linewidth=2.5, alpha=0.8)
end

# 设置坐标轴范围
for i in [2, 3]
    xlims!(p_comparison[i], 0.5, length(cell_types_sorted)+0.5)
    ylims!(p_comparison[i], 0.5, length(cell_types_sorted)+0.5)
end

savefig(p_comparison, "/home/user/ShenRuihong/bishe/corr_principal_validation/fly/fly_corr_comparison_sorted_by_type_subset.png")
println("✅ 并排对比热图已保存")

# ==========================================
# 创建细胞类型图例
# ==========================================
p_legend = plot(size=(800, 600), legend=:outertopright)
for (i, ct) in enumerate(unique_cell_types)
    scatter!(p_legend, [0], [i], 
             color=type_colors[i], 
             label="$(ct) (n=$(count(==(ct), cell_types)))", 
             markersize=8,
             markerstrokewidth=0)
end
xlims!(0, 1)
ylims!(0, length(unique_cell_types)+1)
title!("Cell Type Legend")
savefig(p_legend, "/home/user/ShenRuihong/bishe/corr_principal_validation/fly/fly_cell_type_legend_subset.png")
println("✅ 细胞类型图例已保存")

println("🎉 按类型排序的热图绘制完成！")

# ==========================================
# 特征值密度图
# ==========================================
λ_sim, p_sim = eigendensity(C2, correction=false, λ_min=0.5)
λ_id = findall(λ_sim .> 0.1)
plot(λ_sim[λ_id], p_sim[λ_id], label="Reconstructed", xlabel=L"\lambda", ylabel="pdf", xaxis=:log, yaxis=:log)

λ_sim_orig, p_sim_orig = eigendensity(Corr, correction=false, λ_min=0.5)
# λ_sim_orig, p_sim_orig = eigendensity(Corr_raw, correction=false, λ_min=0.5)
plot!(λ_sim_orig, p_sim_orig, label="Original")
plot!(title=L"L = %$(round(p.L,digits=3)), \mu = %$(round(p.μ,digits=3))")

savefig("/home/user/ShenRuihong/bishe/corr_principal_validation/fly/fly_lambda_density_subset.png")

println("🎉 全部计算完成！")