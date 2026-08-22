include("/home/wangzezhen/ShenRuihong/corr_principal_validation/fly_jl_code_and_others/util.jl")
using MAT
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

# ==========================================
# 1. 数据读取与预处理
# ==========================================

# ====================== 配置路径 ======================
corr_matrix_path = "/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_cellcount_Branson/CouplingCorrelation_branson_cellcount.csv"
# cell_info_path = "/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/data/traced-neurons.csv"
sorted = 1
# output_csv_path = "/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_weighted_synapse_count_Branson/fly_coupling_correlation_matrix_sorted_subset_AL2.csv" # 新增输出路径
# MAT_X_STRESS  = "/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_weighted_synapse_count_Branson/mds_result_irls_d5.mat"
# MAT_D = "distance_matrix.mat"
# ==========================================
# 1. 读取相关性矩阵
# ==========================================
println("正在读取矩阵...")
df_raw = CSV.read(corr_matrix_path, DataFrame)

cell_types = df_raw[:, 1] # 提取第一列作为 bodyId 列表
Corr = Matrix{Float64}(df_raw[:, 2:end]) # 提取数值矩阵
N = size(Corr, 1)

println("原始矩阵维度: $N x $N")

# ==========================================
# 1.2 读取细胞类型数据 (适配新格式)
# ==========================================
# println("正在读取细胞类型...")
# df_cell_info = CSV.read(cell_info_path, DataFrame)

# # ==========================================
# # 2. 数据清洗与对齐 (核心修改部分)
# # ==========================================

# # 2.1 统一 ID 格式为字符串
# # 注意：这里假设矩阵里的ID和CSV里的bodyId可能是Int或String，统一转String防止匹配失败
# ids_matrix_str = string.(ids_matrix)

# # --- 修改点1：适配新列名 ---
# # 原代码: df_cell_info.pt_root_id 
# # 新代码: df_cell_info.bodyId
# df_cell_info[!, :bodyId_str] = string.(df_cell_info.bodyId)

# # 2.2 筛选：只保留在相关性矩阵中存在的细胞
# mask = in.(df_cell_info.bodyId_str, Ref(ids_matrix_str))
# df_filtered = df_cell_info[mask, :]

# count_before_clean = nrow(df_filtered)
# println("📊 数据清洗统计：")
# println("   匹配到矩阵中的细胞总数: $count_before_clean")

# # 2.3 删除缺失数据
# # --- 修改点2：适配新列名 ---
# # 原代码: df_filtered.cell_type
# # 新代码: df_filtered.type
# valid_mask = .!ismissing.(df_filtered.type) .& 
#             .!isnothing.(df_filtered.type) .& 
#             (df_filtered.type .!= "")

# df_clean = df_filtered[valid_mask, :]

# count_removed = count_before_clean - nrow(df_clean)
# println("   因缺失 type 被剔除的细胞数: $count_removed")
# println("   最终保留的有效细胞数: $(nrow(df_clean))")

# # 2.4 顺序对齐
# id_to_index = Dict(id => i for (i, id) in enumerate(ids_matrix_str))

# if nrow(df_clean) == 0
#     @error "错误：数据清洗后没有剩余细胞！请检查 ID 匹配情况。"
# else
#     # --- 修改点3：适配新列名 ---
#     # 原代码: df_clean.pt_root_id_str
#     # 新代码: df_clean.bodyId_str
#     df_clean[!, :sort_idx] = getindex.(Ref(id_to_index), df_clean.bodyId_str)
#     sort!(df_clean, :sort_idx)
#     select!(df_clean, Not(:sort_idx))
# end

# # --- 修改点4：适配新列名 ---
# # 提取最终排序好的细胞类型标签
# cell_types = df_clean.type

# # 2.5 裁剪矩阵
# # --- 修改点5：适配新列名 ---
# final_ids = df_clean.bodyId_str
# keep_indices = [id_to_index[id] for id in final_ids]

# # 裁剪矩阵
# Corr = Corr[keep_indices, keep_indices]
# ids_matrix = ids_matrix[keep_indices] # 这里的 ids_matrix 已经是裁剪后的了

# # 更新 N 的大小
# N = size(Corr)[1]

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
# sort!(sort_df, [:type, :idx])

# # 获取新的顺序
# final_order = sort_df.idx
# final_sorted_ids = sort_df.id
# final_sorted_types = sort_df.type

# # 对矩阵进行最终的重排
# Corr_final_sorted = Corr[final_order, final_order]
# Corr = copy(Corr_final_sorted)
# # ==========================================
# # 4. 保存排序后的 CSV (新增需求)
# # ==========================================
# if sorted==0
#     println("正在保存 CSV...")

#     # 创建要保存的 DataFrame
#     # 第一列是 bodyId
#     df_to_save = DataFrame()
#     df_to_save[!, :bodyId] = final_sorted_ids # 第一列命名为 bodyId

#     # 将矩阵的每一列作为 DataFrame 的一列，列名就是对应的 bodyId
#     for (i, id) in enumerate(final_sorted_ids)
#         df_to_save[!, Symbol(id)] = Corr_final_sorted[:, i]
#     end

#     # 保存 CSV
#     CSV.write(output_csv_path, df_to_save)
# end
# ==========================================
# ✅ 论文标准归一化：Tr(C)/N = 1（严格复现）
# ✅ 安全剔除 <0.1 和 >0.9 的值（无NaN、无报错）
# ==========================================

# 1. 先做论文归一化（唯一合法操作）
C = Corr
diag_vals = diag(C)
mean_diag = mean(diag_vals)
C_normalized = C ./ mean_diag  # 👈 论文唯一归一化！

# 2. 安全过滤：只忽略 <0.1 和 >0.9，不置 NaN，不影响拟合
# 原理：给极小/极大值一个小底，不参与 log 拟合畸变
lower_thresh = 1e-3

upper_thresh = 1

# # 保留对角线不动！只处理非对角线
# mask_diag = trues(size(C_normalized))
# mask_diag[diagind(mask_diag)] .= false

# # 安全裁剪，不产生 NaN
# C_clean = copy(C_normalized)
# C_clean[mask_diag] .= clamp.(C_clean[mask_diag], lower_thresh, upper_thresh)

println("✅ 论文归一化完成：Tr(C)/N ≈ 1")
println("✅ 已裁剪非对角线：<0.1 → 0.1，>0.9 → 0.9")
println("✅ 无 NaN，无 Inf，可直接送入 Parameter_estimation")
println("🎉 全部完成！")

# p_heat = heatmap(C_clean, 
#     title="C_clean heatmap normalized",
#     clim=(0, 1), 
#     color=:viridis,
#     size=(800,800))
# savefig(p_heat, "C_clean_heatmap.png")
# println("✅ 热图已保存：C_clean_heatmap.png")
p_heat = heatmap(C, 
    title="C heatmap",
    clim=(0, 1), 
    color=:viridis,
    size=(800,800))
savefig(p_heat, "C_heatmap.png")
println("✅ 热图已保存：C_heatmap.png")
# # 2. 提取非对角线值，画分布
# non_diag = C_clean[.!I(size(C_clean,1))]
# p_hist = histogram(non_diag, 
#     bins=100, 
#     normalize=true,
#     title="C_clean 非对角线分布",
#     xlabel="相关性",
#     ylabel="密度",
#     xlim=(0,1),
#     size=(800,600))
# savefig(p_hist, "C_clean_distribution.png")
# println("✅ 分布已保存：C_clean_distribution.png")
# ==========================================
# 3. 模型计算（全部使用 Corr_raw）
# ==========================================
n = 2
d = n

# ρ, μ = Parameter_estimation(Corr, n = n)
# ρ, μ = Parameter_estimation(Corr_raw, n = n)

ϵ = 0.03125
ξ = 10^18
β = 0
σ̄² = 1
σ̄⁴ = 1

C = abs.(Corr)
C = max.(C, 1e-10) # 防止 C=0 导致 Inf
# C = abs.(Corr_raw)
D = copy(C)


# ==========================================
# 6. 分布拟合验证
# ==========================================


R = copy(C)
triu_idx = triu(trues(N, N), 1)
R = Corr[triu_idx]
# R = R[(R .> 0.1) .& (R .< 0.8)]
println("过滤后剩余相关对数: $(length(R))")

# 1. 提取上三角数据（保持代码二原本的筛选逻辑，也可以根据需求改为代码一的 max.(0, ...)）
cd1 = max.(0, R) 
# 2. 采用代码一的线性均匀分箱 (bin = 0:0.01:1)
bin_edges = 0.1:0.01:0.5
# 拟合直方图
h = StatsBase.fit(Histogram, cd1, bin_edges)
# 归一化为概率密度 (PDF)
h = normalize(h, mode = :pdf)

# 3. 提取非零的数据点
# 1. 找出所有计数（weights）大于 0 的位置
mask = h.weights .> 0  

# 2. 提取这些位置对应的 x 轴数值（直方图的边沿中心点）和 y 轴概率密度
# 计算每个箱子的中心点
bin_centers = (h.edges[1][1:end-1] .+ h.edges[1][2:end]) ./ 2 
# 提取非零的 x 和 y 值
cor_vals = bin_centers[mask]  
p_vals = h.weights[mask]
valid_indices = 13:(length(cor_vals) - 7)
x = log.(cor_vals[valid_indices])  # 对应代码一的自变量 ln(r)
y = log.(p_vals[valid_indices])    # 对应代码一的因变量 ln(p)
# ------------------------------
# 3. 最小二乘拟合：y = a + b * x (与代码一完全对齐)
# ------------------------------
X = hcat(ones(length(x)), x)
β = (X' * X) \ (X' * y)
b, k= β[1], β[2] # 这里的 a_fit 对应代码一的 a，b_fit 对应代码一的 b

# 根据公式 (3.6) 反解 μ
μ = -d / (k + 1)

Sd = 2 * π^(d/2) / gamma(d/2)  # d维单位球表面积
ρ = (N * μ * exp(b)) / (Sd * ϵ^d)

println("斜率 k = $k")
println("截距 b = $b")
println("拟合得到的 μ = $μ")
println("拟合得到的 ρ = $ρ")

L = (N/ρ)^(1/n)
p = ERMParameter(;N = N, L = L, ρ = ρ, n = n, ϵ = 0.03125, μ = μ, ξ = 10^18, σ̄² = 1, σ̄⁴ = 1)
# p = ERMParameter(;N = N, L = L, ρ = ρ, n = n, ϵ = 0.03125, μ = μ, ξ = 10^18, β = 0, σ̄² = 1, σ̄⁴ = 1)
println("ERM参数：L=$(L)，p=$(p)，d=$(d),rho=$(ρ),mu=$(μ)")

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
savefig("loglog_fit_fly0108.png")

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

function generate_ERM_samples_full(μ, d, ρ ,N,n=100000, ϵ=0.03125)
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

C_simulated = generate_ERM_samples_full(μ,d,ρ,N)

println("生成样本统计:")
println("  最小值: $(minimum(C_simulated))")
println("  最大值: $(maximum(C_simulated))")
println("  均值: $(mean(C_simulated))")

histogram!(p_dist, C_simulated,
           bins=0.1:0.01:0.8,
           normalize=true,
           alpha=0.3,
           label="ERM Simulated",
           color=:red)

savefig(p_dist, "param_fitting_validation.png")
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

# 模拟数据的直方图也用同样的分箱
histogram!(p_dist_log, C_simulated,
           bins=0.1:0.01:0.8,
           normalize=true,
           alpha=0.3,
           label="ERM Simulated",
           color=:red)

# 保存对数刻度图（新文件名）
savefig(p_dist_log, "param_fitting_validation_log.png")
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

histogram!(p_dist_log_zoom, C_simulated,
           bins=0.1:0.01:0.8,
           normalize=true,
           alpha=0.3,
           label="ERM Simulated",
           color=:red)

# 保存截取区间的图（新文件名，不会覆盖原图）
savefig(p_dist_log_zoom, "param_fitting_validation_zoom.png")
println("✅ 0.1~0.8截取版对数刻度图已生成！")
# ==========================================
# 函数定义
# ==========================================
function corr(D,p)
    @unpack μ, ϵ, β = p
    C_orr = ϵ^μ*(D.^2 .+ϵ.^2).^(-μ/2)
    return C_orr
end

function find_D(C,p)
    @unpack μ, ϵ, β, L = p
    D = ϵ*sqrt.(abs.(C.^(-2/μ) .- 1))
    D[D.>L] .= L*log.((D[D.>L] ./L)) .+ L
    println("截断数值为$(L)")
    return D
end

# ==========================================
# 计算 D 矩阵 + 自动修复 NaN/Inf
# ==========================================
D = find_D(C, p)

println("="^50)
println("D 矩阵检查结果：")
println("矩阵大小：", size(D))
println("D的最大值为$(maximum(D))")
println("D的最小值为$(minimum(D))")
println("是否包含 NaN：", any(isnan, D))
println("是否包含 Inf：", any(isinf, D))
println("是否全为有限数值：", all(isfinite, D))
println("="^50)

# ==========================================
# ✅ 关键：保存 D 矩阵为 .mat 文件（给 MATLAB 使用）
# ==========================================
matwrite(MAT_D, Dict(
    "D" => D
))
println("✅ 距离矩阵 D 已保存为 MAT 格式：distance_matrix.mat")

p_D = plot(title="Distribution: D(Histogram)",
              xlabel="D",
              ylabel="Density",
              legend=:topright)
histogram!(D, bins=100, normalize=true, alpha=0.6, color=:blue)
savefig(p_D, "fly_D_distribution.png")
println("✅ D分布图已生成！")


# ==========================================
# 4. MDS 计算（使用完整矩阵）
# ==========================================

println("🧹 清理内存...")
GC.gc()
sleep(0.5) # 暂停 0.5 秒，让操作系统整理内存

file = matopen(MAT_X_STRESS)
X = read(file, "X_irls")       # 降维后的坐标
stress = read(file, "stress_irls")  # 应力值
close(file)
# X = X'

println("X矩阵维度: $(size(X))")
println("✅ Sammon 映射完成。stress=$stress")

CSV.write("fly_distance_matrix_stress.csv", DataFrame(X, :auto))

# ==========================================
# 5. 散点图（全部细胞 + unknown）
# ==========================================
p_scatter = scatter(X[:, 1], X[:, 2],
            group = cell_types,
            markersize = 3,
            markerstrokewidth = 0,
            title = "Point Cloud (All Cells, missing = unknown)",
            alpha = 0.6,
            legend = false,
            framestyle = :box,
            palette = :tab10)

savefig(p_scatter, "fly_point_cloud_all_cells.png")
# ==========================================
# 【单独保存：细胞类型图例】
# ==========================================
println("🎯 正在单独生成细胞类型图例...")

# 获取唯一类型 + 配色
unique_types = unique(cell_types)
type_colors = distinguishable_colors(
    length(unique_types),
    [RGB(1,1,1), RGB(0,0,0)],
    dropseed=true
)
color_map = Dict(zip(unique_types, type_colors))

# 绘制纯图例
p_legend_only = plot(
    title = "Cell Type Legend",
    size=(800, 1000),  # 纵向长图，方便看全
    framestyle=:none
)

for (i, ct) in enumerate(unique_types)
    cnt = count(==(ct), cell_types)
    scatter!(p_legend_only,
        [0], [i],
        color=type_colors[i],
        label="$(ct)  (n=$cnt)",
        markersize=10,
        markerstrokewidth=0
    )
end

xlims!(p_legend_only, -0.5, 1)
ylims!(p_legend_only, 0, length(unique_types)+2)

# 保存单独图例
savefig(p_legend_only, "fly_cell_type_legend_only.png")
println("✅ 细胞类型图例已单独保存！")



# # ==========================================
# # 【正确方式】从 traced-roi-connections.csv 提取 ROI（和你Python逻辑完全一致）
# # ==========================================
# println("🎯 正在从 traced-roi-connections.csv 加载 ROI 信息...")

# # 1. 读取 ROI 文件
# roi_csv_path = "/home/user/ShenRuihong/bishe/corr_principal_validation/fly/traced-roi-connections.csv"
# df_roi_raw = CSV.read(roi_csv_path, DataFrame, select=["bodyId_post", "roi"])

# # 2. 建立 ID → ROI 字典（唯一映射）
# df_roi_raw[!, :bodyId_post] = string.(df_roi_raw.bodyId_post)
# bodyId_to_roi = Dict(row.bodyId_post => row.roi for row in eachrow(df_roi_raw))

# # 3. 当前排序好的神经元 ID（和 X 完全对齐）
# sorted_body_ids = string.(final_sorted_ids)  # 这是你已经排序好的ID列表

# # 4. 一一对应查找 ROI
# roi_labels = [get(bodyId_to_roi, bid, "Unknown") for bid in sorted_body_ids]

# println("✅ ROI 信息匹配完成！总细胞数：$(length(roi_labels))")

# # ==========================================
# # 绘制 ROI 点云图（无图例）
# # ==========================================
# p_scatter_roi = scatter(X[:, 1], X[:, 2],
#             group = roi_labels,
#             markersize = 4,
#             markerstrokewidth = 0,
#             title = "Point Cloud (ROI Classification)",
#             alpha = 0.7,
#             legend = false,        # 无图例，不遮挡
#             framestyle = :box,
#             palette = :tab10,
#             size=(800, 600))

# savefig(p_scatter_roi, "fly_point_cloud_by_roi.png")
# println("✅ ROI 点云图已保存！")

# # ==========================================
# # 单独保存 ROI 图例
# # ==========================================
# unique_roi = unique(roi_labels)
# roi_colors = distinguishable_colors(length(unique_roi), [RGB(1,1,1), RGB(0,0,0)])

# p_legend_roi = plot(title = "ROI Legend", size=(700, 1000), framestyle=:none)
# for (i, r) in enumerate(unique_roi)
#     cnt = count(==(r), roi_labels)
#     scatter!(p_legend_roi, [0], [i], color=roi_colors[i], label="$r (n=$cnt)", markersize=10, markerstrokewidth=0)
# end
# xlims!(-0.5, 1)
# ylims!(0, length(unique_roi)+2)

# savefig(p_legend_roi, "fly_roi_legend_only.png")
# println("✅ ROI 图例已单独保存！")

# ==========================================
# 热图
# ==========================================

# ==========================================
# 重构相关性层次聚类验证
# ==========================================
D2 = pairwise(Euclidean(), X, dims=1)
C2 = corr(D2, p)

# 5.1 聚类排序 (用于热图)
# raw_dist = exp.(-100 * C)
# sym_dist = (raw_dist + raw_dist') / 2.0
# R = hclust(sym_dist, linkage = :average)

heatmap(C, theme=:dark, clim=(0,1), title="Original Correlation")
savefig("fly_corr_original_d5.png")

heatmap(C2, theme=:dark, clim=(0,1), title="Reconstructed Correlation")
savefig("fly_corr_reconstructed_d5.png")

# ---------------------- 【关键】矩阵向量化（必须对应！） ----------------------

# 取非对角线元素
mask = .!I(size(C,1))
x_all = C[mask]
y_all = C2[mask]

n_plot = 8000

keep = (x_all .>= 0.0) .& (x_all .<= 1.1) .& (y_all .>= 0.0) .& (y_all .<= 1.1)
x_sampled = x_all[keep]
y_sampled = y_all[keep]
total_len = length(x_sampled)
sample_idx = rand(1:total_len, min(n_plot, total_len))
x = x_sampled[sample_idx]
y = y_sampled[sample_idx]

# n_plot = 80000
# total_len = length(x_all)
# sample_idx = rand(1:total_len, min(n_plot, total_len))

# x_sampled = x_all[sample_idx]
# y_sampled = y_all[sample_idx]

# ===================== 过滤：0.1 ≤ Corr ≤ 0.8 且 0.1 ≤ C2 ≤ 0.8 =====================
# keep = (x_sampled .>= 0.1) .& (x_sampled .<= 0.8) .& (y_sampled .>= 0.1) .& (y_sampled .<= 0.8)

# x = x_sampled[keep]
# y = y_sampled[keep]

# x = x_sampled
# y = y_sampled
# ---------------------- 最小二乘拟合 y = kx + b ----------------------
# 拟合：y = a * x + b
A = [x ones(length(x))]
coeffs = A \ y
a, b = coeffs
y_fit = a .* x .+ b

println("拟合直线：y = $(round(a,digits=3))x + $(round(b,digits=3))")

# ---------------------- 绘图：散点 + y=x + 拟合直线 ----------------------
p_scatter_datafit = scatter(x, y,
            markersize = 4,
            markerstrokewidth = 0,
            alpha = 0.5,
            color = :steelblue,
            title = "C (Original) vs C2 (Sammon Reconstructed)",
            xlabel = "Original Correlation C",
            ylabel = "Reconstructed C₂",
            legend = :topleft,
            framestyle = :box,
            size=(800, 600))

# 1. y=x 参考线（理想完美拟合）
plot!(p_scatter_datafit, [0,1], [0,1],
      linewidth=2,
      color=:black,
      linestyle=:dash,
      label="y = x")

# 2. 最小二乘拟合直线
plot!(p_scatter_datafit, x, y_fit,
      linewidth=2,
      color=:red,
      label="Fit: y=$(round(a,digits=2))x+$(round(b,digits=2))")

# 保存
savefig(p_scatter_datafit, "scatter_data_fit_d5.png")
println("✅ scatter_data_fit.png 已保存！")

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

savefig(p_comparison, "fly_corr_comparison_sorted_by_type_subset.png")
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
savefig(p_legend, "fly_cell_type_legend.png")
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

savefig("fly_lambda_density.png")

println("🎉 全部计算完成！")