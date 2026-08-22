import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
# ---------------------- 1. 读取并强制对齐连接矩阵 ----------------------
matrix_csv_path = "/home/user/ShenRuihong/bishe/corr_principal_validation/fly/fly_connectivity_matrix_sorted_by_roi.csv"
connectivity_df = pd.read_csv(matrix_csv_path, index_col=0)

# 【关键修改】确保 index 和 columns 都是整数格式
connectivity_df.index = connectivity_df.index.astype(int)
connectivity_df.columns = connectivity_df.columns.astype(int)

# 获取当前排序后的 bodyIds（这是行的顺序）
sorted_body_ids = connectivity_df.index.values

# 【核心操作】对 DataFrame 进行重排
# 1. 行：保持现状（已经是 sorted_body_ids）
# 2. 列：强制重新排列为 sorted_body_ids 的顺序。如果某列不存在，填充 0
connectivity_df_sorted = connectivity_df.reindex(
    index=sorted_body_ids, 
    columns=sorted_body_ids, 
    fill_value=0
)

connectivity_matrix = connectivity_df_sorted.values

print(f"✅ 矩阵对齐完成")
print(f"   原始矩阵尺寸: {connectivity_df.shape}")
print(f"   对齐后矩阵尺寸: {connectivity_df_sorted.shape}")
print(f"   行与列ID是否一致: {np.array_equal(connectivity_df_sorted.index, connectivity_df_sorted.columns)}")

# ---------------------- 2. 还原 sorted_rois ----------------------
# 这里的逻辑保持不变，因为 sorted_body_ids 没变
raw_csv_path = "/home/user/ShenRuihong/bishe/corr_principal_validation/fly/traced-neurons.csv"
df_raw = pd.read_csv(raw_csv_path, usecols=["bodyId", "type"])

# 在建立 bodyId_to_type 之前，先将 type 列的空值填充为 "Unknown"
df_raw["type"] = df_raw["type"].fillna("Unknown").astype(str)
bodyId_to_type = df_raw.groupby("bodyId")["type"].first().to_dict()
sorted_types = [bodyId_to_type.get(bid, "Unknown") for bid in sorted_body_ids]
sorted_types = np.array(sorted_types)

# ---------------------- 3. 为ROI分配颜色 ----------------------
types_unique, type_idx = np.unique(sorted_types, return_inverse=True)
cmap = plt.get_cmap("tab20")
type_colors = cmap(type_idx)

# ---------------------- 4. 绘制热图 ----------------------
fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

# 统计信息 (使用对齐后的矩阵)
nonzero_weights = connectivity_matrix[connectivity_matrix > 0]
vmax_val = np.percentile(nonzero_weights, 95) if len(nonzero_weights) > 0 else 1

# 绘制热图
sns.heatmap(
    connectivity_matrix,
    cmap="gray_r",
    xticklabels=[],
    yticklabels=[],
    ax=ax,
    square=True,
    cbar_kws={"label": "Connection Weight", "shrink": 0.8},
    vmin=0,
    vmax=vmax_val
)

# ---------------------- 5. 添加侧边/顶部ROI颜色条 ----------------------
# 因为行和列现在完全一致了，所以颜色条是对称的
# 左侧颜色条 (对应 Y轴 / 行 / Post)
for i, color in enumerate(type_colors):
    ax.add_patch(plt.Rectangle(
        xy=(-0.02, i), width=0.02, height=1,
        color=color, lw=0,
        transform=ax.get_yaxis_transform(), clip_on=False
    ))

# 顶部颜色条 (对应 X轴 / 列 / Pre)
for i, color in enumerate(type_colors):
    ax.add_patch(plt.Rectangle(
        xy=(i, 1), height=0.02, width=1,
        color=color, lw=0,
        transform=ax.get_xaxis_transform(), clip_on=False
    ))

# 可选：如果你想做一个图例，可以在这里添加
plt.savefig('fly_connectivity_matrix_sorted_by_type_heatmap.png')

# ---------------------- 6. 保存排序后的连接矩阵为 .csv 文件 ----------------------
output_path = "fly_connectivity_matrix_sorted_by_type.csv"
connectivity_df_sorted.to_csv(output_path)
print(f"✅ 排序后的连接矩阵已保存至: {output_path}")

# ---------------------- 7. 按 type 合并神经元 ----------------------
# 获取唯一 type 列表和映射
unique_types = np.unique(sorted_types)
n_types = len(unique_types)

# 为每个 type 分配索引
type_to_idx = {t: i for i, t in enumerate(unique_types)}

# 初始化聚合矩阵
agg_matrix = np.zeros((n_types, n_types), dtype=connectivity_matrix.dtype)

# 累加：对于每个原矩阵中的位置 (row_idx, col_idx)
for row_idx, row_type in enumerate(sorted_types):
    for col_idx, col_type in enumerate(sorted_types):
        i = type_to_idx[row_type]
        j = type_to_idx[col_type]
        agg_matrix[i, j] += connectivity_matrix[row_idx, col_idx]

# 保存聚合矩阵为 CSV（type 作为行和列标签）
agg_df = pd.DataFrame(agg_matrix, index=unique_types, columns=unique_types)
agg_output_path = "fly_connectivity_matrix_aggregated_by_type.csv"
agg_df.to_csv(agg_output_path)
print(f"✅ 按 type 聚合的连接矩阵已保存至: {agg_output_path}")
print(f"   聚合后类型数量: {n_types}")

# 可选：绘制聚合矩阵的热图
plt.figure(figsize=(10, 8), dpi=300)
sns.heatmap(
    agg_matrix,
    cmap="gray_r",
    xticklabels=unique_types,
    yticklabels=unique_types,
    square=True,
    cbar_kws={"label": "Aggregated Connection Weight"}
)
plt.title("Aggregated Connectivity Matrix by Neuron Type")
plt.tight_layout()
plt.savefig("fly_connectivity_matrix_aggregated_by_type_heatmap.png")
print("✅ 聚合矩阵热图已保存")