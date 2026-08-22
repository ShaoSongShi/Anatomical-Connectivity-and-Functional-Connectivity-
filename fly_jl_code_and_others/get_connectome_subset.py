import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# ---------------------- 1. 读取并对齐原始完整连接矩阵 ----------------------
matrix_csv_path = "/home/user/ShenRuihong/bishe/corr_principal_validation/fly/fly_connectivity_matrix_sorted_by_roi.csv"
connectivity_df = pd.read_csv(matrix_csv_path, index_col=0)

# 强制统一index和columns为整数格式
connectivity_df.index = connectivity_df.index.astype(int)
connectivity_df.columns = connectivity_df.columns.astype(int)

# 原始排序的神经元ID
full_sorted_body_ids = connectivity_df.index.values

# 强制对齐原始矩阵的行和列，保证行列ID完全一致
connectivity_df_full_aligned = connectivity_df.reindex(
    index=full_sorted_body_ids,
    columns=full_sorted_body_ids,
    fill_value=0
)
print(f"✅ 原始矩阵对齐完成，尺寸：{connectivity_df_full_aligned.shape}")

# ---------------------- 2. 还原神经元ID-ROI的映射关系 ----------------------
raw_csv_path = "/home/user/ShenRuihong/bishe/corr_principal_validation/fly/traced-roi-connections.csv"
df_raw = pd.read_csv(raw_csv_path, usecols=["bodyId_post", "roi"])

# 建立映射字典：每个神经元ID对应唯一ROI（取首次出现的ROI）
bodyId_to_roi = df_raw.groupby("bodyId_post")["roi"].first().to_dict()

# 生成完整的ROI列表（和原始矩阵ID一一对应）
full_sorted_rois = [bodyId_to_roi.get(bid, "Unknown") for bid in full_sorted_body_ids]
full_sorted_rois = np.array(full_sorted_rois)

# ---------------------- 3. 定义目标ROI，筛选神经元子集 ----------------------
# 严格匹配你图例中的所有ROI名称
# target_rois = [
#     "AB(L)", "AB(R)",
#     "AL(L)", "AL(R)",
#     "AME(R)", "AOTU(R)",
#     "ATL(L)", "ATL(R)",
#     "AVLP(R)",
#     "BU(L)", "BU(R)",
#     "CA(R)", "CAN(R)",
#     "CRE(L)", "CRE(R)",
#     "EB",
#     "EPA(L)", "EPA(R)",
#     "FB"
# ]
target_rois = ["AL(L)", "AL(R)"]

# 筛选出属于目标ROI的神经元
filter_mask = np.isin(full_sorted_rois, target_rois)
subset_body_ids = full_sorted_body_ids[filter_mask]
subset_rois = full_sorted_rois[filter_mask]

# 校验筛选结果
print(f"\n✅ 神经元筛选完成")
print(f"   原始总神经元数：{len(full_sorted_body_ids)}")
print(f"   筛选后目标神经元数：{len(subset_body_ids)}")
print(f"   覆盖ROI数量：{len(np.unique(subset_rois))}")

# 异常处理：筛选结果为空时终止
if len(subset_body_ids) == 0:
    raise ValueError("❌ 未筛选到目标ROI的神经元，请检查ROI名称是否与原始数据完全匹配")

# ---------------------- 4. 提取子连接矩阵，保存为CSV文件 ----------------------
# 提取目标神经元的行和列，保证行列ID完全一致
connectivity_df_subset = connectivity_df_full_aligned.reindex(
    index=subset_body_ids,
    columns=subset_body_ids,
    fill_value=0
)

# 保存子矩阵到CSV，格式和原始矩阵完全一致
subset_csv_path = "fly_connectivity_matrix_target_roi_subset_AL.csv"
connectivity_df_subset.to_csv(subset_csv_path)
print(f"\n✅ 目标ROI子连接矩阵已保存至：{subset_csv_path}")

# 统计子矩阵稀疏度，验证筛选效果
subset_matrix = connectivity_df_subset.values
nonzero_weights = subset_matrix[subset_matrix > 0]
total_elements = subset_matrix.shape[0] * subset_matrix.shape[1]
sparsity = 1 - len(nonzero_weights) / total_elements
print(f"\n📊 子矩阵统计信息：")
print(f"   子矩阵尺寸：{subset_matrix.shape}")
print(f"   非零连接数量：{len(nonzero_weights)}")
print(f"   矩阵稀疏度：{sparsity:.2%}")
print(f"   非零权重最大值：{nonzero_weights.max():.2f}")
print(f"   非零权重95分位数：{np.percentile(nonzero_weights, 95):.2f}")

# ---------------------- 5. 为目标ROI分配固定颜色（严格匹配你的图例） ----------------------
# 按你图例的顺序固定ROI顺序和颜色，保证和原图配色一致
roi_color_order = target_rois
# 使用tab20配色，刚好匹配20个ROI的数量
cmap = plt.get_cmap("tab20")
roi_to_color = {roi: cmap(i) for i, roi in enumerate(roi_color_order)}

# 生成每个神经元对应的颜色
subset_roi_colors = [roi_to_color[roi] for roi in subset_rois]

# ---------------------- 6. 绘制筛选后的连接矩阵热图 ----------------------
fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

# 热图绘制参数，用95分位数限制色阶，避免极端值掩盖细节
vmax_val = np.percentile(nonzero_weights, 95) if len(nonzero_weights) > 0 else 1
sns.heatmap(
    subset_matrix,
    cmap="gray_r",
    xticklabels=[],
    yticklabels=[],
    ax=ax,
    square=True,
    cbar_kws={"label": "Connection Weight", "shrink": 0.8},
    vmin=0,
    vmax=vmax_val
)

# 添加左侧Y轴（突触后神经元）ROI颜色条
for i, color in enumerate(subset_roi_colors):
    ax.add_patch(plt.Rectangle(
        xy=(-0.02, i), width=0.02, height=1,
        color=color, lw=0,
        transform=ax.get_yaxis_transform(), clip_on=False
    ))

# 添加顶部X轴（突触前神经元）ROI颜色条
for i, color in enumerate(subset_roi_colors):
    ax.add_patch(plt.Rectangle(
        xy=(i, 1), height=0.02, width=1,
        color=color, lw=0,
        transform=ax.get_xaxis_transform(), clip_on=False
    ))

# 添加ROI图例（放在图右侧，不遮挡热图）
legend_elements = [Patch(facecolor=roi_to_color[roi], label=roi) for roi in roi_color_order]
ax.legend(
    handles=legend_elements,
    bbox_to_anchor=(1.32, 1),
    loc="upper left",
    fontsize=6,
    ncol=1,
    frameon=False
)

# 保存高清热图
plt.tight_layout()
plt.savefig('fly_connectivity_matrix_target_roi_heatmap_AL.png', bbox_inches="tight", dpi=300)
