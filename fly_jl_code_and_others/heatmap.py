import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors

# ----------------------
# 1. 读取 CSV
# ----------------------
csv_path = "/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_cellcount_Branson/StructuralMatrix_branson_cellcount.csv"
df = pd.read_csv(csv_path, index_col=0)
original_ids = df.index.tolist()

# ----------------------
# 2. 数据预处理
# ----------------------
data = df.apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
data = (data + data.T) / 2.0
np.fill_diagonal(data, 1.0)

# ----------------------
# 3. 为脑区分配颜色
# ----------------------
unique_regions = list(dict.fromkeys(original_ids))
n_regions = len(unique_regions)

def generate_distinct_colors(n, seed=42):
    np.random.seed(seed)
    colors = []
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0
        saturation = 0.75 if i % 2 == 0 else 0.95
        value = 0.85 if i % 3 == 0 else (0.70 if i % 3 == 1 else 0.55)
        rgb = mcolors.hsv_to_rgb([hue, saturation, value])
        colors.append((*rgb, 1.0))
    return colors

region_colors = generate_distinct_colors(n_regions)
region_to_color = {region: region_colors[i] for i, region in enumerate(unique_regions)}

row_colors = [region_to_color[r] for r in original_ids][::-1]
col_colors = [region_to_color[c] for c in original_ids]

# ----------------------
# 4. 绘制热图（colorbar 放在外部）
# ----------------------
fig = plt.figure(figsize=(18, 14))

# 主网格：3列（legend | 热图+colorbar | 留白），2行（col_bar | 热图+row_bar）
# 关键：热图和colorbar分开控制，不共享宽度
gs = fig.add_gridspec(2, 3, 
                      width_ratios=[2.5, 10, 1.2],   # 第3列给colorbar
                      height_ratios=[2.5, 10],
                      wspace=0.03, 
                      hspace=0.03,
                      left=0.08, right=0.95, top=0.95, bottom=0.08)

ax_legend = fig.add_subplot(gs[0, 0])
ax_col = fig.add_subplot(gs[0, 1])      # 上方脑区颜色条
ax_row = fig.add_subplot(gs[1, 0])      # 左侧脑区颜色条
ax_main = fig.add_subplot(gs[1, 1])     # 热图主体
ax_cbar = fig.add_subplot(gs[1, 2])     # 独立的colorbar轴

# 清理辅助轴
for ax in [ax_legend, ax_col, ax_row]:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# --- 上方列颜色条 ---
N = len(col_colors)
ax_col.set_xlim(0, N)
ax_col.set_ylim(0, 1)
for i, color in enumerate(col_colors):
    rect = Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='none')
    ax_col.add_patch(rect)

# --- 左侧行颜色条 ---
M = len(row_colors)
ax_row.set_xlim(0, 1)
ax_row.set_ylim(0, M)
for i, color in enumerate(row_colors):
    rect = Rectangle((0, i), 1, 1, facecolor=color, edgecolor='none')
    ax_row.add_patch(rect)

# --- 热图主体 ---
# im = ax_main.imshow(data, aspect='auto', cmap='RdBu_r', 
#                     vmin=-1, vmax=1, interpolation='nearest',
#                     extent=[0, N, 0, M])
im = ax_main.imshow(data, aspect='auto', cmap='RdBu_r', 
                    vmin=-500, vmax=500, interpolation='nearest',
                    extent=[0, N, 0, M])
ax_main.set_xticks([])
ax_main.set_yticks([])

# --- colorbar 放在独立轴（外部右侧）---
cbar = fig.colorbar(im, cax=ax_cbar)
cbar.set_label('Strength', rotation=270, labelpad=20)
ax_cbar.yaxis.set_ticks_position('right')
ax_cbar.yaxis.set_label_position('right')

# --- 图例 ---
ncol = 1 if n_regions <= 12 else (2 if n_regions <= 30 else 3)
fontsize = max(5, 9 - n_regions // 25)

legend_elements = [Patch(facecolor=region_to_color[r], edgecolor='black', 
                         linewidth=0.3, label=r) for r in unique_regions]

ax_legend.legend(handles=legend_elements, loc='center', fontsize=fontsize,
                 ncol=ncol, frameon=False, title='Brain Regions',
                 title_fontsize=fontsize + 1)

fig.suptitle('Structure Heatmap', fontsize=16, y=0.98)
plt.savefig("heatmap_structure_fly_cellcount_region_sorted.png", dpi=300, bbox_inches='tight')
print("\n✅ 热图已保存: heatmap_structure_fly_cellcount_region_sorted.png")
# plt.show()
# fig.suptitle('Structure Matrix Heatmap', fontsize=16, y=0.98)
# plt.savefig("heatmap_structure.png", dpi=300, bbox_inches='tight')
# print("\n✅ 热图已保存: heatmap_structure.png")