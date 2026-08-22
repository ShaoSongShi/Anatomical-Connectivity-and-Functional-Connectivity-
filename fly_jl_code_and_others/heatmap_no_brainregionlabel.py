import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------
# 1. 读取 CSV
# ----------------------
csv_path = "/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_cellcount_Branson/CouplingCorrelation_branson_cellcount.csv"
df = pd.read_csv(csv_path, index_col=0)

# ----------------------
# 2. 数据预处理
# ----------------------
data = df.apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
data = (data + data.T) / 2.0
np.fill_diagonal(data, 1.0)

# ----------------------
# 3. 绘制热图（无脑区注释）
# ----------------------
fig, ax = plt.subplots(figsize=(12, 10))

# 绘制热图
im = ax.imshow(data, aspect='auto', cmap='RdBu_r', 
               vmin=-1, vmax=1, interpolation='nearest')
# im = ax.imshow(data, aspect='auto', cmap='RdBu_r', 
#                interpolation='nearest')
# 去掉所有坐标轴标记
ax.set_xticks([])
ax.set_yticks([])
ax.spines[:].set_visible(False)   # 去掉边框

# 添加 colorbar（放在右侧）
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# cbar.set_label('Strength', rotation=270, labelpad=20)
# # 标题
# ax.set_title('Structure Matrix Heatmap', fontsize=16, pad=10)

cbar.set_label('coupling Strength', rotation=270, labelpad=20)
# 标题
ax.set_title('Coupling Correlation Matrix Heatmap', fontsize=16, pad=10)

# 保存
plt.savefig("/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_cellcount_Branson/heatmap_coupling_correlation_branson_cellcount.png", dpi=300, bbox_inches='tight')
print("\n✅ 热图已保存: /home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_cellcount_Branson/heatmap_coupling_correlation_branson_cellcount.png")