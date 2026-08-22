import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ==============================================
# 1. 读取降维后的坐标数据
# ==============================================
# 这个文件里存的是 X, Y 坐标，不是距离矩阵
dist_path = "/home/user/ShenRuihong/bishe/corr_principal_validation/distance_matrix_stress.csv"
df_coords = pd.read_csv(dist_path, index_col=0)

print(f"✅ 数据加载完成，维度: {df_coords.shape}")
print("前5行坐标预览：")
print(df_coords.head())

# 提取坐标数组 (N x 2)
X = df_coords.values
X = X.T

# ==============================================
# 6. 肘部法选 K
# ==============================================
k_range = range(2, 11) # 尝试 K=2 到 10
inertias = []

print("正在计算肘部图...")
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    inertias.append(km.fit(X).inertia_)

# 绘制肘部图
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel("Number of Clusters (K)", fontsize=12)
plt.ylabel("Inertia", fontsize=12)
plt.title("Elbow Method")
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("/home/user/ShenRuihong/bishe/corr_principal_validation/elbow_plot.png", dpi=300)
print("✅ 肘部图已保存")
plt.close()

# ==============================================
# 7. 输入 K 进行聚类
# ==============================================
try:
    best_k = int(input("\n👉 请查看 elbow_plot.png，然后在此输入最佳聚类 K: "))
except:
    best_k = 4
    print(f"⚠️ 未检测到输入，使用默认 K={best_k}")

# 执行 K-Means
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# ==============================================
# 10. 绘制聚类散点图
# ==============================================
plt.figure(figsize=(10, 8))

# 绘制散点
plt.scatter(X[:, 0], X[:, 1], 
            c=clusters,       # 根据聚类结果上色
            cmap='tab10',     # 配色方案
            s=40,             # 点大小
            alpha=0.8,        # 透明度
            edgecolors='k',   # 黑色描边
            linewidth=0.5)

# 绘制聚类中心
plt.scatter(centers[:, 0], centers[:, 1], 
            c='black', 
            s=200, 
            marker='X',       # 叉号
            edgecolors='white', 
            linewidths=2, 
            label='Centroids')

# 美化图表
plt.title(f"K-Means Clustering (K={best_k})\nBased on Sammon Coordinates", fontsize=16)
plt.xlabel("Sammon Dimension 1", fontsize=12)
plt.ylabel("Sammon Dimension 2", fontsize=12)
plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# 保存
output_path = "/home/user/ShenRuihong/bishe/corr_principal_validation/kmeans_cluster_scatter.png"
plt.savefig(output_path, dpi=300)
print(f"✅ 聚类散点图已保存: {output_path}")
