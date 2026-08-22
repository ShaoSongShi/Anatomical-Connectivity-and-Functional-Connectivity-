import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

# ----------------------
# 1. 读取数据
# ----------------------
df = pd.read_csv("/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_cellcount_Branson/StructuralMatrix_branson_cellcount.csv", index_col=0)

J = df.apply(pd.to_numeric, errors='coerce').fillna(0).values
N = J.shape[0]
print(f"神经元数量 N = {N}, J形状 = {J.shape}")

# 确保方阵
assert J.shape[0] == J.shape[1], "J必须为方阵"

# ----------------------
# 2. 方法1：原有计算方式
# ----------------------
C0 = J @ J.T
diag_C0 = np.diag(C0).copy()
diag_C0[diag_C0 < 1e-10] = 1e-10
D_inv_sqrt = np.diag(1.0 / np.sqrt(diag_C0))
c_J = D_inv_sqrt @ C0 @ D_inv_sqrt
c_J[c_J < 1e-6] = 1e-12

# 提取非对角线上三角用于统计
upper_tri = np.triu_indices_from(c_J, k=1)
vals_1 = c_J[upper_tri]

# # ----------------------
# # 3. 方法2：带正则化参数 alpha 的线性响应协方差
# # ----------------------
# I = np.eye(N)

# # 计算谱半径
# eigvals = np.linalg.eigvals(J)
# rho_J = np.max(np.abs(eigvals))
# print(f"J 的谱半径 ρ(J) = {rho_J:.6f}")

# # 定义 alpha 值：确保 alpha * rho_J < 1
# # 方案A: 保守值 (0.5 / rho_J)
# # 方案B: 中等值 (0.9 / rho_J)
# # 方案C: 若 rho_J < 1，也可直接用 alpha=1（但通常不推荐）
# alpha_values = [0.3 / rho_J, 0.5 / rho_J, 0.7 / rho_J, 0.9 / rho_J]
# alpha_labels = [f"α={a:.4f}" for a in alpha_values]

# # 存储结果
# c_J_alts = {}
# pearson_corrs = {}

# for alpha in alpha_values:
#     # 计算 (I - alpha*J)^{-1}
#     M = I - alpha * J
#     try:
#         M_inv = np.linalg.inv(M)
#         c_J_alt = M_inv @ M_inv.T
#         label = f"α={alpha:.4f}"
#         c_J_alts[label] = c_J_alt
        
#         # 计算与 Method 1 的 Pearson 相关
#         vals_2 = c_J_alt[upper_tri]
#         corr = np.corrcoef(vals_1, vals_2)[0, 1]
#         pearson_corrs[label] = corr
#         print(f"  {label}: 与Method 1的Pearson r = {corr:.4f}, 值域 = [{np.min(c_J_alt):.2e}, {np.max(c_J_alt):.2e}]")
#     except np.linalg.LinAlgError:
#         print(f"  α={alpha:.4f}: 矩阵奇异，跳过")

# # 选择最佳 alpha（与 Method 1 相关性最高，或根据理论选择）
# best_alpha_label = max(pearson_corrs, key=pearson_corrs.get)
# c_J_alt_best = c_J_alts[best_alpha_label]
# print(f"\n✅ 最佳正则化参数: {best_alpha_label} (r = {pearson_corrs[best_alpha_label]:.4f})")

# # ----------------------
# # 4. 方法3：截断 Neumann 级数（求逆的替代方案）
# # ----------------------
# # 计算 I + J + J^2 + ... + J^k，避免求逆
# k_trunc = 3  # 截断阶数
# M_series = I.copy()
# J_power = J.copy()
# for k in range(1, k_trunc + 1):
#     M_series += J_power
#     J_power = J_power @ J

# c_J_series = M_series @ M_series.T
# vals_series = c_J_series[upper_tri]
# corr_series = np.corrcoef(vals_1, vals_series)[0, 1]
# print(f"\n方法3 (截断级数 k={k_trunc}): 与Method 1的Pearson r = {corr_series:.4f}")

# ----------------------
# 5. 保存结果
# ----------------------
result_df = pd.DataFrame(c_J, index=df.index, columns=df.columns)
result_df.to_csv("/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_cellcount_Branson/CouplingCorrelation_branson_cellcount.csv", encoding='utf-8')
print("\n✅ 方法1结果已保存: /home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_cellcount_Branson/CouplingCorrelation_branson_cellcount.csv")
# result_df_alt = pd.DataFrame(c_J_alt_best, index=df.index, columns=df.columns)
# result_df_alt.to_csv(f"CouplingCorrelationMatrix_by_celltype_method2_alpha_{best_alpha_label}.csv", encoding='utf-8')

# # ----------------------
# # 6. 读取ROI信息
# # ----------------------
# raw_csv_path = "/home/user/ShenRuihong/bishe/corr_principal_validation/fly/traced-roi-connections.csv"
# df_raw = pd.read_csv(raw_csv_path, usecols=["bodyId_post", "roi"])
# bodyId_to_roi = pd.Series(df_raw["roi"].values, index=df_raw["bodyId_post"]).to_dict()

# sorted_body_ids = result_df.index.values.astype(str)
# sorted_rois = [bodyId_to_roi.get(bid, "Unknown") for bid in sorted_body_ids]
# sorted_rois = np.array(sorted_rois)

# rois_unique, roi_idx = np.unique(sorted_rois, return_inverse=True)
# cmap = plt.get_cmap("tab20")
# roi_colors = cmap(roi_idx)

# # ----------------------
# # 7. 绘制综合对比图
# # ----------------------
# plt.style.use('seaborn-v0_8-whitegrid')

# fig = plt.figure(figsize=(22, 16))

# # 布局: 3行，第一行2个，第二行3个，第三行3个
# gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# # --- (0,0) 方法1 热图 ---
# ax1 = fig.add_subplot(gs[0, 0])
# sns.heatmap(c_J, ax=ax1, cmap='RdBu_r', center=0, cbar_kws={'label': 'Correlation'}, square=True, xticklabels=False, yticklabels=False)
# ax1.set_title("Method 1: $D^{-1/2}JJ^TD^{-1/2}$", fontsize=11)
# for i, color in enumerate(roi_colors):
#     ax1.add_patch(plt.Rectangle(xy=(-0.02, i), width=0.02, height=1, color=color, lw=0, transform=ax1.get_yaxis_transform(), clip_on=False))
#     ax1.add_patch(plt.Rectangle(xy=(i, 1), height=0.02, width=1, color=color, lw=0, transform=ax1.get_xaxis_transform(), clip_on=False))

# # --- (0,1) 方法2（最佳alpha）热图 ---
# ax2 = fig.add_subplot(gs[0, 1])
# vmax2 = np.max(np.abs(c_J_alt_best))
# sns.heatmap(c_J_alt_best, ax=ax2, cmap='RdBu_r', center=0, vmin=-vmax2, vmax=vmax2, cbar_kws={'label': 'Value'}, square=True, xticklabels=False, yticklabels=False)
# ax2.set_title(f"Method 2: $(I-{best_alpha_label}J)^{{-1}}(I-{best_alpha_label}J^T)^{{-1}}$", fontsize=11)
# for i, color in enumerate(roi_colors):
#     ax2.add_patch(plt.Rectangle(xy=(-0.02, i), width=0.02, height=1, color=color, lw=0, transform=ax2.get_yaxis_transform(), clip_on=False))
#     ax2.add_patch(plt.Rectangle(xy=(i, 1), height=0.02, width=1, color=color, lw=0, transform=ax2.get_xaxis_transform(), clip_on=False))

# # --- (0,2) 差异热图 ---
# ax3 = fig.add_subplot(gs[0, 2])
# diff = c_J - c_J_alt_best
# vmax_diff = np.max(np.abs(diff))
# sns.heatmap(diff, ax=ax3, cmap='RdBu_r', center=0, vmin=-vmax_diff, vmax=vmax_diff, cbar_kws={'label': 'Difference'}, square=True, xticklabels=False, yticklabels=False)
# ax3.set_title(f"Difference (Method 1 − Method 2)", fontsize=11)
# for i, color in enumerate(roi_colors):
#     ax3.add_patch(plt.Rectangle(xy=(-0.02, i), width=0.02, height=1, color=color, lw=0, transform=ax3.get_yaxis_transform(), clip_on=False))
#     ax3.add_patch(plt.Rectangle(xy=(i, 1), height=0.02, width=1, color=color, lw=0, transform=ax3.get_xaxis_transform(), clip_on=False))

# # --- (0,3) alpha-相关性曲线 ---
# ax4 = fig.add_subplot(gs[0, 3])
# alphas_plot = [float(k.split('=')[1]) for k in pearson_corrs.keys()]
# corrs_plot = list(pearson_corrs.values())
# ax4.plot(alphas_plot, corrs_plot, 'o-', color='darkblue', linewidth=2, markersize=8)
# ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
# ax4.set_xlabel(r'$\alpha$', fontsize=12)
# ax4.set_ylabel('Pearson r with Method 1', fontsize=12)
# ax4.set_title(r'$\alpha$ Sensitivity', fontsize=12)
# ax4.grid(True, alpha=0.3)

# # --- 第二行：分布对比 ---
# ax5 = fig.add_subplot(gs[1, 0])
# ax5.hist(vals_1, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
# ax5.set_title("Method 1 Distribution", fontsize=11)
# ax5.set_xlabel("Value")

# ax6 = fig.add_subplot(gs[1, 1])
# vals_2_best = c_J_alt_best[upper_tri]
# ax6.hist(vals_2_best, bins=100, color='coral', edgecolor='black', alpha=0.7)
# ax6.set_title(f"Method 2 ({best_alpha_label}) Distribution", fontsize=11)
# ax6.set_xlabel("Value")

# ax7 = fig.add_subplot(gs[1, 2])
# ax7.hist(vals_series, bins=100, color='lightgreen', edgecolor='black', alpha=0.7)
# ax7.set_title(f"Method 3 (Truncated k={k_trunc}) Distribution", fontsize=11)
# ax7.set_xlabel("Value")

# # 散点对比
# ax8 = fig.add_subplot(gs[1, 3])
# n_sample = min(5000, len(vals_1))
# sample_idx = np.random.choice(len(vals_1), size=n_sample, replace=False)
# ax8.scatter(vals_1[sample_idx], vals_2_best[sample_idx], alpha=0.3, s=2, c='purple')
# min_val = min(np.min(vals_1), np.min(vals_2_best))
# max_val = max(np.max(vals_1), np.max(vals_2_best))
# ax8.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='y=x')
# ax8.set_xlabel("Method 1")
# ax8.set_ylabel(f"Method 2 ({best_alpha_label})")
# ax8.set_title("Scatter Comparison")
# ax8.legend()

# # --- 第三行：不同alpha的热图对比 ---
# for idx, (label, c_J_alt_i) in enumerate(list(c_J_alts.items())[:3]):
#     ax = fig.add_subplot(gs[2, idx])
#     vmax_i = np.max(np.abs(c_J_alt_i))
#     sns.heatmap(c_J_alt_i, ax=ax, cmap='RdBu_r', center=0, vmin=-vmax_i, vmax=vmax_i, 
#                 cbar_kws={'label': 'Value'}, square=True, xticklabels=False, yticklabels=False)
#     ax.set_title(f"Method 2 ({label})\nr = {pearson_corrs[label]:.3f}", fontsize=10)
#     for i, color in enumerate(roi_colors):
#         ax.add_patch(plt.Rectangle(xy=(-0.02, i), width=0.02, height=1, color=color, lw=0, transform=ax.get_yaxis_transform(), clip_on=False))
#         ax.add_patch(plt.Rectangle(xy=(i, 1), height=0.02, width=1, color=color, lw=0, transform=ax.get_xaxis_transform(), clip_on=False))

# # 截断级数热图
# ax_last = fig.add_subplot(gs[2, 3])
# vmax_s = np.max(np.abs(c_J_series))
# sns.heatmap(c_J_series, ax=ax_last, cmap='RdBu_r', center=0, vmin=-vmax_s, vmax=vmax_s,
#             cbar_kws={'label': 'Value'}, square=True, xticklabels=False, yticklabels=False)
# ax_last.set_title(f"Method 3 (Truncated k={k_trunc})\nr = {corr_series:.3f}", fontsize=10)
# for i, color in enumerate(roi_colors):
#     ax_last.add_patch(plt.Rectangle(xy=(-0.02, i), width=0.02, height=1, color=color, lw=0, transform=ax_last.get_yaxis_transform(), clip_on=False))
#     ax_last.add_patch(plt.Rectangle(xy=(i, 1), height=0.02, width=1, color=color, lw=0, transform=ax_last.get_xaxis_transform(), clip_on=False))

# plt.savefig("fly_c_J_comparison_regularized.png", dpi=300, bbox_inches='tight')
# print("\n✅ 综合对比图已保存: fly_c_J_comparison_regularized.png")

# # ----------------------
# # 8. 统计信息
# # ----------------------
# print("\n" + "=" * 70)
# print("统计信息对比")
# print("=" * 70)
# print(f"{'指标':<35} {'Method 1':<15} {f'Method 2 ({best_alpha_label})':<20}")
# print("-" * 70)
# print(f"{'矩阵维度':<35} {str(c_J.shape):<15} {str(c_J_alt_best.shape):<20}")
# print(f"{'最大值':<35} {np.max(c_J):<15.6f} {np.max(c_J_alt_best):<20.6f}")
# print(f"{'最小值':<35} {np.min(c_J):<15.6f} {np.min(c_J_alt_best):<20.6f}")
# print(f"{'均值 (非对角线)':<35} {np.mean(vals_1):<15.6f} {np.mean(vals_2_best):<20.6f}")
# print(f"{'标准差 (非对角线)':<35} {np.std(vals_1):<15.6f} {np.std(vals_2_best):<20.6f}")
# print(f"{'Frobenius范数':<35} {np.linalg.norm(c_J):<15.6f} {np.linalg.norm(c_J_alt_best):<20.6f}")
# print(f"{'与Method 1的Pearson r':<35} {'1.0000':<15} {pearson_corrs[best_alpha_label]:<20.4f}")
# print("=" * 70)