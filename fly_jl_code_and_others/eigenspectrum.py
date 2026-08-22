import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("__file__")), "..", "src"))
import subsampling

# ----------------------
# 1. 读取功能连接矩阵（保留重复脑区名）
# ----------------------
csv_path = "/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_aggregated_by_cell_type_nature_method/CouplingCorrelationMatrix_by_celltype_inputNormalized.csv"

df = pd.read_csv(csv_path, index_col=0)
original_ids = df.index.tolist()  # 保留原始重复名称

# ----------------------
# 1. 转为数值矩阵（空值变为 NaN）
# ----------------------
Coupling = df.apply(pd.to_numeric, errors='coerce')
FunctionalMatrix_raw = Coupling.to_numpy()
FunctionalMatrix=FunctionalMatrix_raw.copy()
# ----------------------
# 2. 关键：填充对角线为 1.0
# ----------------------
# 相关性矩阵 C_ii 必须为 1（自相关）
np.fill_diagonal(FunctionalMatrix, 1.0)

# 可选：强制对称化（消除数值误差）
# FunctionalMatrix = (FunctionalMatrix + FunctionalMatrix.T) / 2.0

# ----------------------
# 2. 归一化：tr(C) = N（或按你的需求归一化）
# ----------------------
# 注意：这只是线性缩放，不改变谱的幂律指数
# trace = np.trace(FunctionalMatrix)
# if trace > 1e-10:
#     FunctionalMatrix = FunctionalMatrix / trace

# 确保对角线为1（如果是相关系数矩阵）
# np.fill_diagonal(FunctionalMatrix, 1.0)

# ----------------------
# 3. 直接绘制功能连接矩阵的特征谱
# ----------------------
evals = np.linalg.eigvalsh(FunctionalMatrix)
evals = np.sort(evals)[::-1]
fit_corr = subsampling.fit_power_law_eigenvalues(evals, 5)
# fit_corr = subsampling.fit_power_law_eigenvalues_start_end(evals, start_rank=1, end_rank=5)
# 默认取前10个，correlation的取了前30个

k_fractions = [0.125, 0.25, 0.5]
n_iter = 50

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Eigenspectrums', fontsize=16)

# --- 左图：功能连接矩阵（直接输入） ---
ranks_orig = (np.arange(len(evals)) + 1) / len(evals)
ax[0].loglog(ranks_orig, evals, 'k-', linewidth=1.5, alpha=1.0, label='Original')

colors = plt.cm.viridis(np.linspace(0, 0.8, len(k_fractions)))

for i, k_fraction in enumerate(k_fractions):
    mean_evals, std_evals, _ = subsampling.get_subsampled_eigenspectrum(FunctionalMatrix, k_fraction, n_iter)
    ranks_sub = (np.arange(len(mean_evals)) + 1) / len(mean_evals)
    ax[0].loglog(ranks_sub, mean_evals, color=colors[i], linewidth=1.5, label='k={}'.format(k_fraction))
    ax[0].fill_between(ranks_sub, mean_evals - std_evals, mean_evals + std_evals, color=colors[i], alpha=0.2)

ax[0].loglog(fit_corr['data_x'], fit_corr['fitted_y'], 'r--', linewidth=1.5,
              label=r'Power law ($\alpha$={:.2f}, $R^2$={:.2f})'.format(fit_corr['exponent'], fit_corr['r_squared']))

ax[0].set_xlabel('Rank (r/N)')
ax[0].set_ylabel('Eigenvalue')
ax[0].set_ylim([10**-3, 10**3])  # 根据归一化后调整范围
ax[0].legend(fontsize=6)
ax[0].set_title('fit with first 5')
ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)

# --- 右图：如果你想对比，可以保留原Coupling矩阵的谱 ---
# 这里假设你想看原始未归一化的耦合矩阵谱
evals_raw = np.linalg.eigvalsh(Coupling.to_numpy())
evals_raw = np.sort(evals_raw)[::-1]
fit_raw = subsampling.fit_power_law_eigenvalues(evals_raw,10)

ranks_raw = (np.arange(len(evals_raw)) + 1) / len(evals_raw)
ax[1].loglog(ranks_raw, evals_raw, 'k-', linewidth=1.5, alpha=1.0, label='Original')
ax[1].loglog(ranks_raw, np.sort(np.diag(Coupling.to_numpy()))[::-1], 'o-', linewidth=0.5, alpha=0.7, label='Diagonal')

for i, k_fraction in enumerate(k_fractions):
    mean_evals, std_evals, _ = subsampling.get_subsampled_eigenspectrum(Coupling.to_numpy(), k_fraction, n_iter)
    ranks_sub = (np.arange(len(mean_evals)) + 1) / len(mean_evals)
    ax[1].loglog(ranks_sub, mean_evals, color=colors[i], linewidth=1.5, label='k={}'.format(k_fraction))
    ax[1].fill_between(ranks_sub, mean_evals - std_evals, mean_evals + std_evals, color=colors[i], alpha=0.2)

ax[1].loglog(fit_raw['data_x'], fit_raw['fitted_y'], 'r--', linewidth=1.5,
              label=r'Power law ($\alpha$={:.2f}, $R^2$={:.2f})'.format(fit_raw['exponent'], fit_raw['r_squared']))

ax[1].set_xlabel('Rank (r/N)')
ax[1].set_ylabel('Eigenvalue')
ax[1].set_ylim([10**-3, 10**3])
ax[1].legend(fontsize=6)
ax[1].set_title('fit with first 10')
ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)

plt.savefig("eigenvalue_spectrum_coupling_correlation.png")
print("✅ 特征谱图已保存: eigenvalue_spectrum_coupling_correlation.png")