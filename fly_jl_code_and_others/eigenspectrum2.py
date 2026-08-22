import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

# 假设 subsampling 模块已正确导入（请确保路径）
import subsampling

# ==================== 用户需修改的部分 ====================
# 1. 重构的相关性矩阵路径（例如 BIC pipeline 输出的重构 C2，可能是 .mat 或 .csv）
recon_corr_path = "/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_weighted_synapse_count_Branson/C2_fly_branson_weighted_IRLS_dim6.csv"   # 请替换为实际路径

# 2. 原始结构连接矩阵 J 的路径（行、列为脑区/细胞，元素为连接强度）
J_path = "/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_weighted_synapse_count_Branson/StructuralMatrix_branson_weighted.csv"        # 请替换为实际路径

# 3. （可选）协方差矩阵 JJ^T 的保存路径；若为空，则自动生成在 J 同目录下
save_cov_path = "/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_weighted_synapse_count_Branson/Covariance_matrix_branson_weighted.csv"   # 留空则自动生成

# 4. 是否强制重新计算协方差（0=若已存在则读取，1=重新计算并覆盖）
force_recompute = 0

# 5. 输出图片名称
output_fig = "/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_weighted_synapse_count_Branson/cov2_fly_branson_weighted_IRLS_dim6_spectrum.png"
# ========================================================

# ----------------------
# 0. 处理协方差矩阵的读取/计算
# ----------------------
if save_cov_path == "":
    # 自动生成保存路径：在 J 文件同目录下，文件名改为 J 文件名 + "_cov.mat"
    base, ext = os.path.splitext(J_path)
    save_cov_path = base + "_cov.mat"

# 判断是否已存在且不强制重新计算
if (not force_recompute) and os.path.exists(save_cov_path):
    print(f"从已有文件读取协方差矩阵: {save_cov_path}")
    if save_cov_path.endswith('.mat'):
        cov_data = loadmat(save_cov_path)
        # 假设变量名为 'Cov'，可调整
        cov_matrix = cov_data['Cov']
    else:
        df_cov = pd.read_csv(save_cov_path, header=None) # 协方差矩阵没有labels，直接读取为 DataFrame
        cov_matrix = df_cov.to_numpy()
else:
    print(f"从结构连接矩阵 J 计算协方差矩阵: {J_path}")
    # 读取 J
    if J_path.endswith('.mat'):
        J_data = loadmat(J_path)
        # 假设变量名为 'J'，可调整
        J = J_data['J']
    else:
        df_J = pd.read_csv(J_path, index_col=0)
        J = df_J.to_numpy()
    
    # 计算协方差矩阵 JJ^T
    cov_matrix = J @ J.T
    # 强制对称（可能因数值误差）
    cov_matrix = (cov_matrix + cov_matrix.T) / 2.0
    
    # 保存协方差矩阵
    print(f"保存协方差矩阵至: {save_cov_path}")
    if save_cov_path.endswith('.mat'):
        savemat(save_cov_path, {'Cov': cov_matrix})
    else:
        # 假设 CSV，需行/列标签；若 J 有标签，可传递；这里简化，仅保存数值
        # 更好的做法是读取 J 的标签后一并保存，但此处不影响后续方差提取
        np.savetxt(save_cov_path, cov_matrix, delimiter=',')

# 提取方差和标准差
variances = np.diag(cov_matrix)
std_dev = np.sqrt(variances)

# ----------------------
# 1. 读取重构的相关性矩阵
# ----------------------
if recon_corr_path.endswith('.mat'):
    data = loadmat(recon_corr_path)
    # 假设变量名为 'C2' 或 'Corr_recon'，请根据实际情况修改
    Corr_recon = data['C2']  # 请确认变量名
else:
    df = pd.read_csv(recon_corr_path, index_col=0)
    Corr_recon = df.to_numpy()

# 确保对角线为 1
np.fill_diagonal(Corr_recon, 1.0)

# ----------------------
# 2. 计算重构的协方差矩阵
# ----------------------
std_diag = np.diag(std_dev)
Cov_recon = std_diag @ Corr_recon @ std_diag
# 强制对称（防止数值误差）
Cov_recon = (Cov_recon + Cov_recon.T) / 2.0

# ----------------------
# 3. 计算特征谱
# ----------------------
evals = np.linalg.eigvalsh(Cov_recon)
evals = np.sort(evals)[::-1]

# 子采样参数
k_fractions = [0.125, 0.25, 0.5]
n_iter = 50

# 幂律拟合：左图前5个，右图前10个
fit_5 = subsampling.fit_power_law_eigenvalues(evals, 20)
fit_10 = subsampling.fit_power_law_eigenvalues_start_end(evals,10, 80)

# ----------------------
# 4. 绘图（左右两图，仅拟合 rank 不同）
# ----------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Eigenspectrum of Covariance Matrix (IRLS dim6)', fontsize=16)

ranks = (np.arange(len(evals)) + 1) / len(evals)
colors = plt.cm.viridis(np.linspace(0, 0.8, len(k_fractions)))

for i, ax_i in enumerate(ax):
    # --- 原始谱 ---
    ax_i.loglog(ranks, evals, 'k-', linewidth=1.5, alpha=1.0, label='Original Cov')

    # --- 子采样谱 ---
    for j, k in enumerate(k_fractions):
        mean_evals, std_evals, _ = subsampling.get_subsampled_eigenspectrum(Cov_recon, k, n_iter)
        ranks_sub = (np.arange(len(mean_evals)) + 1) / len(mean_evals)
        ax_i.loglog(ranks_sub, mean_evals, color=colors[j], linewidth=1.5, label=f'k={k}',alpha=0.8)
        ax_i.fill_between(ranks_sub, mean_evals - std_evals, mean_evals + std_evals,
                          color=colors[j], alpha=0.2)

    # --- 幂律拟合（左图 fit_5，右图 fit_10） ---
    fit = fit_5 if i == 0 else fit_10
    title = 'fit first 20' if i == 0 else 'fit first 10 to 80'
    ax_i.loglog(fit['data_x'], fit['fitted_y'], 'r--', linewidth=1.5,
                label=r'Power law ($\alpha$={:.2f}, $R^2$={:.2f})'.format(fit['exponent'], fit['r_squared']))

    # --- 添加对角元（方差）灰色散点 ---
    sorted_var = np.sort(variances)[::-1]
    ax_i.scatter(ranks, sorted_var, color='gray', s=10, alpha=0.6, label='Variances (diag)')

    # --- 样式 ---
    ax_i.set_xlabel('Rank (r/N)')
    ax_i.set_ylabel('Eigenvalue / Variance')
    ax_i.set_ylim([10**-1, 10**11])          # 可根据实际数据调整
    ax_i.legend(fontsize=6)
    ax_i.set_title(title)
    ax_i.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig(output_fig, dpi=300)
print(f"✅ 特征谱图已保存: {output_fig}")