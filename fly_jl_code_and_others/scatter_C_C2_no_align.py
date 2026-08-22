import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def plot_corr_scatter_from_csv(orig_csv, recon_csv, 
                               orig_label_col=0, recon_label_col=0,
                               title='Original vs Reconstructed Correlation',
                               output_fig='scatter.png'):
    """
    从两个 CSV 文件绘制原始相关矩阵与重构相关矩阵的非对角元素散点图。
    假设两个矩阵的行列标签完全相同且顺序一致，直接使用。
    """
    # 读取 CSV，第一列作为行索引
    df_orig = pd.read_csv(orig_csv, index_col=orig_label_col)
    df_recon = pd.read_csv(recon_csv, index_col=recon_label_col)

    # 检查形状是否一致
    if df_orig.shape != df_recon.shape:
        raise ValueError(f"两个矩阵形状不一致: orig {df_orig.shape}, recon {df_recon.shape}")

    # 提取数值矩阵
    C = df_orig.to_numpy(dtype=float)
    C2 = df_recon.to_numpy(dtype=float)

    # 取上三角非对角元素 (i < j)
    triu_idx = np.triu_indices_from(C, k=1)
    x = C[triu_idx]
    y = C2[triu_idx]

    # 去除 NaN 或 Inf
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        raise ValueError("没有有效的非对角元素可用于绘图。")

    # 线性拟合
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    a = model.coef_[0]
    b = model.intercept_
    y_fit = model.predict(x.reshape(-1, 1))
    r2 = r2_score(y, y_fit)

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, s=1, alpha=0.5, c='blue', label='Data points')
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, 'k--', linewidth=2, label='y = x (reference)')
    ax.plot(x, y_fit, 'r-', linewidth=2,
            label=f'Fit: y = {a:.3f}x + {b:.3f} (R² = {r2:.3f})')
    ax.set_xlabel('Original Correlation')
    ax.set_ylabel('Reconstructed Correlation')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(output_fig, dpi=300)
    plt.close()
    print(f"散点图已保存为: {output_fig}")

if __name__ == "__main__":
    plot_corr_scatter_from_csv(
        orig_csv='/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_weighted_synapse_count_Branson/CouplingCorrelation_weighted.csv',
        recon_csv='/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_weighted_synapse_count_Branson/C2_fly_branson_weighted_IRLS_dim6.csv',
        title='Correlation Reconstruction (IRLS, d=6)',
        output_fig='scatter_original_vs_reconstructed_IRLS_dim6.png'
    )