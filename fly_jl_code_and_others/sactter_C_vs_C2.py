import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def plot_corr_scatter_from_csv(orig_csv, recon_csv, 
                               orig_label_col=0, recon_label_col=0,
                               sort_by_labels=True,
                               title='Original vs Reconstructed Correlation',
                               output_fig='scatter.png'):
    """
    从两个 CSV 文件绘制原始相关矩阵与重构相关矩阵的非对角元素散点图。

    Parameters
    ----------
    orig_csv : str
        原始耦合相关矩阵 CSV 文件路径（含行列标签）。
    recon_csv : str
        重构耦合相关矩阵 CSV 文件路径（含行列标签）。
    orig_label_col : int, default=0
        原始 CSV 中标签所在列（通常为第一列，即 0）。
    recon_label_col : int, default=0
        重构 CSV 中标签所在列。
    sort_by_labels : bool, default=True
        是否按行标签排序后再对齐（确保两个矩阵的脑区/细胞顺序一致）。
    title : str
        图标题。
    output_fig : str
        输出图片文件名。
    """
    # 读取 CSV，第一列作为行索引
    df_orig = pd.read_csv(orig_csv, index_col=orig_label_col)
    df_recon = pd.read_csv(recon_csv, index_col=recon_label_col)
    

    # 可选：按行索引排序，确保顺序一致
    if sort_by_labels:
        df_orig.sort_index(inplace=True)
        df_recon.sort_index(inplace=True)

    # 确保行和列标签完全相同（否则可能对应错位）
    if not (df_orig.index.equals(df_recon.index) and df_orig.columns.equals(df_recon.columns)):
        # 尝试只取两个 DataFrame 共有的标签（按行列交集）
        common_rows = df_orig.index.intersection(df_recon.index)
        common_cols = df_orig.columns.intersection(df_recon.columns)
        if len(common_rows) == 0 or len(common_cols) == 0:
            raise ValueError("两个矩阵没有共同的标签，无法对齐。")
        df_orig = df_orig.loc[common_rows, common_cols]
        df_recon = df_recon.loc[common_rows, common_cols]
        # 重新排序以保证顺序一致
        common_rows = common_rows.sort_values()
        common_cols = common_cols.sort_values()
        df_orig = df_orig.loc[common_rows, common_cols]
        df_recon = df_recon.loc[common_rows, common_cols]

    # 提取数值矩阵（确保顺序对齐）
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

    # 线性拟合 y = a*x + b
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    a = model.coef_[0]
    b = model.intercept_
    y_fit = model.predict(x.reshape(-1, 1))
    r2 = r2_score(y, y_fit)

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 8))

    # 散点图（半透明）
    ax.scatter(x, y, s=1, alpha=0.5, c='blue', label='Data points')

    # y = x 参考线（黑色虚线）
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, 'k--', linewidth=2, label='y = x (reference)')

    # 拟合线（红色）
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

# ==================== 使用示例 ====================
if __name__ == "__main__":
    plot_corr_scatter_from_csv(
        orig_csv='/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_aggregated_by_cell_type_nature_method/CouplingCorrelationMatrix_by_celltype_inputNormalized.csv',
        recon_csv='/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_aggregated_by_cell_type_nature_method/C2_IRLS_dim5_ct_normalized.csv',
        title='Correlation Reconstruction (IRLS, d=5)',
        output_fig='scatter_original_vs_reconstructed_normalized_irls_dim5.png'
    )