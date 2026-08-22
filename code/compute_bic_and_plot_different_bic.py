#!/usr/bin/env python3
"""
compute_bic_and_plot.py

读取 MATLAB 批量 Sammon MDS 结果（results.mat）与 ERM 准备数据（prepared.mat），
对每个维度：
  1. 由嵌入坐标 X 重构欧氏距离矩阵 D2；
  2. 按 ERM 反解重构相关性矩阵 C2 = eps^mu * (D2^2 + eps^2)^(-mu/2)；
  3. 在相关性域比较 Corr 与 C2，计算 NCC / MSE / MAE；
  4. 在距离域按各方法 stress 对应的（准）似然计算 BIC（公式见
     《MDS 降维的 BIC 公式与推导》）：
       Classic Sammon: BIC = n*ln(WRSS/n) + k_d*ln(n),
                       WRSS = sum w_ij r_ij^2, w_ij = 1/d_ij   （异方差高斯）
       Huber Sammon:   BIC = 2*sum w_ij * rho_c(r_ij/s_hat) + k_d*ln(n),
                       s_hat = 1.4826*MAD(r), c = 1.345        （Huber 最小有利分布）
       IRLS Sammon:    硬截断权重不对应可积密度，采用截断对比函数的 quasi-BIC：
                       仅保留收敛掩码内的 n_eff 对，
                       BIC = n_eff*ln(WRSS_eff/n_eff) + k_d*ln(n_eff)
       其中 k_d = N*d - d(d+1)/2 + 1（扣除 d 维欧氏嵌入的旋转/平移/反射
       不可识别自由度，+1 为噪声尺度参数，为常数不影响选维）。

用法：
  python3 compute_bic_and_plot.py <results.mat> <prepared.mat> [output_prefix]
"""

import sys
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

C_HUBER_BIC = 1.345   # Huber 最小有利分布阈值（95% 渐近效率）


# ---------------------------------------------------------------------------
# .mat 读取（v7 / v7.3 HDF5 自适应）
# ---------------------------------------------------------------------------
def load_mat_auto(filepath):
    try:
        from scipy.io import loadmat
        return loadmat(filepath, simplify_cells=False)
    except NotImplementedError:
        import h5py
        data = {}
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                if key.startswith('#'):
                    continue
                item = f[key]
                if isinstance(item, h5py.Dataset):
                    val = item[()]
                    if hasattr(val, 'dtype') and h5py.check_dtype(ref=val.dtype):
                        # reference 数组（MATLAB cell）：解引用
                        cell_data = []
                        for ref in val.flat:
                            cell_data.append(np.array(f[ref][()]).T)
                        data[key] = cell_data
                    else:
                        data[key] = val if val.shape == () else np.array(val).T
                elif isinstance(item, h5py.Group):
                    cell_data = []
                    subkeys = sorted(item.keys(),
                                     key=lambda x: int(x) if x.isdigit() else float('inf'))
                    for subkey in subkeys:
                        subitem = item[subkey]
                        if isinstance(subitem, h5py.Dataset):
                            arr = subitem[()]
                            if hasattr(arr, 'dtype') and h5py.check_dtype(ref=arr.dtype):
                                ref = arr.flat[0] if arr.size > 0 else None
                                if ref is not None and ref in f:
                                    cell_data.append(np.array(f[ref][()]).T)
                                else:
                                    cell_data.append(np.array(arr))
                            else:
                                cell_data.append(np.array(arr).T)
                    data[key] = cell_data
        return data


def get_cell_array(data, key, n):
    """把 .mat 中的 cell 统一成 python list。"""
    v = data[key]
    if isinstance(v, list):
        return v
    v = np.asarray(v, dtype=object)
    return [v.flat[i] for i in range(n)]


def extract_scalar(data, key, idx):
    val = np.asarray(data[key], dtype=float)
    return float(val.flat[idx])


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------
def corr_metrics(C_orig, C_recon, iu):
    """在相关性域比较（仅上三角非对角）。"""
    a = C_orig[iu]
    b = C_recon[iu]
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    x = a - a.mean()
    y = b - b.mean()
    denom = np.sqrt(np.sum(x**2) * np.sum(y**2))
    ncc = float(np.sum(x * y) / denom) if denom > 0 else np.nan
    mse = float(np.mean((a - b) ** 2))
    mae = float(np.mean(np.abs(a - b)))
    return ncc, mse, mae


def n_free_params(N, d):
    """k_d = N*d - d(d+1)/2 + 1（扣除旋转/平移/反射自由度，+1 为噪声尺度）。"""
    return N * d - d * (d + 1) // 2 + 1


def bic_classic_sammon(r, w, n_pairs, k_d):
    """经典 Sammon：异方差高斯似然，BIC = n*ln(WRSS/n) + k_d*ln(n)。"""
    wrss = np.sum(w * r**2)
    wrss = max(wrss, np.finfo(float).tiny)
    return n_pairs * np.log(wrss / n_pairs) + k_d * np.log(n_pairs)


def huber_rho(u, c):
    """Huber rho 函数。"""
    a = np.abs(u)
    return np.where(a <= c, 0.5 * u**2, c * a - 0.5 * c**2)


def bic_huber_sammon(r, w, n_pairs, k_d, c=C_HUBER_BIC):
    """Huber：BIC = 2*sum w*rho_c(r/s_hat) + k_d*ln(n)，s_hat = 1.4826*MAD。"""
    s_hat = 1.4826 * np.median(np.abs(r - np.median(r)))
    s_hat = max(s_hat, np.finfo(float).eps)
    fit_term = 2.0 * np.sum(w * huber_rho(r / s_hat, c))
    return fit_term + k_d * np.log(n_pairs)


def bic_irls_sammon(r, w, keep, k_d):
    """IRLS 硬截断：截断对比函数的 quasi-BIC，仅用收敛时保留的距离对。"""
    r_k, w_k = r[keep], w[keep]
    n_eff = len(r_k)
    if n_eff < 2:
        return np.nan
    wrss = max(np.sum(w_k * r_k**2), np.finfo(float).tiny)
    return n_eff * np.log(wrss / n_eff) + k_d * np.log(n_eff)


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def plot_metric(dims, results, metric_name, ylabel, filename, lower_is_better=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        'Classic Sammon MDS': '#2E86AB',
        'Huber Sammon MDS':   '#A23B72',
        'IRLS Sammon MDS':    '#F18F01'
    }
    markers = {
        'Classic Sammon MDS': 'o',
        'Huber Sammon MDS':   's',
        'IRLS Sammon MDS':    '^'
    }
    for name, res in results.items():
        vals = res[metric_name]
        ax.plot(dims, vals, marker=markers[name], color=colors[name],
                linewidth=2.5, markersize=8, label=name,
                markeredgecolor='white', markeredgewidth=1.5, alpha=0.5)
        finite = np.isfinite(vals)
        if not finite.any():
            continue
        idx_local = np.argmin(np.where(finite, vals, np.inf)) if lower_is_better \
            else np.argmax(np.where(finite, vals, -np.inf))
        ax.scatter([dims[idx_local]], [vals[idx_local]], s=350, c=colors[name],
                   marker='*', edgecolors='black', linewidths=2, zorder=5)
    ax.set_xlabel('Embedding Dimension', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(f'{metric_name.upper()} vs. Dimension: Sammon MDS Methods', fontsize=15)
    ax.set_xticks(dims)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(results_path, prepared_path, output_prefix):
    print(f"Loading results : {results_path}")
    print(f"Loading prepared: {prepared_path}")
    res_data = load_mat_auto(results_path)
    prep = load_mat_auto(prepared_path)

    dims = np.asarray(prep['dims']).flatten().astype(int)
    n_dims = len(dims)
    Corr = np.asarray(prep['Corr'], dtype=float)
    N = Corr.shape[0]
    D_cells = get_cell_array(prep, 'D_cells', n_dims)
    mu_vec = np.asarray(prep['mu_vec']).flatten()
    epsilon = float(np.asarray(prep['epsilon']).flatten()[0])
    print(f"N = {N}, dims = {list(dims)}, epsilon = {epsilon}")

    iu = np.triu_indices(N, 1)
    n_pairs = N * (N - 1) // 2

    methods = {
        'Classic Sammon MDS': {'coords': 'classic_coords', 'stress': 'classic_stress'},
        'Huber Sammon MDS':   {'coords': 'huber_coords',   'stress': 'huber_stress'},
        'IRLS Sammon MDS':    {'coords': 'irls_coords',    'stress': 'irls_stress',
                               'masks': 'irls_masks'},
    }

    results = {}
    for name, keys in methods.items():
        coords_list = get_cell_array(res_data, keys['coords'], n_dims)
        masks_list = (get_cell_array(res_data, keys['masks'], n_dims)
                      if 'masks' in keys else None)
        bics, stresses, nccs, mses, maes = [], [], [], [], []

        for i, d in enumerate(dims):
            X = np.asarray(coords_list[i], dtype=float)
            if X.ndim == 3:
                X = X.squeeze()
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if X.shape[0] != N:          # v7.3 转置方向兜底
                X = X.T

            # 1. 嵌入坐标 -> 欧氏距离矩阵
            D2 = squareform(pdist(X, metric='euclidean'))
            np.fill_diagonal(D2, 0.0)

            # 2. ERM 反解重构相关性矩阵 C2 = eps^mu (D2^2 + eps^2)^(-mu/2)
            mu = float(mu_vec[i])
            C2 = epsilon**mu * np.power(D2**2 + epsilon**2, -mu / 2.0)

            # 3. 相关性域指标（Corr vs C2）
            ncc, mse, mae = corr_metrics(Corr, C2, iu)
            nccs.append(ncc); mses.append(mse); maes.append(mae)

            # 4. 距离域残差 -> 按方法 stress 对应的（准）似然计算 BIC
            D_d = np.asarray(D_cells[i], dtype=float)
            D_d = (D_d + D_d.T) / 2.0
            r = D_d[iu] - D2[iu]
            w = 1.0 / np.maximum(D_d[iu], 1e-12)
            k_d = n_free_params(N, int(d))

            if name == 'Classic Sammon MDS':
                bic = bic_classic_sammon(r, w, n_pairs, k_d)
            elif name == 'Huber Sammon MDS':
                bic = bic_huber_sammon(r, w, n_pairs, k_d)
            else:
                keep = np.asarray(masks_list[i]).astype(bool)
                if keep.shape != (N, N):
                    keep = keep.T
                bic = bic_irls_sammon(r, w, keep[iu], k_d)
            bics.append(bic)
            stresses.append(extract_scalar(res_data, keys['stress'], i))

            print(f"  {name:22s} d={d:2d}: BIC={bic:14.2f} NCC={ncc:.4f} "
                  f"MSE={mse:.6f} MAE={mae:.6f}")

        results[name] = {'dims': dims, 'bics': bics, 'stresses': stresses,
                         'ncc': nccs, 'mse': mses, 'mae': maes}

    # 绘图（注意：三种方法的 BIC 基于不同噪声模型，绝对值不可横向比较，
    # 仅各曲线自身的最小值/拐点可用于选维）
    print("\nGenerating plots...")
    plot_metric(dims, results, 'bics', 'BIC  (lower is better; curves NOT cross-comparable)',
                f'{output_prefix}_bic.png', lower_is_better=True)
    plot_metric(dims, results, 'ncc',  'NCC (Corr vs reconstructed, higher is better)',
                f'{output_prefix}_ncc.png', lower_is_better=False)
    plot_metric(dims, results, 'mse',  'MSE (Corr vs reconstructed, lower is better)',
                f'{output_prefix}_mse.png', lower_is_better=True)
    plot_metric(dims, results, 'mae',  'MAE (Corr vs reconstructed, lower is better)',
                f'{output_prefix}_mae.png', lower_is_better=True)

    # 文本摘要
    summary_file = f'{output_prefix}_summary.txt'
    lines = []
    lines.append("=" * 70)
    lines.append("Sammon MDS Batch Results Summary (correlation-domain metrics)")
    lines.append("BIC: per-method noise model (classic: heteroscedastic Gaussian;")
    lines.append("Huber: least favorable distribution; IRLS: truncated quasi-BIC)")
    lines.append("=" * 70)
    for metric, label, lower in [('bics', 'BIC', True), ('ncc', 'NCC', False),
                                 ('mse', 'MSE', True), ('mae', 'MAE', True)]:
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"{label} Summary ({'lower' if lower else 'higher'} is better)")
        lines.append(f"{'Method':<25} {'Best Dim':>10} {label:>16} {'Stress':>12}")
        lines.append("-" * 70)
        for name, res in results.items():
            vals = np.array(res[metric], dtype=float)
            finite = np.isfinite(vals)
            idx = int(np.argmin(np.where(finite, vals, np.inf))) if lower \
                else int(np.argmax(np.where(finite, vals, -np.inf)))
            lines.append(f"{name:<25} {dims[idx]:>10} {vals[idx]:>16.4f} "
                         f"{res['stresses'][idx]:>12.6f}")
    text = "\n".join(lines)
    with open(summary_file, 'w') as f:
        f.write(text + "\n")
    print(f"\n{text}")
    print(f"\nSummary saved to: {summary_file}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 compute_bic_and_plot.py "
              "<sammon_results.mat> <prepared.mat> [output_prefix]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2],
         sys.argv[3] if len(sys.argv) > 3 else 'sammon')
